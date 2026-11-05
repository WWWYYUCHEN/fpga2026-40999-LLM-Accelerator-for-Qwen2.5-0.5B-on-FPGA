/* Qwen byte BPE, NFC normalization, and the model's ordered pre-tokenizer. */
#include "tokenizer.h"
#include "vendor/utf8proc/utf8proc.h"
#include <stdlib.h>
#include <string.h>

#define HASH_SIZE (1U << 19)
struct record { uint32_t offset; uint16_t length, flags; };
struct merge { uint32_t left, right, output; };
static const struct record *records;
static const struct merge *merges;
static const unsigned char *blob;
static uint32_t vocab_count, merge_count;
static uint32_t pair_hash[HASH_SIZE], byte_ids[256];
static uint32_t pieces[QT_TEXT_LIMIT * 4];
static const char *last_error = "not initialized";

static int fail(const char *message) { last_error = message; return -1; }
const char *qt_error(void) { return last_error; }
unsigned qt_vocab_size(void) { return vocab_count; }
static uint32_t read32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static uint32_t checksum(const unsigned char *p, size_t bytes) {
    uint32_t crc = ~0U;
    while (bytes--) {
        crc ^= *p++;
        for (unsigned bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xedb88320U & (0U - (crc & 1U)));
    }
    return ~crc;
}
static unsigned pair_slot(uint32_t left, uint32_t right) {
    uint32_t h = (left * 0x9e3779b1U) ^ (right * 0x85ebca77U);
    h ^= h >> 16;
    return h & (HASH_SIZE - 1);
}
static uint32_t pair_rank(uint32_t left, uint32_t right) {
    unsigned slot = pair_slot(left, right);
    while (pair_hash[slot]) {
        uint32_t rank = pair_hash[slot] - 1;
        if (merges[rank].left == left && merges[rank].right == right) return rank;
        slot = (slot + 1) & (HASH_SIZE - 1);
    }
    return UINT32_MAX;
}

int qt_init(const void *data, size_t bytes) {
    const unsigned char *p = data;
    vocab_count = 0;
    if (((uintptr_t)data & 3U) || bytes < 32 || memcmp(p, "QWNBPE01", 8)) return fail("bad tokenizer header");
    uint32_t count = read32(p + 8), blob_bytes = read32(p + 12), pairs = read32(p + 16);
    uint64_t payload = (uint64_t)count * 8 + blob_bytes + (uint64_t)pairs * 12;
    if (count > 151936 || count < 256 || pairs > 200000 || payload + 32 != bytes ||
        payload != read32(p + 20) || read32(p + 28) != 1 || (blob_bytes & 3U))
        return fail("invalid tokenizer sizes");
    if (checksum(p + 32, bytes - 32) != read32(p + 24)) return fail("tokenizer CRC mismatch");
    records = (const struct record *)(p + 32);
    blob = p + 32 + count * 8;
    merges = (const struct merge *)(blob + blob_bytes);
    memset(pair_hash, 0, sizeof(pair_hash));
    for (unsigned i = 0; i < 256; ++i) byte_ids[i] = UINT32_MAX;
    for (uint32_t i = 0; i < count; ++i) {
        if ((uint64_t)records[i].offset + records[i].length > blob_bytes || !records[i].length)
            return fail("invalid token record");
        if (!(records[i].flags & 1) && records[i].length == 1)
            byte_ids[blob[records[i].offset]] = i;
    }
    for (unsigned i = 0; i < 256; ++i)
        if (byte_ids[i] == UINT32_MAX) return fail("missing byte token");
    for (uint32_t rank = 0; rank < pairs; ++rank) {
        struct merge m = merges[rank];
        if (m.left >= count || m.right >= count || m.output >= count) return fail("invalid merge ID");
        const struct record *a = records + m.left, *b = records + m.right, *c = records + m.output;
        if (a->length + b->length != c->length ||
            memcmp(blob + a->offset, blob + c->offset, a->length) ||
            memcmp(blob + b->offset, blob + c->offset + a->length, b->length))
            return fail("invalid merge bytes");
        unsigned slot = pair_slot(m.left, m.right);
        while (pair_hash[slot]) slot = (slot + 1) & (HASH_SIZE - 1);
        pair_hash[slot] = rank + 1;
    }
    vocab_count = count;
    merge_count = pairs;
    last_error = "OK";
    return 0;
}

struct point { utf8proc_int32_t cp; unsigned offset; unsigned char letter, number, space; };
static struct point points[QT_TEXT_LIMIT * 4 + 1];
static int is_space(utf8proc_int32_t c) {
    return (c >= 9 && c <= 13) || c == 32 || c == 0x85 || c == 0xa0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f ||
           c == 0x205f || c == 0x3000;
}
static int newline(unsigned i) { return points[i].cp == '\r' || points[i].cp == '\n'; }
static unsigned ascii_lower(unsigned c) {
    if (c >= 'A' && c <= 'Z') return c + 32;
    return c;
}
/* Ordered alternatives of Qwen's regex, applied over Unicode codepoints. */
static unsigned next_piece(unsigned i, unsigned n) {
    if (points[i].cp == '\'' && i + 1 < n) {
        unsigned c = ascii_lower((unsigned)points[i + 1].cp);
        if (c == 's' || c == 't' || c == 'm' || c == 'd') return i + 2;
        if (i + 2 < n) {
            unsigned d = ascii_lower((unsigned)points[i + 2].cp);
            if ((c == 'r' && d == 'e') || (c == 'v' && d == 'e') || (c == 'l' && d == 'l')) return i + 3;
        }
    }
    unsigned j = i;
    if (!newline(j) && !points[j].letter && !points[j].number) ++j;
    if (j < n && points[j].letter) {
        while (j < n && points[j].letter) ++j;
        return j;
    }
    if (points[i].letter) {
        j = i + 1;
        while (j < n && points[j].letter) ++j;
        return j;
    }
    if (points[i].number) return i + 1;
    j = i;
    if (points[j].cp == ' ') ++j;
    unsigned begin = j;
    while (j < n && !points[j].space && !points[j].letter && !points[j].number) ++j;
    if (j > begin) {
        while (j < n && newline(j)) ++j;
        return j;
    }
    if (points[i].space) {
        unsigned end = i, last_newline = i;
        while (end < n && points[end].space) {
            if (newline(end)) last_newline = end + 1;
            ++end;
        }
        if (last_newline > i) return last_newline;
        if (end == n) return end;
        /* Greedy whitespace lookahead retains the final space for the next word. */
        if (end - i > 1) return end - 1;
        return end;
    }
    return i + 1;
}

static int bpe(const unsigned char *text, unsigned bytes, uint32_t *out, unsigned cap, unsigned *count) {
    if (bytes > sizeof(pieces) / sizeof(pieces[0])) return fail("normalized piece too long");
    unsigned n = bytes;
    for (unsigned i = 0; i < n; ++i) pieces[i] = byte_ids[text[i]];
    while (n > 1) {
        uint32_t best = UINT32_MAX;
        unsigned at = 0;
        for (unsigned i = 0; i + 1 < n; ++i) {
            uint32_t rank = pair_rank(pieces[i], pieces[i + 1]);
            if (rank < best) { best = rank; at = i; }
        }
        if (best == UINT32_MAX) break;
        if (best >= merge_count) return fail("bad merge rank");
        pieces[at] = merges[best].output;
        memmove(pieces + at + 1, pieces + at + 2, (n - at - 2) * sizeof(*pieces));
        --n;
    }
    if (n > cap - *count) return fail("prompt exceeds token capacity");
    memcpy(out + *count, pieces, n * sizeof(*pieces));
    *count += n;
    return 0;
}

static int encode_plain(const unsigned char *text, size_t bytes, uint32_t *out, unsigned cap, unsigned *count) {
    utf8proc_uint8_t *normalized = NULL;
    utf8proc_ssize_t length = utf8proc_map(text, (utf8proc_ssize_t)bytes, &normalized,
                                          UTF8PROC_STABLE | UTF8PROC_COMPOSE);
    if (length < 0) return fail("invalid UTF-8 or NFC allocation failure");
    unsigned n = 0, at = 0;
    while (at < (unsigned)length) {
        if (n >= sizeof(points) / sizeof(points[0]) - 1) { free(normalized); return fail("too many Unicode points"); }
        utf8proc_int32_t cp;
        utf8proc_ssize_t used = utf8proc_iterate(normalized + at, length - at, &cp);
        if (used <= 0) { free(normalized); return fail("invalid normalized UTF-8"); }
        utf8proc_category_t cat = utf8proc_category(cp);
        points[n].cp = cp;
        points[n].offset = at;
        points[n].letter = cat >= UTF8PROC_CATEGORY_LU && cat <= UTF8PROC_CATEGORY_LO;
        points[n].number = cat >= UTF8PROC_CATEGORY_ND && cat <= UTF8PROC_CATEGORY_NO;
        points[n].space = (unsigned char)is_space(cp);
        ++n;
        at += (unsigned)used;
    }
    points[n].offset = (unsigned)length;
    int status = 0;
    for (unsigned i = 0; i < n;) {
        unsigned end = next_piece(i, n);
        status = bpe(normalized + points[i].offset, points[end].offset - points[i].offset, out, cap, count);
        if (status) break;
        i = end;
    }
    free(normalized);
    return status;
}

int qt_encode(const char *utf8, uint32_t *ids, unsigned capacity, unsigned *count) {
    *count = 0;
    if (!vocab_count) return fail("tokenizer not initialized");
    size_t bytes = strlen(utf8);
    if (bytes > QT_TEXT_LIMIT) return fail("text exceeds 4096 UTF-8 bytes");
    size_t begin = 0, at = 0;
    while (at < bytes) {
        uint32_t found = UINT32_MAX;
        unsigned longest = 0;
        for (uint32_t id = 151643; id < vocab_count; ++id) {
            const struct record *r = records + id;
            if ((r->flags & 1) && r->length <= bytes - at && r->length > longest &&
                !memcmp(utf8 + at, blob + r->offset, r->length)) {
                found = id; longest = r->length;
            }
        }
        if (found == UINT32_MAX) { ++at; continue; }
        if (encode_plain((const unsigned char *)utf8 + begin, at - begin, ids, capacity, count)) return -1;
        if (*count >= capacity) return fail("prompt exceeds token capacity");
        ids[(*count)++] = found;
        at += longest;
        begin = at;
    }
    int status = encode_plain((const unsigned char *)utf8 + begin, bytes - begin, ids, capacity, count);
    if (!status) last_error = "OK";
    return status;
}

int qt_chat_encode(const char *utf8, uint32_t *ids, unsigned capacity, unsigned *count) {
    static const char prefix[] = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n";
    static const char suffix[] = "<|im_end|>\n<|im_start|>assistant\n";
    char formatted[QT_TEXT_LIMIT + 1];
    size_t bytes = strlen(utf8);
    if (bytes + sizeof(prefix) - 1 + sizeof(suffix) - 1 > QT_TEXT_LIMIT) return fail("chat template plus text exceeds 4096 bytes");
    memcpy(formatted, prefix, sizeof(prefix) - 1);
    memcpy(formatted + sizeof(prefix) - 1, utf8, bytes);
    memcpy(formatted + sizeof(prefix) - 1 + bytes, suffix, sizeof(suffix));
    return qt_encode(formatted, ids, capacity, count);
}

int qt_decode(uint32_t id, const unsigned char **bytes, unsigned *length, int *special) {
    if (id >= vocab_count) return fail("token ID has no tokenizer entry");
    *bytes = blob + records[id].offset;
    *length = records[id].length;
    *special = (records[id].flags & 2) != 0;
    return 0;
}
