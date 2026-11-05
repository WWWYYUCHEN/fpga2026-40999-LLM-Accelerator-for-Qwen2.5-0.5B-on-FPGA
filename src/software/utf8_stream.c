/* Preserve complete codepoints across byte-BPE token boundaries. */
#include "utf8_stream.h"
static const unsigned char replacement[] = {0xef, 0xbf, 0xbd};
void qutf8_init(struct qutf8_stream *s, qutf8_sink sink, void *context) {
    s->used = s->expected = 0; s->sink = sink; s->context = context;
}
static void emit_bad(struct qutf8_stream *s) {
    s->sink(replacement, sizeof(replacement), s->context);
    s->used = s->expected = 0;
}
void qutf8_push(struct qutf8_stream *s, const unsigned char *bytes, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        unsigned char c = bytes[i];
        if (s->used) {
            int valid = c >= 0x80 && c <= 0xbf;
            if (s->used == 1) {
                unsigned char lead = s->pending[0];
                if ((lead == 0xe0 && c < 0xa0) || (lead == 0xed && c > 0x9f) ||
                    (lead == 0xf0 && c < 0x90) || (lead == 0xf4 && c > 0x8f)) valid = 0;
            }
            if (!valid) { emit_bad(s); --i; continue; }
            s->pending[s->used++] = c;
            if (s->used == s->expected) {
                s->sink(s->pending, s->used, s->context);
                s->used = s->expected = 0;
            }
        } else if (c < 0x80) {
            s->sink(&c, 1, s->context);
        } else if (c >= 0xc2 && c <= 0xf4) {
            s->pending[0] = c; s->used = 1;
            s->expected = c < 0xe0 ? 2 : (c < 0xf0 ? 3 : 4);
        } else emit_bad(s);
    }
}
void qutf8_finish(struct qutf8_stream *s) { if (s->used) emit_bad(s); }
