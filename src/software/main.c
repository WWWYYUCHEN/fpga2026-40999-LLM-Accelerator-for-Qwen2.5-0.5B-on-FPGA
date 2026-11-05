/* KV260 UART interactive inference. Model math is exclusively executed by PL. */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ff.h"
#include "sleep.h"
#include "xil_cache.h"
#include "xil_io.h"
#include "xiltimer.h"
#include "bspconfig.h"
#include "xuartps.h"
#include "../board_parameters.h"
#include "tokenizer.h"
#include "utf8_stream.h"
#include "generation.h"

static FATFS filesystem;
static FIL result_file;
static XUartPs uart;
static unsigned char tokenizer_data[QT_FILE_LIMIT] __attribute__((aligned(8)));
static char input[QT_TEXT_LIMIT + 1];
static uint32_t prompt[256];
static unsigned max_new = 20, request_id, output_mode, raw_template;
static int skip_lf, log_active;
static struct qutf8_stream decoder;
extern int qwen_board_prepare(void);

static void barrier(void) {
#ifndef QWEN_HOST_SIM
    __asm__ volatile("dsb sy" ::: "memory");
#endif
}
static void halt(void) {
#ifdef QWEN_HOST_SIM
    exit(1);
#else
    for (;;) __asm__ volatile("wfe");
#endif
}
static void csr_write(void *context, unsigned offset, uint64_t value) {
    (void)context;
    Xil_Out64((UINTPTR)(QWEN_CSR_BASE + offset), value);
    barrier();
}
static uint64_t csr_read(void *context, unsigned offset) {
    (void)context; barrier();
    return Xil_In64((UINTPTR)(QWEN_CSR_BASE + offset));
}
static uint64_t timer_ticks(void *context) {
    XTime value; (void)context; XTime_GetTime(&value); return value;
}
static void wait_tick(void *context) { (void)context; usleep(100); }
static void console_bytes(const unsigned char *bytes, size_t count, void *context) {
    (void)context;
    for (size_t i = 0; i < count; ++i) XUartPs_SendByte(STDOUT_BASEADDRESS, bytes[i]);
}
static int read_line(void) {
    size_t used = 0;
    int overflow = 0;
    for (;;) {
        unsigned char c = XUartPs_RecvByte(STDIN_BASEADDRESS);
        if (skip_lf) { skip_lf = 0; if (c == '\n') continue; }
        if (c == '\r' || c == '\n') {
            skip_lf = c == '\r';
            input[used] = '\0';
            printf("\r\n");
            return overflow ? -1 : (int)used;
        }
        if (c == 3) { printf("^C\r\n"); input[0] = 0; return 0; }
        if (c == 8 || c == 127) {
            if (used && !overflow) {
                do { --used; } while (used && ((unsigned char)input[used] & 0xc0U) == 0x80U);
                printf("\b \b");
            }
            continue;
        }
        if (c < 32 && c != '\t') continue;
        if (used == QT_TEXT_LIMIT) { overflow = 1; continue; }
        if (!overflow) {
            input[used++] = (char)c;
            XUartPs_SendByte(STDOUT_BASEADDRESS, c);
        }
    }
}

static int load_asset(const struct asset_entry *asset) {
    FIL file;
    if (f_open(&file, asset->filename, FA_READ) != FR_OK) {
        printf("ERROR cannot open %s\r\n", asset->filename); return -1;
    }
    if ((uint64_t)f_size(&file) != asset->bytes) {
        printf("ERROR wrong size %s\r\n", asset->filename); f_close(&file); return -1;
    }
    printf("Loading %s (%" PRIu64 " bytes)\r\n", asset->filename, asset->bytes);
    for (uint64_t offset = 0; offset < asset->bytes;) {
        UINT wanted = (UINT)(asset->bytes - offset > 1048576 ? 1048576 : asset->bytes - offset), got = 0;
        FRESULT status = f_read(&file, (void *)(UINTPTR)(asset->address + offset), wanted, &got);
        if (status != FR_OK || got != wanted) { f_close(&file); return -1; }
        offset += got;
    }
    f_close(&file); return 0;
}
static int load_tokenizer(void) {
    FIL file; UINT got = 0;
    if (f_open(&file, "0:/QWEN25/TOKEN.BIN", FA_READ) != FR_OK) return -1;
    size_t bytes = (size_t)f_size(&file);
    if (bytes > sizeof(tokenizer_data)) { f_close(&file); return -1; }
    int status = f_read(&file, tokenizer_data, (UINT)bytes, &got) != FR_OK || got != bytes;
    f_close(&file);
    return status ? -1 : qt_init(tokenizer_data, bytes);
}
static void copy_embedding(void *context, unsigned token, unsigned row) {
    (void)context;
#ifdef QWEN_HOST_SIM
    extern void sim_embedding(unsigned, unsigned);
    sim_embedding(token, row);
#else
    uint64_t stride = QWEN_HIDDEN * 2;
    memcpy((void *)(UINTPTR)(QWEN_INPUT_BASE + row * stride),
           (const void *)(UINTPTR)(QWEN_EMBEDDING_BASE + token * stride), (size_t)stride);
    barrier();
#endif
}
static int on_token(void *context, unsigned iteration, unsigned token, uint64_t elapsed) {
    (void)context;
    const unsigned char *bytes = NULL; unsigned length = 0; int special = 0;
    int unmapped = qt_decode(token, &bytes, &length, &special);
    if (output_mode == 2) {
        printf("@TOKEN %u %u %u %" PRIu64 " ", request_id, iteration, token, elapsed);
        for (unsigned i = 0; i < length; ++i) printf("%02x", bytes[i]);
        printf("\r\n");
    } else if (output_mode == 1) {
        printf("TOKEN iteration=%u id=%u ticks=%" PRIu64 "\r\n", iteration, token, elapsed);
    } else if (!unmapped && !special) qutf8_push(&decoder, bytes, length);
    if (unmapped) printf("\r\nERROR unmapped token %u; PL numerical output requires diagnosis\r\n", token);
    if (log_active) {
        char line[128]; UINT wrote;
        int n = snprintf(line, sizeof(line), "%u,%u,%u,%" PRIu64 "\n", request_id, iteration, token, elapsed);
        if (n <= 0 || (size_t)n >= sizeof(line) || f_write(&result_file, line, (UINT)n, &wrote) != FR_OK ||
            wrote != (UINT)n || f_sync(&result_file) != FR_OK) {
            f_close(&result_file); log_active = 0;
            printf("\r\nWARNING SD token log failed; UART generation continues\r\n");
        }
    }
    return unmapped ? -1 : 0;
}
static void configure_accelerator(void) {
    csr_write(NULL, 0x0140, 255);
    csr_write(NULL, 0x00d0, 0x3586);
    csr_write(NULL, 0x0110, QWEN_DESCRIPTOR_BASE);
    csr_write(NULL, 0x0100, QWEN_INPUT_BASE);
    csr_write(NULL, 0x0108, QWEN_CONTEXT_BASE);
    csr_write(NULL, 0x0118, QWEN_K_BASE);
    csr_write(NULL, 0x0120, QWEN_V_BASE);
    csr_write(NULL, 0x0128, QWEN_Q_BASE);
    csr_write(NULL, 0x0130, 0);
    csr_write(NULL, 0x00b0, 0);
    csr_write(NULL, 0x00b8, 0);
}
static void reset_context(void) {
#ifdef QWEN_HOST_SIM
    extern void sim_clear(void); sim_clear();
#else
    for (unsigned i = 0; i < QWEN_CLEAR_REGION_COUNT; ++i)
        memset((void *)(UINTPTR)clear_regions[i].address, 0, (size_t)clear_regions[i].bytes);
#endif
    barrier();
}
static int tokenizer_selftest(void) {
    static const uint32_t expected[] = {151644,8948,198,2610,525,1207,16948,11,3465,553,54364,
        14817,13,1446,525,264,10950,17847,13,151645,198,151644,872,198,14880,17714,35946,
        100157,102434,99562,3837,20412,100342,99887,99562,151645,198,151644,77091,198};
    unsigned count;
    if (qt_chat_encode("请为我介绍厦门大学，是全国重点大学", prompt, 256, &count) ||
        count != sizeof(expected) / sizeof(expected[0]) || memcmp(prompt, expected, sizeof(expected))) return -1;
    return 0;
}
static void help(void) {
    printf("Enter UTF-8 prompt and press Enter. Each request starts a fresh context.\r\n"
           "/max N (1..255), /mode text|ids|events, /template chat|raw, /encode TEXT, /selftest, /reset, /help\r\n"
           "KV capacity 256 including chat template and output reserve. Default /max 20.\r\n");
}
static int command(void) {
    if (!strcmp(input, "/help")) { help(); return 1; }
    if (!strcmp(input, "/reset")) { reset_context(); puts("CONTEXT_CLEARED"); return 1; }
    if (!strcmp(input, "/selftest")) {
        puts(tokenizer_selftest() ? "TOKENIZER_SELFTEST_FAILED" : "TOKENIZER_SELFTEST_OK 40 tokens"); return 1;
    }
    if (!strncmp(input, "/max ", 5)) {
        char *end; unsigned long n = strtoul(input + 5, &end, 10);
        if (!*end && end != input + 5 && n >= 1 && n <= 255) { max_new = (unsigned)n; printf("MAX_NEW=%u\r\n", max_new); }
        else puts("ERROR use /max 1..255");
        return 1;
    }
    if (!strncmp(input, "/mode ", 6)) {
        if (!strcmp(input + 6, "text")) output_mode = 0;
        else if (!strcmp(input + 6, "ids")) output_mode = 1;
        else if (!strcmp(input + 6, "events")) output_mode = 2;
        else { puts("ERROR use /mode text|ids|events"); return 1; }
        printf("MODE=%s\r\n", input + 6); return 1;
    }
    if (!strncmp(input, "/template ", 10)) {
        if (!strcmp(input + 10, "chat")) raw_template = 0;
        else if (!strcmp(input + 10, "raw")) raw_template = 1;
        else { puts("ERROR use /template chat|raw"); return 1; }
        printf("TEMPLATE=%s\r\n", input + 10); return 1;
    }
    if (!strncmp(input, "/encode ", 8)) {
        unsigned count = 0;
        int status = raw_template ? qt_encode(input + 8, prompt, 256, &count) : qt_chat_encode(input + 8, prompt, 256, &count);
        if (status) printf("ERROR %s\r\n", qt_error());
        else { printf("@PROMPT %u", count); for (unsigned i = 0; i < count; ++i) printf(" %u", prompt[i]); printf("\r\n"); }
        return 1;
    }
    if (input[0] == '/') { puts("ERROR unknown command; /help"); return 1; }
    return 0;
}
int main(void) {
    Xil_DCacheDisable();
    XUartPs_Config *config = XUartPs_LookupConfig(STDIN_BASEADDRESS);
    if (!config || XUartPs_CfgInitialize(&uart, config, config->BaseAddress) || XUartPs_SetBaudRate(&uart, 115200)) return 1;
    XUartPs_SetOperMode(&uart, XUARTPS_OPER_MODE_NORMAL);
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("\r\nQwen25 KV260 100 MHz UART interactive inference\r\n");
    printf("PL numerical model signoff pending. No CPU inference fallback.\r\n");
    if (qwen_board_prepare()) { puts("ERROR PL clock/reset preparation failed"); halt(); }
    printf("CSR_ID=%016" PRIx64 " VERSION=%016" PRIx64 "\r\n", csr_read(NULL, 0), csr_read(NULL, 8));
    if (csr_read(NULL, 0) != 0x515732354b563236ULL) { puts("ERROR accelerator ID"); halt(); }
    if (f_mount(&filesystem, "0:/", 1) != FR_OK) { puts("ERROR SD mount"); halt(); }
    if (load_tokenizer()) { printf("ERROR loading TOKEN.BIN: %s\r\n", qt_error()); halt(); }
    if (tokenizer_selftest()) { puts("ERROR tokenizer 40-token golden selftest"); halt(); }
    puts("TOKENIZER_SELFTEST_OK 40 tokens");
    for (unsigned i = 0; i < QWEN_ASSET_COUNT; ++i) if (load_asset(assets + i)) { puts("ERROR model loading"); halt(); }
    configure_accelerator();
    help();
    struct qgen_ops ops = {NULL, csr_write, csr_read, timer_ticks, wait_tick, copy_embedding, on_token, COUNTS_PER_SECOND};
    for (;;) {
        printf("\r\nREADY\r\nprompt> ");
        int length = read_line();
        if (length < 0) { puts("ERROR input exceeds 4096 UTF-8 bytes"); continue; }
        if (!length || command()) continue;
        unsigned count = 0;
        int encode_status = raw_template ? qt_encode(input, prompt, 256 - max_new, &count) : qt_chat_encode(input, prompt, 256 - max_new, &count);
        if (encode_status) { printf("ERROR %s\r\n", qt_error()); continue; }
        reset_context();
        ++request_id;
        printf("@BEGIN %u prompt_tokens=%u max_new=%u\r\n", request_id, count, max_new);
        log_active = f_open(&result_file, "0:/CHAT.CSV", FA_OPEN_ALWAYS | FA_WRITE) == FR_OK;
        if (log_active) {
            if (f_lseek(&result_file, f_size(&result_file)) != FR_OK) { f_close(&result_file); log_active = 0; }
            else if (!f_size(&result_file)) {
                static const char header[] = "request,iteration,token_id,timer_ticks\n";
                UINT wrote;
                if (f_write(&result_file, header, sizeof(header) - 1, &wrote) != FR_OK || wrote != sizeof(header) - 1) {
                    f_close(&result_file); log_active = 0;
                }
            }
        }
        if (!log_active) puts("WARNING SD log unavailable; UART generation continues");
        qutf8_init(&decoder, console_bytes, NULL);
        if (!output_mode) printf("answer> ");
        struct qgen_result result;
        int status = qgen_run(&ops, prompt, count, max_new, &result);
        qutf8_finish(&decoder);
        if (log_active) { f_close(&result_file); log_active = 0; }
        printf("\r\n@END %u produced=%u reason=%s status=%d\r\n", request_id, result.produced,
               status ? "error" : (result.eos ? "eos" : "limit"), status);
        if (status == -2 || status == -3) {
            printf("FATAL PL status=%016" PRIx64 "; reload PL or power-cycle before retry\r\n", result.last_status);
            halt();
        }
    }
}
