/* Native integration harness. PL token outputs are fixtures, not neural math. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <io.h>
#ifdef QWEN_MODEL_SIM
#include <windows.h>
#endif
#include "ff.h"
#include "xil_io.h"
#include "xuartps.h"
#include "xiltimer.h"
#include "../../board_parameters.h"
static uint64_t regs[128], tick;
static unsigned request, iteration, poll_count, token_count, tokens[256], output;
static FILE *trace;
static const char *fault;

void Xil_DCacheDisable(void) { }
int qwen_board_prepare(void) { return 0; }
#if defined(QWEN_RTL_SIM)
extern void rtl_advance(unsigned);
extern uint64_t rtl_ticks(void);
void usleep(unsigned us) { rtl_advance(us * 100); }
void XTime_GetTime(XTime *out) { *out = rtl_ticks(); }
#else
void usleep(unsigned us) { tick += us; }
#ifdef QWEN_MODEL_SIM
void XTime_GetTime(XTime *out) { *out = GetTickCount64() * 1000; }
#else
void XTime_GetTime(XTime *out) { tick += 100; *out = tick; }
#endif
#endif
XUartPs_Config *XUartPs_LookupConfig(uintptr_t base) { static XUartPs_Config c; c.BaseAddress = base; return &c; }
int XUartPs_CfgInitialize(XUartPs *u, XUartPs_Config *c, uintptr_t base) {
    (void)u; (void)c; (void)base;
    _setmode(_fileno(stdin), _O_BINARY); _setmode(_fileno(stdout), _O_BINARY);
    fault = getenv("SIM_FAULT");
    const char *name = getenv("SIM_TRACE");
    if (name) trace = fopen(name, "wb");
#if defined(QWEN_RTL_SIM)
    puts("SIMULATION: real PS C and compiled accelerator RTL; virtual UART/SD; vendor math simulation models.");
#elif defined(QWEN_MODEL_SIM)
    puts("SIMULATION: production PS C; mathematical PL substitute executes deployed INT8/BF16 weights on PC CPU. Not RTL cycle simulation.");
#else
    puts("SIMULATION: real PS C; virtual UART/SD/MMIO; scripted PL tokens, NO model computation.");
#endif
    return 0;
}
int XUartPs_SetBaudRate(XUartPs *u, unsigned baud) { (void)u; return baud == 115200 ? 0 : -1; }
void XUartPs_SetOperMode(XUartPs *u, unsigned mode) { (void)u; (void)mode; }
void XUartPs_SendByte(uintptr_t base, unsigned char c) { (void)base; putchar(c); }
unsigned char XUartPs_RecvByte(uintptr_t base) { (void)base; int c = getchar(); if (c == EOF) exit(0); return (unsigned char)c; }
FRESULT f_mount(FATFS *fs, const char *path, int now) { (void)fs; (void)path; (void)now; return fault && !strcmp(fault, "mount") ? 1 : 0; }
FRESULT f_open(FIL *f, const char *path, int flags) {
    memset(f, 0, sizeof(*f));
    for (unsigned i = 0; i < QWEN_ASSET_COUNT; ++i)
        if (!strcmp(path, assets[i].filename)) { f->asset = 1; f->size = assets[i].bytes; return 0; }
    if (fault && !strcmp(fault, "sd_log") && strstr(path, "CHAT.CSV")) return 1;
    char filename[2048];
    const char *root = getenv("SIM_SD");
    if (!root) root = ".";
    snprintf(filename, sizeof(filename), "%s/%s", root, path + 3);
    f->handle = fopen(filename, flags & FA_WRITE ? "r+b" : "rb");
    if (!f->handle && (flags & FA_OPEN_ALWAYS)) f->handle = fopen(filename, "w+b");
    if (!f->handle) return 1;
    fseek(f->handle, 0, SEEK_END); f->size = (uint64_t)ftell(f->handle); rewind(f->handle);
    return 0;
}
FRESULT f_close(FIL *f) { if (f->handle) fclose(f->handle); f->handle = NULL; return 0; }
FRESULT f_read(FIL *f, void *out, UINT n, UINT *got) {
    if (f->asset) { *got = n; f->offset += n; return 0; }
    *got = (UINT)fread(out, 1, n, f->handle); return ferror(f->handle) ? 1 : 0;
}
FRESULT f_write(FIL *f, const void *data, UINT n, UINT *written) {
    *written = (UINT)fwrite(data, 1, n, f->handle);
    f->size = (uint64_t)ftell(f->handle); return *written == n ? 0 : 1;
}
FRESULT f_sync(FIL *f) { return fflush(f->handle); }
FRESULT f_lseek(FIL *f, uint64_t offset) { return fseek(f->handle, (long)offset, SEEK_SET); }
void sim_clear(void) {
#ifdef QWEN_RTL_SIM
    extern void rtl_clear(void); rtl_clear();
#endif
#ifdef QWEN_MODEL_SIM
    extern void math_clear(void); math_clear();
#endif
    if (trace) { fprintf(trace, "CLEAR\n"); fflush(trace); }
}
void sim_embedding(unsigned token, unsigned row) {
#ifdef QWEN_RTL_SIM
    extern void rtl_embedding(unsigned, unsigned); rtl_embedding(token, row);
#endif
#ifdef QWEN_MODEL_SIM
    extern void math_embedding(unsigned, unsigned); math_embedding(token, row);
#endif
    if (trace) { fprintf(trace, "EMBED %u %u\n", token, row); fflush(trace); }
}
#if !defined(QWEN_RTL_SIM) && !defined(QWEN_MODEL_SIM)
static void load_plan(void) {
    const char *name = getenv("SIM_PLAN");
    FILE *plan = name ? fopen(name, "r") : NULL;
    if (!plan) { fputs("SIM_PLAN missing\n", stderr); exit(2); }
    char line[4096] = {0};
    for (unsigned r = 0; r < request; ++r) if (!fgets(line, sizeof(line), plan)) { fclose(plan); exit(2); }
    fclose(plan);
    token_count = 0;
    char *p = line;
    while (*p) {
        char *end;
        unsigned long id = strtoul(p, &end, 10);
        if (end == p) break;
        if (token_count >= 256) exit(2);
        tokens[token_count++] = (unsigned)id;
        p = end;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
    }
    if (!token_count) exit(2);
}
void Xil_Out64(UINTPTR address, uint64_t value) {
    unsigned off = (unsigned)(address - QWEN_CSR_BASE);
    if (off / 8 >= 128) exit(3);
    regs[off / 8] = value;
    if (trace) fprintf(trace, "WRITE %04x %llu\n", off, (unsigned long long)value);
    if (off == 0x10 && value == 1) {
        if (regs[0x80 / 8] == 0) { ++request; iteration = 0; load_plan(); }
        else ++iteration;
        poll_count = 0;
        output = tokens[iteration < token_count ? iteration : token_count - 1];
        if (trace) fprintf(trace, "START %u %u mode=%llu seq=%llu context=%llu pos=%llu\n", request, iteration,
            (unsigned long long)regs[0x80 / 8], (unsigned long long)regs[0x88 / 8],
            (unsigned long long)regs[0x90 / 8], (unsigned long long)regs[0x98 / 8]);
    }
    if (trace) fflush(trace);
}
uint64_t Xil_In64(UINTPTR address) {
    unsigned off = (unsigned)(address - QWEN_CSR_BASE);
    if (off == 0) return fault && !strcmp(fault, "id") ? 0 : 0x515732354b563236ULL;
    if (off == 8) return 1;
    if (off == 0x18) {
        if (fault && !strcmp(fault, "timeout")) { tick += 600000000ULL; return 2; }
        if (fault && !strcmp(fault, "pl_error")) return 8;
        return ++poll_count < 3 ? 2 : 0x44;
    }
    if (off == 0x1f8) return fault && !strcmp(fault, "invalid_token") ? 151936 : output;
    if (off / 8 >= 128) exit(3);
    return regs[off / 8];
}
#elif defined(QWEN_RTL_SIM)
extern void rtl_write(unsigned, uint64_t);
extern uint64_t rtl_read(unsigned);
void Xil_Out64(UINTPTR address, uint64_t value) {
    unsigned off = (unsigned)(address - QWEN_CSR_BASE);
    if (trace) { fprintf(trace, "WRITE %04x %llu\n", off, (unsigned long long)value); fflush(trace); }
    rtl_write(off, value);
}
uint64_t Xil_In64(UINTPTR address) { return rtl_read((unsigned)(address - QWEN_CSR_BASE)); }
#else
static unsigned input_ids[256], serial_number;
static int waiting;
static const char *model_io(void) {
    const char *path = getenv("MODEL_IO");
    if (!path) { fputs("MODEL_IO required\n", stderr); exit(2); }
    return path;
}
static void path_for(char *out, const char *file) { snprintf(out, 2048, "%s/%s", model_io(), file); }
void math_clear(void) { waiting = 0; }
void math_embedding(unsigned token, unsigned row) { if (row >= 256) exit(3); input_ids[row] = token; }
void Xil_Out64(UINTPTR address, uint64_t value) {
    unsigned off = (unsigned)(address - QWEN_CSR_BASE);
    if (off / 8 >= 128) exit(3);
    regs[off / 8] = value;
    if (trace) { fprintf(trace, "WRITE %04x %llu\n", off, (unsigned long long)value); fflush(trace); }
    if (off == 0x10 && value == 1) {
        char temporary[2048], request_path[2048];
        path_for(temporary, "request.tmp"); path_for(request_path, "request.json");
        FILE *f = fopen(temporary, "wb"); if (!f) exit(3);
        fprintf(f, "{\"serial\":%u,\"clear\":%s,\"position\":%u,\"ids\":[", ++serial_number,
                regs[0x80 / 8] == 0 ? "true" : "false", (unsigned)regs[0x98 / 8]);
        unsigned rows = (unsigned)regs[0x88 / 8]; if (!rows || rows > 256) exit(3);
        for (unsigned i = 0; i < rows; ++i) fprintf(f, "%s%u", i ? "," : "", input_ids[i]);
        fprintf(f, "]}"); fclose(f);
        if (rename(temporary, request_path)) exit(3);
        waiting = 1;
    }
}
uint64_t Xil_In64(UINTPTR address) {
    unsigned off = (unsigned)(address - QWEN_CSR_BASE);
    if (off == 0) return 0x515732354b563236ULL;
    if (off == 8) return 1;
    if (off == 0x18 && waiting) {
        char path[2048]; path_for(path, "response.txt");
        FILE *f = fopen(path, "rb");
        if (!f) { Sleep(1); return 2; }
        unsigned seq, token, error;
        if (fscanf(f, "%u %u %u", &seq, &token, &error) != 3 || seq != serial_number) { fclose(f); exit(3); }
        fclose(f); remove(path);
        output = token; waiting = 0;
        regs[0x18 / 8] = error ? 8 : 0x44;
    }
    if (off == 0x1f8) return output;
    return regs[off / 8];
}
#endif
