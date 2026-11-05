/*
 * uart_console_override.c -- newlib console hooks for the PS UART.
 * Linked as uart_console_override.o ahead of -lxil and -lxilstandalone.
 *
 * Which symbol to override, and why exactly this one (nm on bsp/lib):
 *   libxil.a            xuartps_hw.c.obj  outbyte / inbyte -> T (strong)
 *   libxilstandalone.a  write.c.obj       _write           -> W (weak)
 *   libxilstandalone.a  read.c.obj        _read            -> W (weak)
 * `outbyte` lives in xuartps_hw.c.obj, which main() always pulls in because it
 * drives XUartPs_* directly, so redefining it here is a duplicate definition.
 * The two newlib syscall hooks are weak, so a strong definition in this object
 * wins at link time and write.c / read.c are never extracted from the archive.
 *
 * Why override at all instead of letting the BSP path run: printf() is used
 * from the very first banner and from inside the per-token callback. The BSP
 * path ends in XUartPs_SendByte(), which re-reads driver state on every byte
 * and does nothing useful before XUartPs_CfgInitialize() has run -- a
 * configuration failure would take the error message down with it. One FIFO
 * write per byte keeps the console alive independently of the driver.
 *
 * Register offsets and status bits: bsp/include/xuartps_hw.h
 *   SR   0x2C  bit4 TXFULL, bit1 RXEMPTY
 *   FIFO 0x30
 * Console instance: STDOUT_BASEADDRESS / STDIN_BASEADDRESS in
 * bsp/include/bspconfig.h, both 0xFF01_0000 on this platform.
 */

#include <stdint.h>
#include "bspconfig.h"

#define QWEN_UART_BASE       ((uintptr_t)STDOUT_BASEADDRESS)
#define QWEN_UART_SR         0x002CU
#define QWEN_UART_FIFO       0x0030U
#define QWEN_UART_SR_TXFULL  0x00000010U
#define QWEN_UART_SR_RXEMPTY 0x00000002U

static volatile uint32_t *qwen_uart_reg(uintptr_t offset)
{
    return (volatile uint32_t *)(QWEN_UART_BASE + offset);
}

/* Backs printf()/puts(): block only while the 64-byte TX FIFO is full. */
int _write(int fd, char *buf, int len)
{
    volatile uint32_t *status = qwen_uart_reg(QWEN_UART_SR);
    volatile uint32_t *fifo = qwen_uart_reg(QWEN_UART_FIFO);
    int written;

    (void)fd;
    for (written = 0; written < len; ++written) {
        while ((*status & QWEN_UART_SR_TXFULL) != 0U) {
        }
        *fifo = (uint32_t)(unsigned char)buf[written];
    }
    return written;
}

/* Backs getchar()/scanf(): block for the first byte, then drain what is left. */
int _read(int fd, char *buf, int len)
{
    volatile uint32_t *status = qwen_uart_reg(QWEN_UART_SR);
    volatile uint32_t *fifo = qwen_uart_reg(QWEN_UART_FIFO);
    int count = 0;

    (void)fd;
    if (len <= 0) {
        return 0;
    }
    while ((*status & QWEN_UART_SR_RXEMPTY) != 0U) {
    }
    while (count < len) {
        buf[count++] = (char)(unsigned char)(*fifo & 0xffU);
        if ((*status & QWEN_UART_SR_RXEMPTY) != 0U) {
            break;
        }
    }
    return count;
}
