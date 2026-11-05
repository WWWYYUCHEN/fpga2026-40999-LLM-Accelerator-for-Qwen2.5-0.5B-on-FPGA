/*
 * sd_prepare.c -- PS-side bring-up handshake with the candidate9 accelerator.
 * Module       : qwen_board_prepare()
 * Caller       : fw/main.c, before the first CSR write
 * Simulator    : fw/tests/board_sim.c supplies its own stub, so this file is
 *                only compiled into the AArch64 image.
 * Register map : candidate9_rtl/qwen25_result_regs.v, qwen25_compute_cfg.v
 * Address map  : vivado/address_map.h (QWEN25_CTRL @ 0x8000_0000)
 *
 * Scope. The boot chain configures the PL and loads QWENSD.ELF afterwards, so
 * by the time this runs PL0 is already at frequency, the PL reset is already
 * released and DDR is already trained. Re-programming any of those from
 * EL1/EL2/EL3 would fight the FSBL and U-Boot. This module owns only what the
 * loader cannot know:
 *
 *   1. ID      0x0000 == 0x515732354b563236. A mismatch means the PL is not
 *              configured, the CSR window is not decoded, or QWEN_CSR_BASE in
 *              board_parameters.h does not match the BD address editor.
 *              Retried, because PL configuration can still be settling.
 *   2. VERSION 0x0008 == 0x00010000. Guards against a bitstream older than the
 *              register contract this firmware was built against.
 *   3. IDLE    STATUS[1] low, so a task left running by a previous entry cannot
 *              be mistaken for this session's first request.
 *
 * candidate9 removed the top-level IRQ: qwen25_result_regs.v ties abort_req,
 * soft_reset_req and error_clear_req to 0 and reads back 0 at 0x0010 / 0x0020 /
 * 0x0028. This firmware is therefore poll-only and never touches 0x0020 or
 * 0x0028; CLEAR (0x0010 bit3) is still honoured and is used to drop a stale
 * result register without starting a task.
 */

#include <stdint.h>
#include "xil_io.h"
#include "../board_parameters.h"

/* Result/control window, from candidate9 qwen25_result_regs.v. */
#define QWEN_REG_ID            0x0000U  /* RO  {64'h515732354b563236}        */
#define QWEN_REG_VERSION       0x0008U  /* RO  {64'h0000000000010000}        */
#define QWEN_REG_CONTROL       0x0010U  /* WO  [0] START [3] CLEAR           */
#define QWEN_REG_STATUS        0x0018U  /* RO  {result_valid, status[5:0]}   */
#define QWEN_REG_ERROR_CODE    0x0030U  /* RO  status[95:64]                 */

#define QWEN_CTRL_CLEAR        0x08U    /* clear_event = bits[0] | bits[3]   */
#define QWEN_STATUS_BUSY       (1U << 1)

#define QWEN_ID_VALUE          0x515732354b563236ULL
#define QWEN_VERSION_VALUE     0x0000000000010000ULL
#define QWEN_PREPARE_POLLS     10000U

static void qwen_barrier(void)
{
#if defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");
#endif
}

/* Short settle loop. usleep() needs the timer driver, which main() owns. */
static void qwen_settle(void)
{
    for (volatile unsigned spin = 0; spin < 1000U; ++spin) {
    }
}

static uint64_t qwen_csr_read(unsigned offset)
{
    qwen_barrier();
    return Xil_In64((UINTPTR)(QWEN_CSR_BASE + offset));
}

static void qwen_csr_write(unsigned offset, uint64_t value)
{
    Xil_Out64((UINTPTR)(QWEN_CSR_BASE + offset), value);
    qwen_barrier();
}

/* 0 ready, -1 no accelerator answer, -2 contract mismatch, -3 still busy. */
int qwen_board_prepare(void)
{
    uint64_t id = 0;
    unsigned poll;

    for (poll = 0; poll < QWEN_PREPARE_POLLS; ++poll) {
        id = qwen_csr_read(QWEN_REG_ID);
        if (id == QWEN_ID_VALUE) {
            break;
        }
        qwen_settle();
    }
    if (id != QWEN_ID_VALUE) {
        return -1;
    }
    if (qwen_csr_read(QWEN_REG_VERSION) != QWEN_VERSION_VALUE) {
        return -2;
    }

    /* CLEAR is honoured even while BUSY and does not start a task. */
    qwen_csr_write(QWEN_REG_CONTROL, QWEN_CTRL_CLEAR);

    for (poll = 0; poll < QWEN_PREPARE_POLLS; ++poll) {
        if ((qwen_csr_read(QWEN_REG_STATUS) & QWEN_STATUS_BUSY) == 0U) {
            return 0;
        }
        qwen_settle();
    }
    return -3;
}
