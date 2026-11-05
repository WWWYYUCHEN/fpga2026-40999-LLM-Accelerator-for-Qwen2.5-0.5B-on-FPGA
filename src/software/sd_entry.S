/*
 * sd_entry.S -- AArch64 reset entry for QWENSD.ELF (candidate9)
 * Linker entry : ENTRY(sd_entry) in bsp/lscript.ld
 * Load address : 0x0010_0000 (psu_ddr_0_memory, 63 MiB)
 * Dependencies : __el3_stack / __el2_stack / __el1_stack,
 *                __bss_start__ / __bss_end__ from lscript.ld
 *
 * Boot chain: FSBL -> U-Boot -> boot.scr -> configure PL -> load QWENSD.ELF.
 * U-Boot has already trained DDR, released the PL reset, programmed PL0 to the
 * frequency this bitstream was placed for, and its cleanup_before_linux() has
 * disabled the MMU and both caches before the jump. This entry therefore does
 * NOT touch clock, reset, MMU or cache registers; it only prepares the C
 * runtime, in this order:
 *
 *   1. mask DAIF   -- no vector table is installed and the firmware polls the
 *                     accelerator, so no interrupt may be taken before main().
 *   2. pick SP     -- main() uses printf with a 1 MiB stack; the usable stack
 *                     top depends on the EL the loader left us in, and
 *                     lscript.ld reserves one segment per EL.
 *   3. clear BSS   -- .bss is NOLOAD, so nothing else clears it. FATFS, FIL,
 *                     XUartPs objects and the tokenizer image live there.
 *   4. enable FP   -- newlib printf and memcpy/memset inline paths may emit
 *                     FP/SIMD accesses; trapping them would fault.
 */

    .section .boot,"ax"
    .global sd_entry
    .type   sd_entry,%function
    .balign 64

sd_entry:
    /* x0 holds the loader argument; main(void) takes none. */
    msr     daifset, #0xf               /* mask IRQ, FIQ, SError, Debug */

    /* Only the boot core runs; the other three park in WFE. */
    mrs     x2, mpidr_el1
    and     x2, x2, #0xff
    cbnz    x2, sd_secondary

    mrs     x3, CurrentEL
    ubfx    x3, x3, #2, #2
    cmp     x3, #3
    b.eq    sd_prepare_el3
    cmp     x3, #2
    b.eq    sd_prepare_el2
    b       sd_prepare_el1

sd_prepare_el3:
    msr     cptr_el3, xzr               /* no FP/SIMD trap from EL3 */
    msr     SPSel, #1
    ldr     x0, =__el3_stack
    b       sd_stack

sd_prepare_el2:
    msr     cptr_el2, xzr
    msr     SPSel, #1
    ldr     x0, =__el2_stack
    b       sd_stack

sd_prepare_el1:
    msr     SPSel, #1
    ldr     x0, =__el1_stack

sd_stack:
    mov     sp, x0                      /* full-descending stack top */

    /* .bss is NOLOAD and 64-byte aligned at both ends. */
    ldr     x0, =__bss_start__
    ldr     x1, =__bss_end__
    cmp     x0, x1
    b.hs    sd_bss_done
sd_bss:
    str     xzr, [x0], #8
    cmp     x0, x1
    b.lo    sd_bss
sd_bss_done:

    /* CPACR_EL1.FPEN = 0b11: FP/SIMD usable at EL0 and EL1. */
    mov     x0, #0x00300000
    msr     cpacr_el1, x0
    isb

    bl      main

sd_halt:
    wfe
    b       sd_halt

sd_secondary:
    wfe
    b       sd_secondary

    .size   sd_entry, .-sd_entry
