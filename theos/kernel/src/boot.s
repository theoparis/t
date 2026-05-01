.section .text.boot
.global _start

_start:
    /* ARM64 Linux kernel boot header */
    b       real_start              /* branch to kernel start */
    .long   0                       /* reserved */
    .quad   0x80000                 /* Image load offset from start of RAM (512KB) */
    .quad   _end - _start           /* Effective size of kernel image */
    .quad   0                       /* Informative flags */
    .quad   0                       /* reserved */
    .quad   0                       /* reserved */
    .quad   0                       /* reserved */
    .ascii  "ARM\x64"               /* Magic number, "ARM\x64" */
    .long   0                       /* reserved */

real_start:
    /* Read CPU ID, stop custom cores */
    mrs x0, mpidr_el1
    and x0, x0, #3
    cbz x0, 1f
    b   hang

1:
    /* Initialize BSS */
    ldr x0, =__bss_start
    ldr x1, =__bss_end
    cmp x0, x1
    bge 2f

    /* Zero out BSS */
zero_bss:
    str xzr, [x0], #8
    cmp x0, x1
    blt zero_bss

2:
    /* Enable FPU/SIMD */
    mov x0, #0x300000               /* CPACR_EL1.FPEN = 0b11 (EL0/EL1 enabled) */
    msr cpacr_el1, x0
    isb

    /* Jump to Rust code */
    /* Set stack pointer before jump */
    ldr x0, =0x80000000
    mov sp, x0
    bl  kmain

hang:
    wfe
    b   hang
