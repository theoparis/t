.section .text
.global vectors
.global internal_exception_handler

.balign 2048
vectors:
    /* Current EL with SP0 */
    .balign 128
    b sync_handler
    .balign 128
    b irq_handler
    .balign 128
    b .
    .balign 128
    b .

    /* Current EL with SPx */
    .balign 128
    b sync_handler
    .balign 128
    b irq_handler
    .balign 128
    b .
    .balign 128
    b .

    /* Lower EL using AArch64 */
    .balign 128
    b sync_handler
    .balign 128
    b irq_handler
    .balign 128
    b .
    .balign 128
    b .

    /* Lower EL using AArch32 */
    .balign 128
    b sync_handler
    .balign 128
    b irq_handler
    .balign 128
    b .
    .balign 128
    b .

sync_handler:
    /* Save context */
    /* TrapFrame size = 32 * 8 + 2 * 8 = 272 bytes. We use 288 for alignment. */
    sub sp, sp, #288
    stp x0, x1, [sp, #16 * 0]
    stp x2, x3, [sp, #16 * 1]
    stp x4, x5, [sp, #16 * 2]
    stp x6, x7, [sp, #16 * 3]
    stp x8, x9, [sp, #16 * 4]
    stp x10, x11, [sp, #16 * 5]
    stp x12, x13, [sp, #16 * 6]
    stp x14, x15, [sp, #16 * 7]
    stp x16, x17, [sp, #16 * 8]
    stp x18, x19, [sp, #16 * 9]
    stp x20, x21, [sp, #16 * 10]
    stp x22, x23, [sp, #16 * 11]
    stp x24, x25, [sp, #16 * 12]
    stp x26, x27, [sp, #16 * 13]
    stp x28, x29, [sp, #16 * 14]
    str x30, [sp, #16 * 15]

    mrs x10, elr_el1
    mrs x11, spsr_el1
    mrs x12, sp_el0
    stp x10, x11, [sp, #16 * 16]
    str x12, [sp, #272]

    /* Pass trap frame to Rust */
    mov x0, sp
    bl handle_sync_exception

    /* Restore context */
    ldr x12, [sp, #272]
    msr sp_el0, x12
    ldp x10, x11, [sp, #16 * 16]
    msr elr_el1, x10
    msr spsr_el1, x11

    ldp x0, x1, [sp, #16 * 0]
    ldp x2, x3, [sp, #16 * 1]
    ldp x4, x5, [sp, #16 * 2]
    ldp x6, x7, [sp, #16 * 3]
    ldp x8, x9, [sp, #16 * 4]
    ldp x10, x11, [sp, #16 * 5]
    ldp x12, x13, [sp, #16 * 6]
    ldp x14, x15, [sp, #16 * 7]
    ldp x16, x17, [sp, #16 * 8]
    ldp x18, x19, [sp, #16 * 9]
    ldp x20, x21, [sp, #16 * 10]
    ldp x22, x23, [sp, #16 * 11]
    ldp x24, x25, [sp, #16 * 12]
    ldp x26, x27, [sp, #16 * 13]
    ldp x28, x29, [sp, #16 * 14]
    ldr x30, [sp, #16 * 15]
    add sp, sp, #800
    eret

irq_handler:
    b .
]

    /* Pass trap frame to Rust */
    mov x0, sp
    bl handle_sync_exception

    /* Restore SIMD/FP registers q0-q31 */
    ldp q0, q1, [sp, #288 + 32 * 0]
    ldp q2, q3, [sp, #288 + 32 * 1]
    ldp q4, q5, [sp, #288 + 32 * 2]
    ldp q6, q7, [sp, #288 + 32 * 3]
    ldp q8, q9, [sp, #288 + 32 * 4]
    ldp q10, q11, [sp, #288 + 32 * 5]
    ldp q12, q13, [sp, #288 + 32 * 6]
    ldp q14, q15, [sp, #288 + 32 * 7]
    ldp q16, q17, [sp, #288 + 32 * 8]
    ldp q18, q19, [sp, #288 + 32 * 9]
    ldp q20, q21, [sp, #288 + 32 * 10]
    ldp q22, q23, [sp, #288 + 32 * 11]
    ldp q24, q25, [sp, #288 + 32 * 12]
    ldp q26, q27, [sp, #288 + 32 * 13]
    ldp q28, q29, [sp, #288 + 32 * 14]
    ldp q30, q31, [sp, #288 + 32 * 15]

    /* Restore context */
    ldr x12, [sp, #272]
    msr sp_el0, x12
    ldp x10, x11, [sp, #16 * 16]
    msr elr_el1, x10
    msr spsr_el1, x11

    ldp x0, x1, [sp, #16 * 0]
    ldp x2, x3, [sp, #16 * 1]
    ldp x4, x5, [sp, #16 * 2]
    ldp x6, x7, [sp, #16 * 3]
    ldp x8, x9, [sp, #16 * 4]
    ldp x10, x11, [sp, #16 * 5]
    ldp x12, x13, [sp, #16 * 6]
    ldp x14, x15, [sp, #16 * 7]
    ldp x16, x17, [sp, #16 * 8]
    ldp x18, x19, [sp, #16 * 9]
    ldp x20, x21, [sp, #16 * 10]
    ldp x22, x23, [sp, #16 * 11]
    ldp x24, x25, [sp, #16 * 12]
    ldp x26, x27, [sp, #16 * 13]
    ldp x28, x29, [sp, #16 * 14]
    ldr x30, [sp, #16 * 15]
    add sp, sp, #288
    eret

irq_handler:
    b .
