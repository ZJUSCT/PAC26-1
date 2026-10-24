#pragma once
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <omp.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdint>
#include <atomic>
#include <memory>
#include <sched.h>
#include <sys/syscall.h>
#include <ctime>
#include <cstddef>
#include <array>
#include <arm_sve.h>
#include <cerrno>
#include <stdexcept>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <time.h>
#include <asm-generic/ioctl.h>
#include <linux/errno.h>
#include <linux/types.h>

// Selected TC1 policy, validated on cn00655 with 8 x 38 workers.

// FP64 tensor contraction: 8 MPI ranks, 38 workers per NUMA.
// TC1 C-ready and TC2 K-first AC pipelines, both with fused TB Strassen operands.

struct SmeNPanel {
    long panels, k;
    const double *a, *b;
    double *c, *packed_b;
    const double *next_a;
    long b_stride, packed_stride, prefetch_stride, add;
};
static_assert(sizeof(SmeNPanel) == 88, "descriptor layout");
__attribute__((noinline, aligned(64))) static void sme_npanel_pipeline(const SmeNPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z6.d}, p0/z, [x9]\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z12.d}, p0/z, [x9]\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z18.d}, p0/z, [x9]\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 1b\n"
        "8:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "93:\n"
        "st1d {za0v.d[w12, 0]}, p0, [x13]\n"
        "st1d {za4v.d[w12, 0]}, p0, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "st1d {za1v.d[w12, 0]}, p0, [x14]\n"
        "st1d {za5v.d[w12, 0]}, p0, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "st1d {za2v.d[w12, 0]}, p0, [x15]\n"
        "st1d {za6v.d[w12, 0]}, p0, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "st1d {za3v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","cc","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}
__attribute__((noinline, aligned(64))) static void sme_npanel_prefetch_a(const SmeNPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "mov x19, x26\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z6.d}, p0/z, [x9]\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z12.d}, p0/z, [x9]\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z18.d}, p0/z, [x9]\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "prfm pldl2keep, [x19]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "prfm pldl2keep, [x19]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 1b\n"
        "8:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "93:\n"
        "st1d {za0v.d[w12, 0]}, p0, [x13]\n"
        "st1d {za4v.d[w12, 0]}, p0, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "st1d {za1v.d[w12, 0]}, p0, [x14]\n"
        "st1d {za5v.d[w12, 0]}, p0, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "st1d {za2v.d[w12, 0]}, p0, [x15]\n"
        "st1d {za6v.d[w12, 0]}, p0, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "st1d {za3v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","cc","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}
__attribute__((noinline, aligned(64))) static void sme_npanel_pack_b_pipeline(const SmeNPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "mov x19, x25\n"
        "mov x20, #256\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "st1d {z8.d}, p0, [x19]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z9.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z10.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z11.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "st1d {z14.d}, p0, [x19]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z15.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z16.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z17.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "st1d {z20.d}, p0, [x19]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z21.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z22.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z23.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "st1d {z8.d}, p0, [x19]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z9.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z10.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z11.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "st1d {z14.d}, p0, [x19]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z15.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z16.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z17.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "st1d {z20.d}, p0, [x19]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z21.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z22.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z23.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 1b\n"
        "8:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "93:\n"
        "st1d {za0v.d[w12, 0]}, p0, [x13]\n"
        "st1d {za4v.d[w12, 0]}, p0, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "st1d {za1v.d[w12, 0]}, p0, [x14]\n"
        "st1d {za5v.d[w12, 0]}, p0, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "st1d {za2v.d[w12, 0]}, p0, [x15]\n"
        "st1d {za6v.d[w12, 0]}, p0, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "st1d {za3v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","cc","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}

struct SmeCPanel {
    SmeNPanel n;
    const double *last_c;
};
static_assert(sizeof(SmeCPanel) == 96, "descriptor layout");

__attribute__((noinline, aligned(64))) static void sme_npanel_pipeline_c64_spread32(const SmeCPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z6.d}, p0/z, [x9]\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z12.d}, p0/z, [x9]\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z18.d}, p0/z, [x9]\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #32\n"
        "b.lt 94f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #32\n"
        "b.ge 1b\n"
        "94:\n"
        "add x5, x24, #4096\n"
        "cmp x27, #1\n"
        "b.ne 98f\n"
        "ldr x5, [x28, #88]\n"
        "98:\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "95:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #64]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #192]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #320]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #448]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "add x5, x5, #512\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 95b\n"
        "8:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "prfm pldl1keep, [x5, #64]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "prfm pldl1keep, [x5, #192]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "prfm pldl1keep, [x5, #320]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "prfm pldl1keep, [x5, #448]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "93:\n"
        "st1d {za0v.d[w12, 0]}, p0, [x13]\n"
        "st1d {za4v.d[w12, 0]}, p0, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "st1d {za1v.d[w12, 0]}, p0, [x14]\n"
        "st1d {za5v.d[w12, 0]}, p0, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "st1d {za2v.d[w12, 0]}, p0, [x15]\n"
        "st1d {za6v.d[w12, 0]}, p0, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "st1d {za3v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","cc","x5","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}
__attribute__((noinline, aligned(64))) static void sme_npanel_prefetch_a_c64_spread32(const SmeCPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "mov x19, x26\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z6.d}, p0/z, [x9]\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z12.d}, p0/z, [x9]\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z18.d}, p0/z, [x9]\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #32\n"
        "b.lt 94f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "prfm pldl2keep, [x19]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "prfm pldl2keep, [x19]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #32\n"
        "b.ge 1b\n"
        "94:\n"
        "add x5, x24, #4096\n"
        "cmp x27, #1\n"
        "b.ne 98f\n"
        "ldr x5, [x28, #88]\n"
        "98:\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "95:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #64]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #192]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #320]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #448]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "add x5, x5, #512\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 95b\n"
        "8:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "prfm pldl1keep, [x5, #64]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "prfm pldl1keep, [x5, #192]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "prfm pldl1keep, [x5, #320]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "prfm pldl1keep, [x5, #448]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "93:\n"
        "st1d {za0v.d[w12, 0]}, p0, [x13]\n"
        "st1d {za4v.d[w12, 0]}, p0, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "st1d {za1v.d[w12, 0]}, p0, [x14]\n"
        "st1d {za5v.d[w12, 0]}, p0, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "st1d {za2v.d[w12, 0]}, p0, [x15]\n"
        "st1d {za6v.d[w12, 0]}, p0, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "st1d {za3v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","cc","x5","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}
__attribute__((noinline, aligned(64))) static void sme_npanel_pack_b_pipeline_c64_spread32(const SmeCPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "mov x19, x25\n"
        "mov x20, #256\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "st1d {z8.d}, p0, [x19]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z9.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z10.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z11.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "st1d {z14.d}, p0, [x19]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z15.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z16.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z17.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "st1d {z20.d}, p0, [x19]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z21.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z22.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z23.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #32\n"
        "b.lt 94f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "st1d {z8.d}, p0, [x19]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z9.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z10.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z11.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "st1d {z14.d}, p0, [x19]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z15.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z16.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z17.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "st1d {z20.d}, p0, [x19]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z21.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z22.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z23.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #32\n"
        "b.ge 1b\n"
        "94:\n"
        "add x5, x24, #4096\n"
        "cmp x27, #1\n"
        "b.ne 98f\n"
        "ldr x5, [x28, #88]\n"
        "98:\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "95:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #64]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #192]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "st1d {z8.d}, p0, [x19]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z9.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z10.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z11.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #320]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "st1d {z14.d}, p0, [x19]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z15.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z16.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z17.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #448]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "st1d {z20.d}, p0, [x19]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z21.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z22.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z23.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "add x5, x5, #512\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 95b\n"
        "8:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "prfm pldl1keep, [x5, #64]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "prfm pldl1keep, [x5, #192]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "prfm pldl1keep, [x5, #320]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "prfm pldl1keep, [x5, #448]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "93:\n"
        "st1d {za0v.d[w12, 0]}, p0, [x13]\n"
        "st1d {za4v.d[w12, 0]}, p0, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "st1d {za1v.d[w12, 0]}, p0, [x14]\n"
        "st1d {za5v.d[w12, 0]}, p0, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "st1d {za2v.d[w12, 0]}, p0, [x15]\n"
        "st1d {za6v.d[w12, 0]}, p0, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "st1d {za3v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","cc","x5","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}

struct SmeMRun {
    SmeCPanel first;
    const double *const *a_rows;
    long rows;
};
// The M-run prefetches quarter t of the next A strip into L2 during panel tile t, and repeats the request for
// quarter t-1 (x20 = -step): with 36 threads streaming, some first requests are dropped (+1.5% in kbench8).
static_assert(sizeof(SmeMRun) == 112 && __builtin_offsetof(SmeMRun, a_rows) == 96 &&
                  __builtin_offsetof(SmeMRun, rows) == 104,
              "M-run descriptor");

__attribute__((noinline, aligned(64))) static void sme_mrun_accumulate(const SmeMRun &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "ldr x20, [x28, #72]\n"
        "neg x20, x20\n"
        "ldr x6, [x28, #96]\n"
        "ldr x7, [x28, #104]\n"
        "80:\n"
        "ldr x22, [x6], #8\n"
        "ldr x26, [x6]\n"
        "ldr x27, [x28]\n"
        "ldr x23, [x28, #24]\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "mov x19, x26\n"

        "zero {za}\n"

        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z6.d}, p0/z, [x9]\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z12.d}, p0/z, [x9]\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z18.d}, p0/z, [x9]\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #316\n" "b.ne 97f\n" "add x5, x24, #4096\n" "96:\n" "prfm pstl2keep, [x5]\n" "add x5, x5, #64\n" "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "prfm pldl2keep, [x19]\n"
        "prfm pldl2keep, [x19, x20]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "prfm pldl2keep, [x19]\n"
        "prfm pldl2keep, [x19, x20]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "sub x11, x11, #4\n" "cmp x11, #64\n" "b.ge 96b\n" "97:\n"
        "cmp x11, #32\n"
        "b.lt 94f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "prfm pldl2keep, [x19]\n"
        "prfm pldl2keep, [x19, x20]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "prfm pldl2keep, [x19]\n"
        "prfm pldl2keep, [x19, x20]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #32\n"
        "b.ge 1b\n"
        "94:\n"
        "add x5, x24, #4096\n"
        "cmp x27, #1\n"
        "b.ne 98f\n"
        "nop\n"
        "98:\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "95:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "prfm pldl2keep, [x19, x20]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #64]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #192]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "prfm pldl2keep, [x19]\n"
        "prfm pldl2keep, [x19, x20]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #320]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #448]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "add x5, x5, #512\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 95b\n"
        // Finish four ZA banks, then interleave their stores with the final
        // K4 of the other four banks. Each accumulator keeps its K order.
        // Nonmultiple-of-four K uses the original arithmetic tail below.
        "8:\n"
        "cbnz x11, 68f\n"
        "mov x16, x24\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "mov w13, #2\n"
        "mov w14, #4\n"
        "mov w15, #6\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "prfm pldl2keep, [x19]\n"
        "prfm pldl2keep, [x19, x20]\n"
        "add x19, x19, #64\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "prfm pldl1keep, [x5, #64]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "prfm pldl1keep, [x5, #192]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "st1d {za0v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za4v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "st1d {za0v.d[w12, 1]}, p0, [x16]\n"
        "st1d {za4v.d[w12, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "prfm pldl2keep, [x19]\n"
        "prfm pldl2keep, [x19, x20]\n"
        "add x19, x19, #64\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "st1d {za0v.d[w13, 0]}, p0, [x16]\n"
        "st1d {za4v.d[w13, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "st1d {za0v.d[w13, 1]}, p0, [x16]\n"
        "st1d {za4v.d[w13, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "prfm pldl1keep, [x5, #320]\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "st1d {za0v.d[w14, 0]}, p0, [x16]\n"
        "st1d {za4v.d[w14, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "st1d {za0v.d[w14, 1]}, p0, [x16]\n"
        "st1d {za4v.d[w14, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "st1d {za0v.d[w15, 0]}, p0, [x16]\n"
        "st1d {za4v.d[w15, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "st1d {za0v.d[w15, 1]}, p0, [x16]\n"
        "st1d {za4v.d[w15, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "st1d {za1v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za5v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "st1d {za1v.d[w12, 1]}, p0, [x16]\n"
        "st1d {za5v.d[w12, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "st1d {za1v.d[w13, 0]}, p0, [x16]\n"
        "st1d {za5v.d[w13, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "st1d {za1v.d[w13, 1]}, p0, [x16]\n"
        "st1d {za5v.d[w13, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "prfm pldl1keep, [x5, #448]\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "st1d {za1v.d[w14, 0]}, p0, [x16]\n"
        "st1d {za5v.d[w14, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "st1d {za1v.d[w14, 1]}, p0, [x16]\n"
        "st1d {za5v.d[w14, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "st1d {za1v.d[w15, 0]}, p0, [x16]\n"
        "st1d {za5v.d[w15, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "st1d {za1v.d[w15, 1]}, p0, [x16]\n"
        "st1d {za5v.d[w15, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za2v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za6v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za2v.d[w12, 1]}, p0, [x16]\n"
        "st1d {za6v.d[w12, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za2v.d[w13, 0]}, p0, [x16]\n"
        "st1d {za6v.d[w13, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za2v.d[w13, 1]}, p0, [x16]\n"
        "st1d {za6v.d[w13, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za2v.d[w14, 0]}, p0, [x16]\n"
        "st1d {za6v.d[w14, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za2v.d[w14, 1]}, p0, [x16]\n"
        "st1d {za6v.d[w14, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za2v.d[w15, 0]}, p0, [x16]\n"
        "st1d {za6v.d[w15, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za2v.d[w15, 1]}, p0, [x16]\n"
        "st1d {za6v.d[w15, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za3v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za3v.d[w12, 1]}, p0, [x16]\n"
        "st1d {za7v.d[w12, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za3v.d[w13, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w13, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za3v.d[w13, 1]}, p0, [x16]\n"
        "st1d {za7v.d[w13, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za3v.d[w14, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w14, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za3v.d[w14, 1]}, p0, [x16]\n"
        "st1d {za7v.d[w14, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za3v.d[w15, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w15, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za3v.d[w15, 1]}, p0, [x16]\n"
        "st1d {za7v.d[w15, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "b 69f\n"
        "68:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "prfm pldl2keep, [x19, x20]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "prfm pldl1keep, [x5, #64]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "prfm pldl1keep, [x5, #192]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "prfm pldl2keep, [x19]\n"
        "prfm pldl2keep, [x19, x20]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "prfm pldl1keep, [x5, #320]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "prfm pldl1keep, [x5, #448]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "prfm pldl2keep, [x19, x20]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "mov x16, x24\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "mov w13, #2\n"
        "mov w14, #4\n"
        "mov w15, #6\n"
        "st1d {za0v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za4v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za0v.d[w12, 1]}, p0, [x16]\n"
        "st1d {za4v.d[w12, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za0v.d[w13, 0]}, p0, [x16]\n"
        "st1d {za4v.d[w13, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za0v.d[w13, 1]}, p0, [x16]\n"
        "st1d {za4v.d[w13, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za0v.d[w14, 0]}, p0, [x16]\n"
        "st1d {za4v.d[w14, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za0v.d[w14, 1]}, p0, [x16]\n"
        "st1d {za4v.d[w14, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za0v.d[w15, 0]}, p0, [x16]\n"
        "st1d {za4v.d[w15, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za0v.d[w15, 1]}, p0, [x16]\n"
        "st1d {za4v.d[w15, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za1v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za5v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za1v.d[w12, 1]}, p0, [x16]\n"
        "st1d {za5v.d[w12, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za1v.d[w13, 0]}, p0, [x16]\n"
        "st1d {za5v.d[w13, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za1v.d[w13, 1]}, p0, [x16]\n"
        "st1d {za5v.d[w13, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za1v.d[w14, 0]}, p0, [x16]\n"
        "st1d {za5v.d[w14, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za1v.d[w14, 1]}, p0, [x16]\n"
        "st1d {za5v.d[w14, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za1v.d[w15, 0]}, p0, [x16]\n"
        "st1d {za5v.d[w15, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za1v.d[w15, 1]}, p0, [x16]\n"
        "st1d {za5v.d[w15, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za2v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za6v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za2v.d[w12, 1]}, p0, [x16]\n"
        "st1d {za6v.d[w12, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za2v.d[w13, 0]}, p0, [x16]\n"
        "st1d {za6v.d[w13, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za2v.d[w13, 1]}, p0, [x16]\n"
        "st1d {za6v.d[w13, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za2v.d[w14, 0]}, p0, [x16]\n"
        "st1d {za6v.d[w14, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za2v.d[w14, 1]}, p0, [x16]\n"
        "st1d {za6v.d[w14, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za2v.d[w15, 0]}, p0, [x16]\n"
        "st1d {za6v.d[w15, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za2v.d[w15, 1]}, p0, [x16]\n"
        "st1d {za6v.d[w15, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za3v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za3v.d[w12, 1]}, p0, [x16]\n"
        "st1d {za7v.d[w12, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za3v.d[w13, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w13, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za3v.d[w13, 1]}, p0, [x16]\n"
        "st1d {za7v.d[w13, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za3v.d[w14, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w14, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za3v.d[w14, 1]}, p0, [x16]\n"
        "st1d {za7v.d[w14, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za3v.d[w15, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w15, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "st1d {za3v.d[w15, 1]}, p0, [x16]\n"
        "st1d {za7v.d[w15, 1]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "69:\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "subs x7, x7, #1\n"
        "b.ne 80b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","cc","x6","x7","x5","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}

// AC N32 tail: prefetch the next A row without advancing to a nonexistent B panel.
__attribute__((noinline, aligned(64))) static void sme_mrun_ac_n32(const SmeMRun &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "ldr x20, [x28, #72]\n"
        "neg x20, x20\n"
        "ldr x6, [x28, #96]\n"
        "ldr x7, [x28, #104]\n"
        "80:\n"
        "ldr x22, [x6], #8\n"
        "ldr x26, [x6]\n"
        "ldr x27, [x28]\n"
        "ldr x23, [x28, #24]\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "mov x19, x26\n"

        "zero {za}\n"

        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z6.d}, p0/z, [x9]\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z12.d}, p0/z, [x9]\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z18.d}, p0/z, [x9]\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #316\n" "b.ne 97f\n" "add x5, x24, #4096\n" "96:\n" "prfm pldl2keep, [x5]\n" "add x5, x5, #64\n" "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "prfm pldl2keep, [x19]\n" "prfm pldl2keep, [x19, #64]\n" "prfm pldl2keep, [x19, #128]\n" "prfm pldl2keep, [x19, #192]\n" "add x19, x19, #256\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "prfm pldl2keep, [x19]\n" "prfm pldl2keep, [x19, #64]\n" "prfm pldl2keep, [x19, #128]\n" "prfm pldl2keep, [x19, #192]\n" "add x19, x19, #256\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "sub x11, x11, #4\n" "cmp x11, #64\n" "b.ge 96b\n" "97:\n"
        "cmp x11, #32\n"
        "b.lt 94f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "prfm pldl2keep, [x19]\n" "prfm pldl2keep, [x19, #64]\n" "prfm pldl2keep, [x19, #128]\n" "prfm pldl2keep, [x19, #192]\n" "add x19, x19, #256\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "prfm pldl2keep, [x19]\n" "prfm pldl2keep, [x19, #64]\n" "prfm pldl2keep, [x19, #128]\n" "prfm pldl2keep, [x19, #192]\n" "add x19, x19, #256\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #32\n"
        "b.ge 1b\n"
        "94:\n"
        "add x5, x24, #4096\n"
        "cmp x27, #1\n"
        "b.ne 98f\n"
        "nop\n"
        "98:\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "95:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n" "prfm pldl2keep, [x19, #64]\n" "prfm pldl2keep, [x19, #128]\n" "prfm pldl2keep, [x19, #192]\n" "add x19, x19, #256\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #64]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #192]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "prfm pldl2keep, [x19]\n" "prfm pldl2keep, [x19, #64]\n" "prfm pldl2keep, [x19, #128]\n" "prfm pldl2keep, [x19, #192]\n" "add x19, x19, #256\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #320]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #448]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "add x5, x5, #512\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 95b\n"
        "8:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n" "prfm pldl2keep, [x19, #64]\n" "prfm pldl2keep, [x19, #128]\n" "prfm pldl2keep, [x19, #192]\n" "add x19, x19, #256\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "prfm pldl1keep, [x5, #64]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "prfm pldl1keep, [x5, #192]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "prfm pldl2keep, [x19]\n" "prfm pldl2keep, [x19, #64]\n" "prfm pldl2keep, [x19, #128]\n" "prfm pldl2keep, [x19, #192]\n" "add x19, x19, #256\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "prfm pldl1keep, [x5, #320]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "prfm pldl1keep, [x5, #448]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n" "prfm pldl2keep, [x19, #64]\n" "prfm pldl2keep, [x19, #128]\n" "prfm pldl2keep, [x19, #192]\n" "add x19, x19, #256\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "93:\n"
        "st1d {za0v.d[w12, 0]}, p0, [x13]\n"
        "st1d {za4v.d[w12, 0]}, p0, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "st1d {za1v.d[w12, 0]}, p0, [x14]\n"
        "st1d {za5v.d[w12, 0]}, p0, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "st1d {za2v.d[w12, 0]}, p0, [x15]\n"
        "st1d {za6v.d[w12, 0]}, p0, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "st1d {za3v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "subs x7, x7, #1\n"
        "b.ne 80b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","cc","x6","x7","x5","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}

__attribute__((noinline, aligned(64))) static void sme_mrun_accumulate_reset(const SmeMRun &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "ldr x6, [x28, #96]\n"
        "ldr x7, [x28, #104]\n"
        "80:\n"
        "ldr x22, [x6], #8\n"
        "ldr x26, [x6]\n"
        "ldr x27, [x28]\n"
        "ldr x23, [x28, #24]\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "mov x19, x26\n"

        "zero {za}\n"

        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"

        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"

        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"

        "ld1d {z0.d}, p0/z, [x9]\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z6.d}, p0/z, [x9]\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z12.d}, p0/z, [x9]\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z18.d}, p0/z, [x9]\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"

        "sub x11, x11, #4\n"

        "cmp x11, #316\n" "b.ne 97f\n" "add x5, x24, #4096\n" "96:\n" "prfm pldl2keep, [x5]\n" "add x5, x5, #64\n" "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "prfm pldl2keep, [x19]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "prfm pldl2keep, [x19]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "sub x11, x11, #4\n" "cmp x11, #64\n" "b.ge 96b\n" "97:\n"

        "cmp x11, #32\n"
        "b.lt 94f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "prfm pldl2keep, [x19]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "prfm pldl2keep, [x19]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #32\n"
        "b.ge 1b\n"
        "94:\n"
        "add x5, x24, #4096\n"
        "cmp x27, #1\n"
        "b.ne 98f\n"
        "nop\n"
        "98:\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "95:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #64]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #192]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #320]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #448]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "add x5, x5, #512\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 95b\n"
        "8:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "prfm pldl1keep, [x5, #64]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "prfm pldl1keep, [x5, #192]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "prfm pldl1keep, [x5, #320]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "prfm pldl1keep, [x5, #448]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "93:\n"
        "st1d {za0v.d[w12, 0]}, p0, [x13]\n"
        "st1d {za4v.d[w12, 0]}, p0, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "st1d {za1v.d[w12, 0]}, p0, [x14]\n"
        "st1d {za5v.d[w12, 0]}, p0, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "st1d {za2v.d[w12, 0]}, p0, [x15]\n"
        "st1d {za6v.d[w12, 0]}, p0, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "st1d {za3v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "subs x7, x7, #1\n"
        "b.eq 81f\n"
        "smstop\n"
        "smstart\n"
        "ptrue p0.d\n"
        "b 80b\n"
        "81:\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","cc","x6","x7","x5","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}

__attribute__((noinline, aligned(64))) static void sme_npanel_pipeline_c64_spread32_m8(const SmeCPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #32\n"
        "b.lt 94f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #32\n"
        "b.ge 1b\n"
        "94:\n"
        "add x5, x24, #4096\n"
        "cmp x27, #1\n"
        "b.ne 98f\n"
        "ldr x5, [x28, #88]\n"
        "98:\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "95:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #64]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #192]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #320]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #448]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "add x5, x5, #512\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 95b\n"
        "8:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "prfm pldl1keep, [x5, #64]\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "prfm pldl1keep, [x5, #192]\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "prfm pldl1keep, [x5, #320]\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "prfm pldl1keep, [x5, #448]\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "93:\n"
        "st1d {za0v.d[w12, 0]}, p0, [x13]\n"
        "st1d {za4v.d[w12, 0]}, p0, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "st1d {za1v.d[w12, 0]}, p0, [x14]\n"
        "st1d {za5v.d[w12, 0]}, p0, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "st1d {za2v.d[w12, 0]}, p0, [x15]\n"
        "st1d {za6v.d[w12, 0]}, p0, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "st1d {za3v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","cc","x5","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}
__attribute__((noinline, aligned(64))) static void sme_npanel_prefetch_a_c64_spread32_m8(const SmeCPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "mov x19, x26\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #32\n"
        "b.lt 94f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #32\n"
        "b.ge 1b\n"
        "94:\n"
        "add x5, x24, #4096\n"
        "cmp x27, #1\n"
        "b.ne 98f\n"
        "ldr x5, [x28, #88]\n"
        "98:\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "95:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #64]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #192]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #320]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #448]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "add x5, x5, #512\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 95b\n"
        "8:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "prfm pldl1keep, [x5, #64]\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "prfm pldl1keep, [x5, #192]\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "prfm pldl1keep, [x5, #320]\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "prfm pldl1keep, [x5, #448]\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "93:\n"
        "st1d {za0v.d[w12, 0]}, p0, [x13]\n"
        "st1d {za4v.d[w12, 0]}, p0, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "st1d {za1v.d[w12, 0]}, p0, [x14]\n"
        "st1d {za5v.d[w12, 0]}, p0, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "st1d {za2v.d[w12, 0]}, p0, [x15]\n"
        "st1d {za6v.d[w12, 0]}, p0, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "st1d {za3v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","cc","x5","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}

struct SmeOPanel {
    SmeNPanel inner;
    double *output;
    long row_bytes, col_bytes, rows, columns;
};
static_assert(sizeof(SmeOPanel) == 128, "output descriptor layout");

struct SmeJoinedPanel {
    SmeOPanel output;
    const double *previous;
    long plane_stride;
};
static_assert(sizeof(SmeJoinedPanel) == 144, "joined descriptor layout");
__attribute__((noinline, aligned(64))) static void sme_joined_r4_pipeline(const SmeJoinedPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z6.d}, p0/z, [x9]\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z12.d}, p0/z, [x9]\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z18.d}, p0/z, [x9]\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 1b\n"
        "8:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "ldr x9, [x28]\n"
        "sub x9, x9, x27\n"
        "lsl x10, x9, #5\n"
        "ldr x19, [x28, #104]\n"
        "ldr x13, [x28, #88]\n"
        "madd x13, x10, x19, x13\n"
        "ldr x20, [x28, #96]\n"
        "add x14, x13, x20, lsl #3\n"
        "ldr x21, [x28, #112]\n"
        "ldr x11, [x28, #120]\n"
        "sub x11, x11, x10\n"
        "ldr x16, [x28, #128]\n"
        "add x9, x16, x9, lsl #12\n"
        "ldr x10, [x28, #136]\n"
        "mov x17, #128\n"
        "index z8.d, #0, x17\n"
        "mov w12, #0\n"
        "93:\n"
        "cmp x12, x21\n"
        "b.ge 94f\n"
        "mov x15, x13\n"
        "mov x16, x9\n"
        "mov x19, #0\n"
        "whilelt p1.d, x19, x11\n"
        "mova z0.d, p0/m, za0h.d[w12, 0]\n"
        "ld1d {z1.d}, p1/z, [x16, z8.d]\n"
        "add x17, x16, x10\n"
        "ld1d {z2.d}, p1/z, [x17, z8.d]\n"
        "add x17, x17, x10\n"
        "ld1d {z3.d}, p1/z, [x17, z8.d]\n"
        "zip1 z4.d, z1.d, z3.d\n"
        "zip2 z5.d, z1.d, z3.d\n"
        "zip1 z6.d, z2.d, z0.d\n"
        "zip2 z7.d, z2.d, z0.d\n"
        "zip1 z0.d, z4.d, z6.d\n"
        "zip2 z1.d, z4.d, z6.d\n"
        "zip1 z2.d, z5.d, z7.d\n"
        "zip2 z3.d, z5.d, z7.d\n"
        "sub x19, x11, x19\n"
        "lsl x19, x19, #2\n"
        "mov x17, #0\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z0.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z1.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z2.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z3.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "add x16, x16, #1024\n"
        "mov x19, #8\n"
        "whilelt p1.d, x19, x11\n"
        "mova z0.d, p0/m, za1h.d[w12, 0]\n"
        "ld1d {z1.d}, p1/z, [x16, z8.d]\n"
        "add x17, x16, x10\n"
        "ld1d {z2.d}, p1/z, [x17, z8.d]\n"
        "add x17, x17, x10\n"
        "ld1d {z3.d}, p1/z, [x17, z8.d]\n"
        "zip1 z4.d, z1.d, z3.d\n"
        "zip2 z5.d, z1.d, z3.d\n"
        "zip1 z6.d, z2.d, z0.d\n"
        "zip2 z7.d, z2.d, z0.d\n"
        "zip1 z0.d, z4.d, z6.d\n"
        "zip2 z1.d, z4.d, z6.d\n"
        "zip1 z2.d, z5.d, z7.d\n"
        "zip2 z3.d, z5.d, z7.d\n"
        "sub x19, x11, x19\n"
        "lsl x19, x19, #2\n"
        "mov x17, #0\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z0.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z1.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z2.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z3.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "add x16, x16, #1024\n"
        "mov x19, #16\n"
        "whilelt p1.d, x19, x11\n"
        "mova z0.d, p0/m, za2h.d[w12, 0]\n"
        "ld1d {z1.d}, p1/z, [x16, z8.d]\n"
        "add x17, x16, x10\n"
        "ld1d {z2.d}, p1/z, [x17, z8.d]\n"
        "add x17, x17, x10\n"
        "ld1d {z3.d}, p1/z, [x17, z8.d]\n"
        "zip1 z4.d, z1.d, z3.d\n"
        "zip2 z5.d, z1.d, z3.d\n"
        "zip1 z6.d, z2.d, z0.d\n"
        "zip2 z7.d, z2.d, z0.d\n"
        "zip1 z0.d, z4.d, z6.d\n"
        "zip2 z1.d, z4.d, z6.d\n"
        "zip1 z2.d, z5.d, z7.d\n"
        "zip2 z3.d, z5.d, z7.d\n"
        "sub x19, x11, x19\n"
        "lsl x19, x19, #2\n"
        "mov x17, #0\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z0.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z1.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z2.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z3.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "add x16, x16, #1024\n"
        "mov x19, #24\n"
        "whilelt p1.d, x19, x11\n"
        "mova z0.d, p0/m, za3h.d[w12, 0]\n"
        "ld1d {z1.d}, p1/z, [x16, z8.d]\n"
        "add x17, x16, x10\n"
        "ld1d {z2.d}, p1/z, [x17, z8.d]\n"
        "add x17, x17, x10\n"
        "ld1d {z3.d}, p1/z, [x17, z8.d]\n"
        "zip1 z4.d, z1.d, z3.d\n"
        "zip2 z5.d, z1.d, z3.d\n"
        "zip1 z6.d, z2.d, z0.d\n"
        "zip2 z7.d, z2.d, z0.d\n"
        "zip1 z0.d, z4.d, z6.d\n"
        "zip2 z1.d, z4.d, z6.d\n"
        "zip1 z2.d, z5.d, z7.d\n"
        "zip2 z3.d, z5.d, z7.d\n"
        "sub x19, x11, x19\n"
        "lsl x19, x19, #2\n"
        "mov x17, #0\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z0.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z1.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z2.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z3.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "add x16, x16, #1024\n"
        "add x16, x12, #8\n"
        "cmp x16, x21\n"
        "b.ge 95f\n"
        "mov x15, x14\n"
        "add x16, x9, #64\n"
        "mov x19, #0\n"
        "whilelt p1.d, x19, x11\n"
        "mova z0.d, p0/m, za4h.d[w12, 0]\n"
        "ld1d {z1.d}, p1/z, [x16, z8.d]\n"
        "add x17, x16, x10\n"
        "ld1d {z2.d}, p1/z, [x17, z8.d]\n"
        "add x17, x17, x10\n"
        "ld1d {z3.d}, p1/z, [x17, z8.d]\n"
        "zip1 z4.d, z1.d, z3.d\n"
        "zip2 z5.d, z1.d, z3.d\n"
        "zip1 z6.d, z2.d, z0.d\n"
        "zip2 z7.d, z2.d, z0.d\n"
        "zip1 z0.d, z4.d, z6.d\n"
        "zip2 z1.d, z4.d, z6.d\n"
        "zip1 z2.d, z5.d, z7.d\n"
        "zip2 z3.d, z5.d, z7.d\n"
        "sub x19, x11, x19\n"
        "lsl x19, x19, #2\n"
        "mov x17, #0\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z0.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z1.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z2.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z3.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "add x16, x16, #1024\n"
        "mov x19, #8\n"
        "whilelt p1.d, x19, x11\n"
        "mova z0.d, p0/m, za5h.d[w12, 0]\n"
        "ld1d {z1.d}, p1/z, [x16, z8.d]\n"
        "add x17, x16, x10\n"
        "ld1d {z2.d}, p1/z, [x17, z8.d]\n"
        "add x17, x17, x10\n"
        "ld1d {z3.d}, p1/z, [x17, z8.d]\n"
        "zip1 z4.d, z1.d, z3.d\n"
        "zip2 z5.d, z1.d, z3.d\n"
        "zip1 z6.d, z2.d, z0.d\n"
        "zip2 z7.d, z2.d, z0.d\n"
        "zip1 z0.d, z4.d, z6.d\n"
        "zip2 z1.d, z4.d, z6.d\n"
        "zip1 z2.d, z5.d, z7.d\n"
        "zip2 z3.d, z5.d, z7.d\n"
        "sub x19, x11, x19\n"
        "lsl x19, x19, #2\n"
        "mov x17, #0\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z0.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z1.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z2.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z3.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "add x16, x16, #1024\n"
        "mov x19, #16\n"
        "whilelt p1.d, x19, x11\n"
        "mova z0.d, p0/m, za6h.d[w12, 0]\n"
        "ld1d {z1.d}, p1/z, [x16, z8.d]\n"
        "add x17, x16, x10\n"
        "ld1d {z2.d}, p1/z, [x17, z8.d]\n"
        "add x17, x17, x10\n"
        "ld1d {z3.d}, p1/z, [x17, z8.d]\n"
        "zip1 z4.d, z1.d, z3.d\n"
        "zip2 z5.d, z1.d, z3.d\n"
        "zip1 z6.d, z2.d, z0.d\n"
        "zip2 z7.d, z2.d, z0.d\n"
        "zip1 z0.d, z4.d, z6.d\n"
        "zip2 z1.d, z4.d, z6.d\n"
        "zip1 z2.d, z5.d, z7.d\n"
        "zip2 z3.d, z5.d, z7.d\n"
        "sub x19, x11, x19\n"
        "lsl x19, x19, #2\n"
        "mov x17, #0\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z0.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z1.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z2.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z3.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "add x16, x16, #1024\n"
        "mov x19, #24\n"
        "whilelt p1.d, x19, x11\n"
        "mova z0.d, p0/m, za7h.d[w12, 0]\n"
        "ld1d {z1.d}, p1/z, [x16, z8.d]\n"
        "add x17, x16, x10\n"
        "ld1d {z2.d}, p1/z, [x17, z8.d]\n"
        "add x17, x17, x10\n"
        "ld1d {z3.d}, p1/z, [x17, z8.d]\n"
        "zip1 z4.d, z1.d, z3.d\n"
        "zip2 z5.d, z1.d, z3.d\n"
        "zip1 z6.d, z2.d, z0.d\n"
        "zip2 z7.d, z2.d, z0.d\n"
        "zip1 z0.d, z4.d, z6.d\n"
        "zip2 z1.d, z4.d, z6.d\n"
        "zip1 z2.d, z5.d, z7.d\n"
        "zip2 z3.d, z5.d, z7.d\n"
        "sub x19, x11, x19\n"
        "lsl x19, x19, #2\n"
        "mov x17, #0\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z0.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z1.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z2.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z3.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "add x16, x16, #1024\n"
        "95:\n"
        "add x13, x13, x20\n"
        "add x14, x14, x20\n"
        "add x9, x9, #8\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "94:\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","cc","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}
__attribute__((noinline, aligned(64))) static void sme_joined_r4_prefetch_a(const SmeJoinedPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "mov x19, x26\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z6.d}, p0/z, [x9]\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z12.d}, p0/z, [x9]\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z18.d}, p0/z, [x9]\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "prfm pldl2keep, [x19]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "prfm pldl2keep, [x19]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 1b\n"
        "8:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "ldr x9, [x28]\n"
        "sub x9, x9, x27\n"
        "lsl x10, x9, #5\n"
        "ldr x19, [x28, #104]\n"
        "ldr x13, [x28, #88]\n"
        "madd x13, x10, x19, x13\n"
        "ldr x20, [x28, #96]\n"
        "add x14, x13, x20, lsl #3\n"
        "ldr x21, [x28, #112]\n"
        "ldr x11, [x28, #120]\n"
        "sub x11, x11, x10\n"
        "ldr x16, [x28, #128]\n"
        "add x9, x16, x9, lsl #12\n"
        "ldr x10, [x28, #136]\n"
        "mov x17, #128\n"
        "index z8.d, #0, x17\n"
        "mov w12, #0\n"
        "93:\n"
        "cmp x12, x21\n"
        "b.ge 94f\n"
        "mov x15, x13\n"
        "mov x16, x9\n"
        "mov x19, #0\n"
        "whilelt p1.d, x19, x11\n"
        "mova z0.d, p0/m, za0h.d[w12, 0]\n"
        "ld1d {z1.d}, p1/z, [x16, z8.d]\n"
        "add x17, x16, x10\n"
        "ld1d {z2.d}, p1/z, [x17, z8.d]\n"
        "add x17, x17, x10\n"
        "ld1d {z3.d}, p1/z, [x17, z8.d]\n"
        "zip1 z4.d, z1.d, z3.d\n"
        "zip2 z5.d, z1.d, z3.d\n"
        "zip1 z6.d, z2.d, z0.d\n"
        "zip2 z7.d, z2.d, z0.d\n"
        "zip1 z0.d, z4.d, z6.d\n"
        "zip2 z1.d, z4.d, z6.d\n"
        "zip1 z2.d, z5.d, z7.d\n"
        "zip2 z3.d, z5.d, z7.d\n"
        "sub x19, x11, x19\n"
        "lsl x19, x19, #2\n"
        "mov x17, #0\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z0.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z1.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z2.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z3.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "add x16, x16, #1024\n"
        "mov x19, #8\n"
        "whilelt p1.d, x19, x11\n"
        "mova z0.d, p0/m, za1h.d[w12, 0]\n"
        "ld1d {z1.d}, p1/z, [x16, z8.d]\n"
        "add x17, x16, x10\n"
        "ld1d {z2.d}, p1/z, [x17, z8.d]\n"
        "add x17, x17, x10\n"
        "ld1d {z3.d}, p1/z, [x17, z8.d]\n"
        "zip1 z4.d, z1.d, z3.d\n"
        "zip2 z5.d, z1.d, z3.d\n"
        "zip1 z6.d, z2.d, z0.d\n"
        "zip2 z7.d, z2.d, z0.d\n"
        "zip1 z0.d, z4.d, z6.d\n"
        "zip2 z1.d, z4.d, z6.d\n"
        "zip1 z2.d, z5.d, z7.d\n"
        "zip2 z3.d, z5.d, z7.d\n"
        "sub x19, x11, x19\n"
        "lsl x19, x19, #2\n"
        "mov x17, #0\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z0.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z1.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z2.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z3.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "add x16, x16, #1024\n"
        "mov x19, #16\n"
        "whilelt p1.d, x19, x11\n"
        "mova z0.d, p0/m, za2h.d[w12, 0]\n"
        "ld1d {z1.d}, p1/z, [x16, z8.d]\n"
        "add x17, x16, x10\n"
        "ld1d {z2.d}, p1/z, [x17, z8.d]\n"
        "add x17, x17, x10\n"
        "ld1d {z3.d}, p1/z, [x17, z8.d]\n"
        "zip1 z4.d, z1.d, z3.d\n"
        "zip2 z5.d, z1.d, z3.d\n"
        "zip1 z6.d, z2.d, z0.d\n"
        "zip2 z7.d, z2.d, z0.d\n"
        "zip1 z0.d, z4.d, z6.d\n"
        "zip2 z1.d, z4.d, z6.d\n"
        "zip1 z2.d, z5.d, z7.d\n"
        "zip2 z3.d, z5.d, z7.d\n"
        "sub x19, x11, x19\n"
        "lsl x19, x19, #2\n"
        "mov x17, #0\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z0.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z1.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z2.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z3.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "add x16, x16, #1024\n"
        "mov x19, #24\n"
        "whilelt p1.d, x19, x11\n"
        "mova z0.d, p0/m, za3h.d[w12, 0]\n"
        "ld1d {z1.d}, p1/z, [x16, z8.d]\n"
        "add x17, x16, x10\n"
        "ld1d {z2.d}, p1/z, [x17, z8.d]\n"
        "add x17, x17, x10\n"
        "ld1d {z3.d}, p1/z, [x17, z8.d]\n"
        "zip1 z4.d, z1.d, z3.d\n"
        "zip2 z5.d, z1.d, z3.d\n"
        "zip1 z6.d, z2.d, z0.d\n"
        "zip2 z7.d, z2.d, z0.d\n"
        "zip1 z0.d, z4.d, z6.d\n"
        "zip2 z1.d, z4.d, z6.d\n"
        "zip1 z2.d, z5.d, z7.d\n"
        "zip2 z3.d, z5.d, z7.d\n"
        "sub x19, x11, x19\n"
        "lsl x19, x19, #2\n"
        "mov x17, #0\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z0.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z1.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z2.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z3.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "add x16, x16, #1024\n"
        "add x16, x12, #8\n"
        "cmp x16, x21\n"
        "b.ge 95f\n"
        "mov x15, x14\n"
        "add x16, x9, #64\n"
        "mov x19, #0\n"
        "whilelt p1.d, x19, x11\n"
        "mova z0.d, p0/m, za4h.d[w12, 0]\n"
        "ld1d {z1.d}, p1/z, [x16, z8.d]\n"
        "add x17, x16, x10\n"
        "ld1d {z2.d}, p1/z, [x17, z8.d]\n"
        "add x17, x17, x10\n"
        "ld1d {z3.d}, p1/z, [x17, z8.d]\n"
        "zip1 z4.d, z1.d, z3.d\n"
        "zip2 z5.d, z1.d, z3.d\n"
        "zip1 z6.d, z2.d, z0.d\n"
        "zip2 z7.d, z2.d, z0.d\n"
        "zip1 z0.d, z4.d, z6.d\n"
        "zip2 z1.d, z4.d, z6.d\n"
        "zip1 z2.d, z5.d, z7.d\n"
        "zip2 z3.d, z5.d, z7.d\n"
        "sub x19, x11, x19\n"
        "lsl x19, x19, #2\n"
        "mov x17, #0\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z0.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z1.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z2.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z3.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "add x16, x16, #1024\n"
        "mov x19, #8\n"
        "whilelt p1.d, x19, x11\n"
        "mova z0.d, p0/m, za5h.d[w12, 0]\n"
        "ld1d {z1.d}, p1/z, [x16, z8.d]\n"
        "add x17, x16, x10\n"
        "ld1d {z2.d}, p1/z, [x17, z8.d]\n"
        "add x17, x17, x10\n"
        "ld1d {z3.d}, p1/z, [x17, z8.d]\n"
        "zip1 z4.d, z1.d, z3.d\n"
        "zip2 z5.d, z1.d, z3.d\n"
        "zip1 z6.d, z2.d, z0.d\n"
        "zip2 z7.d, z2.d, z0.d\n"
        "zip1 z0.d, z4.d, z6.d\n"
        "zip2 z1.d, z4.d, z6.d\n"
        "zip1 z2.d, z5.d, z7.d\n"
        "zip2 z3.d, z5.d, z7.d\n"
        "sub x19, x11, x19\n"
        "lsl x19, x19, #2\n"
        "mov x17, #0\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z0.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z1.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z2.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z3.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "add x16, x16, #1024\n"
        "mov x19, #16\n"
        "whilelt p1.d, x19, x11\n"
        "mova z0.d, p0/m, za6h.d[w12, 0]\n"
        "ld1d {z1.d}, p1/z, [x16, z8.d]\n"
        "add x17, x16, x10\n"
        "ld1d {z2.d}, p1/z, [x17, z8.d]\n"
        "add x17, x17, x10\n"
        "ld1d {z3.d}, p1/z, [x17, z8.d]\n"
        "zip1 z4.d, z1.d, z3.d\n"
        "zip2 z5.d, z1.d, z3.d\n"
        "zip1 z6.d, z2.d, z0.d\n"
        "zip2 z7.d, z2.d, z0.d\n"
        "zip1 z0.d, z4.d, z6.d\n"
        "zip2 z1.d, z4.d, z6.d\n"
        "zip1 z2.d, z5.d, z7.d\n"
        "zip2 z3.d, z5.d, z7.d\n"
        "sub x19, x11, x19\n"
        "lsl x19, x19, #2\n"
        "mov x17, #0\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z0.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z1.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z2.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z3.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "add x16, x16, #1024\n"
        "mov x19, #24\n"
        "whilelt p1.d, x19, x11\n"
        "mova z0.d, p0/m, za7h.d[w12, 0]\n"
        "ld1d {z1.d}, p1/z, [x16, z8.d]\n"
        "add x17, x16, x10\n"
        "ld1d {z2.d}, p1/z, [x17, z8.d]\n"
        "add x17, x17, x10\n"
        "ld1d {z3.d}, p1/z, [x17, z8.d]\n"
        "zip1 z4.d, z1.d, z3.d\n"
        "zip2 z5.d, z1.d, z3.d\n"
        "zip1 z6.d, z2.d, z0.d\n"
        "zip2 z7.d, z2.d, z0.d\n"
        "zip1 z0.d, z4.d, z6.d\n"
        "zip2 z1.d, z4.d, z6.d\n"
        "zip1 z2.d, z5.d, z7.d\n"
        "zip2 z3.d, z5.d, z7.d\n"
        "sub x19, x11, x19\n"
        "lsl x19, x19, #2\n"
        "mov x17, #0\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z0.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z1.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z2.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "whilelt p2.d, x17, x19\n"
        "st1d {z3.d}, p2, [x15]\n"
        "add x15, x15, #64\n"
        "add x17, x17, #8\n"
        "add x16, x16, #1024\n"
        "95:\n"
        "add x13, x13, x20\n"
        "add x14, x14, x20\n"
        "add x9, x9, #8\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "94:\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","cc","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}

struct SmeTPanel {
    SmeNPanel inner;
    double *output;
    long r_stride;
};
static_assert(sizeof(SmeTPanel) == 104, "direct T descriptor layout");
__attribute__((noinline, aligned(64))) static void sme_tstore_pipeline(const SmeTPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z6.d}, p0/z, [x9]\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z12.d}, p0/z, [x9]\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z18.d}, p0/z, [x9]\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 1b\n"
        "8:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "ldr x9, [x28]\n"
        "sub x9, x9, x27\n"
        "lsl x9, x9, #10\n"
        "ldr x13, [x28, #88]\n"
        "add x13, x13, x9\n"
        "add x14, x13, #256\n"
        "add x15, x14, #256\n"
        "add x16, x15, #256\n"
        "ldr x9, [x28, #96]\n"
        "add x10, x9, x9, lsl #1\n"
        "neg x10, x10\n"
        "add x10, x10, #128\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "93:\n"
        "st1d {za0v.d[w12, 0]}, p0, [x13]\n"
        "st1d {za4v.d[w12, 0]}, p0, [x13, x21, lsl #3]\n"
        "st1d {za1v.d[w12, 0]}, p0, [x14]\n"
        "st1d {za5v.d[w12, 0]}, p0, [x14, x21, lsl #3]\n"
        "st1d {za2v.d[w12, 0]}, p0, [x15]\n"
        "st1d {za6v.d[w12, 0]}, p0, [x15, x21, lsl #3]\n"
        "st1d {za3v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add w12, w12, #1\n"
        "tst w12, #3\n"
        "csel x11, x10, x9, eq\n"
        "add x13, x13, x11\n"
        "add x14, x14, x11\n"
        "add x15, x15, x11\n"
        "add x16, x16, x11\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","cc","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}
__attribute__((noinline, aligned(64))) static void sme_tstore_prefetch_a(const SmeTPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "mov x19, x26\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z6.d}, p0/z, [x9]\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z12.d}, p0/z, [x9]\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z18.d}, p0/z, [x9]\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "prfm pldl2keep, [x19]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "prfm pldl2keep, [x19]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 1b\n"
        "8:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "ldr x9, [x28]\n"
        "sub x9, x9, x27\n"
        "lsl x9, x9, #10\n"
        "ldr x13, [x28, #88]\n"
        "add x13, x13, x9\n"
        "add x14, x13, #256\n"
        "add x15, x14, #256\n"
        "add x16, x15, #256\n"
        "ldr x9, [x28, #96]\n"
        "add x10, x9, x9, lsl #1\n"
        "neg x10, x10\n"
        "add x10, x10, #128\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "93:\n"
        "st1d {za0v.d[w12, 0]}, p0, [x13]\n"
        "st1d {za4v.d[w12, 0]}, p0, [x13, x21, lsl #3]\n"
        "st1d {za1v.d[w12, 0]}, p0, [x14]\n"
        "st1d {za5v.d[w12, 0]}, p0, [x14, x21, lsl #3]\n"
        "st1d {za2v.d[w12, 0]}, p0, [x15]\n"
        "st1d {za6v.d[w12, 0]}, p0, [x15, x21, lsl #3]\n"
        "st1d {za3v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add w12, w12, #1\n"
        "tst w12, #3\n"
        "csel x11, x10, x9, eq\n"
        "add x13, x13, x11\n"
        "add x14, x14, x11\n"
        "add x15, x15, x11\n"
        "add x16, x16, x11\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","cc","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}
__attribute__((noinline, aligned(64))) static void sme_tstore_pack_b_pipeline(const SmeTPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "mov x19, x25\n"
        "mov x20, #256\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "st1d {z8.d}, p0, [x19]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z9.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z10.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z11.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "st1d {z14.d}, p0, [x19]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z15.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z16.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z17.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "st1d {z20.d}, p0, [x19]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z21.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z22.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z23.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "st1d {z8.d}, p0, [x19]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z9.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z10.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z11.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "st1d {z14.d}, p0, [x19]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z15.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z16.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z17.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "st1d {z20.d}, p0, [x19]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z21.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z22.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z23.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 1b\n"
        "8:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "ldr x9, [x28]\n"
        "sub x9, x9, x27\n"
        "lsl x9, x9, #10\n"
        "ldr x13, [x28, #88]\n"
        "add x13, x13, x9\n"
        "add x14, x13, #256\n"
        "add x15, x14, #256\n"
        "add x16, x15, #256\n"
        "ldr x9, [x28, #96]\n"
        "add x10, x9, x9, lsl #1\n"
        "neg x10, x10\n"
        "add x10, x10, #128\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "93:\n"
        "st1d {za0v.d[w12, 0]}, p0, [x13]\n"
        "st1d {za4v.d[w12, 0]}, p0, [x13, x21, lsl #3]\n"
        "st1d {za1v.d[w12, 0]}, p0, [x14]\n"
        "st1d {za5v.d[w12, 0]}, p0, [x14, x21, lsl #3]\n"
        "st1d {za2v.d[w12, 0]}, p0, [x15]\n"
        "st1d {za6v.d[w12, 0]}, p0, [x15, x21, lsl #3]\n"
        "st1d {za3v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add w12, w12, #1\n"
        "tst w12, #3\n"
        "csel x11, x10, x9, eq\n"
        "add x13, x13, x11\n"
        "add x14, x14, x11\n"
        "add x15, x15, x11\n"
        "add x16, x16, x11\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","cc","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}

__attribute__((noinline, aligned(64))) static void sme_m8_n64_interleaved(const SmeCPanel &d) {
    if (d.n.k != 320 || d.n.panels != 4 || d.n.add != 1)
        std::abort();
    asm volatile(
".arch_extension sme-f64f64\n"
"mov x28,%[descriptor]\n"
"ldr x27,[x28]\n"
"ldr x22,[x28,#16]\n"
"ldr x23,[x28,#24]\n"
"ldr x24,[x28,#32]\n"
"ldr x26,[x28,#48]\n"
"smstart\n"
"ptrue p0.d\n"
"90:\n"
"mov x9,x22\n"
"mov x10,x23\n"
"ldr x17,[x28,#56]\n"
"add x20,x23,x17\n"
"ldr x11,[x28,#8]\n"
"mov x19,x26\n"
"zero {za}\n"
"mov x13,x24\n"
"add x14,x13,#1024\n"
"add x15,x14,#1024\n"
"add x16,x15,#1024\n"
"mov x21,#512\n"
"mov w12,#0\n"
"92:\n"
"ld1d {za0v.d[w12,0]},p0/z,[x13]\n"
"ld1d {za4v.d[w12,0]},p0/z,[x13,x21,lsl #3]\n"
"add x13,x13,#128\n"
"ld1d {za1v.d[w12,0]},p0/z,[x14]\n"
"ld1d {za5v.d[w12,0]},p0/z,[x14,x21,lsl #3]\n"
"add x14,x14,#128\n"
"ld1d {za2v.d[w12,0]},p0/z,[x15]\n"
"ld1d {za6v.d[w12,0]},p0/z,[x15,x21,lsl #3]\n"
"add x15,x15,#128\n"
"ld1d {za3v.d[w12,0]},p0/z,[x16]\n"
"ld1d {za7v.d[w12,0]},p0/z,[x16,x21,lsl #3]\n"
"add x16,x16,#128\n"
"add w12,w12,#1\n"
"cmp w12,#8\n"
"b.ne 92b\n"
"add x5,x24,#8192\n"
"cmp x27,#2\n"
"b.ne 97f\n"
"ldr x5,[x28,#88]\n"
"97:\n"
"ld1d {z0.d},p0/z,[x9]\n"
"ld1d {z1.d},p0/z,[x10,#0,mul vl]\n"
"ld1d {z2.d},p0/z,[x10,#1,mul vl]\n"
"ld1d {z3.d},p0/z,[x10,#2,mul vl]\n"
"ld1d {z4.d},p0/z,[x10,#3,mul vl]\n"
"ld1d {z5.d},p0/z,[x20,#0,mul vl]\n"
"ld1d {z6.d},p0/z,[x20,#1,mul vl]\n"
"ld1d {z7.d},p0/z,[x20,#2,mul vl]\n"
"ld1d {z8.d},p0/z,[x20,#3,mul vl]\n"
"add x9,x9,#128\n"
"add x10,x10,#256\n"
"add x20,x20,#256\n"
"ld1d {z9.d},p0/z,[x9]\n"
"ld1d {z10.d},p0/z,[x10,#0,mul vl]\n"
"ld1d {z11.d},p0/z,[x10,#1,mul vl]\n"
"ld1d {z12.d},p0/z,[x10,#2,mul vl]\n"
"ld1d {z13.d},p0/z,[x10,#3,mul vl]\n"
"ld1d {z14.d},p0/z,[x20,#0,mul vl]\n"
"ld1d {z15.d},p0/z,[x20,#1,mul vl]\n"
"ld1d {z16.d},p0/z,[x20,#2,mul vl]\n"
"ld1d {z17.d},p0/z,[x20,#3,mul vl]\n"
"add x9,x9,#128\n"
"add x10,x10,#256\n"
"add x20,x20,#256\n"
"sub x11,x11,#2\n"
"1:\n"
"cmp x11,#32\n"
"b.lt 2f\n"
"fmopa za0.d,p0/m,p0/m,z0.d,z1.d\n"
"ld1d {z1.d},p0/z,[x10,#0,mul vl]\n"
"fmopa za1.d,p0/m,p0/m,z0.d,z2.d\n"
"ld1d {z2.d},p0/z,[x10,#1,mul vl]\n"
"fmopa za2.d,p0/m,p0/m,z0.d,z3.d\n"
"ld1d {z3.d},p0/z,[x10,#2,mul vl]\n"
"fmopa za3.d,p0/m,p0/m,z0.d,z4.d\n"
"ld1d {z4.d},p0/z,[x10,#3,mul vl]\n"
"fmopa za4.d,p0/m,p0/m,z0.d,z5.d\n"
"ld1d {z5.d},p0/z,[x20,#0,mul vl]\n"
"fmopa za5.d,p0/m,p0/m,z0.d,z6.d\n"
"ld1d {z6.d},p0/z,[x20,#1,mul vl]\n"
"fmopa za6.d,p0/m,p0/m,z0.d,z7.d\n"
"ld1d {z7.d},p0/z,[x20,#2,mul vl]\n"
"fmopa za7.d,p0/m,p0/m,z0.d,z8.d\n"
"ld1d {z8.d},p0/z,[x20,#3,mul vl]\n"
"cbz x19,96f\n"
"prfm pldl2keep,[x19]\n"
"add x19,x19,#64\n"
"96:\n"
"ld1d {z0.d},p0/z,[x9]\n"
"add x9,x9,#128\n"
"add x10,x10,#256\n"
"add x20,x20,#256\n"
"fmopa za0.d,p0/m,p0/m,z9.d,z10.d\n"
"ld1d {z10.d},p0/z,[x10,#0,mul vl]\n"
"fmopa za1.d,p0/m,p0/m,z9.d,z11.d\n"
"ld1d {z11.d},p0/z,[x10,#1,mul vl]\n"
"fmopa za2.d,p0/m,p0/m,z9.d,z12.d\n"
"ld1d {z12.d},p0/z,[x10,#2,mul vl]\n"
"fmopa za3.d,p0/m,p0/m,z9.d,z13.d\n"
"ld1d {z13.d},p0/z,[x10,#3,mul vl]\n"
"fmopa za4.d,p0/m,p0/m,z9.d,z14.d\n"
"ld1d {z14.d},p0/z,[x20,#0,mul vl]\n"
"fmopa za5.d,p0/m,p0/m,z9.d,z15.d\n"
"ld1d {z15.d},p0/z,[x20,#1,mul vl]\n"
"fmopa za6.d,p0/m,p0/m,z9.d,z16.d\n"
"ld1d {z16.d},p0/z,[x20,#2,mul vl]\n"
"fmopa za7.d,p0/m,p0/m,z9.d,z17.d\n"
"ld1d {z17.d},p0/z,[x20,#3,mul vl]\n"
"cbz x19,96f\n"
"prfm pldl2keep,[x19]\n"
"add x19,x19,#64\n"
"96:\n"
"ld1d {z9.d},p0/z,[x9]\n"
"add x9,x9,#128\n"
"add x10,x10,#256\n"
"add x20,x20,#256\n"
"sub x11,x11,#2\n"
"b 1b\n"
"2:\n"
"cbz x11,3f\n"
"fmopa za0.d,p0/m,p0/m,z0.d,z1.d\n"
"ld1d {z1.d},p0/z,[x10,#0,mul vl]\n"
"prfm pldl1keep,[x5,#0]\n"
"fmopa za1.d,p0/m,p0/m,z0.d,z2.d\n"
"ld1d {z2.d},p0/z,[x10,#1,mul vl]\n"
"fmopa za2.d,p0/m,p0/m,z0.d,z3.d\n"
"ld1d {z3.d},p0/z,[x10,#2,mul vl]\n"
"prfm pldl1keep,[x5,#64]\n"
"fmopa za3.d,p0/m,p0/m,z0.d,z4.d\n"
"ld1d {z4.d},p0/z,[x10,#3,mul vl]\n"
"fmopa za4.d,p0/m,p0/m,z0.d,z5.d\n"
"ld1d {z5.d},p0/z,[x20,#0,mul vl]\n"
"prfm pldl1keep,[x5,#128]\n"
"fmopa za5.d,p0/m,p0/m,z0.d,z6.d\n"
"ld1d {z6.d},p0/z,[x20,#1,mul vl]\n"
"fmopa za6.d,p0/m,p0/m,z0.d,z7.d\n"
"ld1d {z7.d},p0/z,[x20,#2,mul vl]\n"
"prfm pldl1keep,[x5,#192]\n"
"fmopa za7.d,p0/m,p0/m,z0.d,z8.d\n"
"ld1d {z8.d},p0/z,[x20,#3,mul vl]\n"
"cbz x19,96f\n"
"prfm pldl2keep,[x19]\n"
"add x19,x19,#64\n"
"96:\n"
"add x5,x5,#256\n"
"ld1d {z0.d},p0/z,[x9]\n"
"add x9,x9,#128\n"
"add x10,x10,#256\n"
"add x20,x20,#256\n"
"fmopa za0.d,p0/m,p0/m,z9.d,z10.d\n"
"ld1d {z10.d},p0/z,[x10,#0,mul vl]\n"
"prfm pldl1keep,[x5,#0]\n"
"fmopa za1.d,p0/m,p0/m,z9.d,z11.d\n"
"ld1d {z11.d},p0/z,[x10,#1,mul vl]\n"
"fmopa za2.d,p0/m,p0/m,z9.d,z12.d\n"
"ld1d {z12.d},p0/z,[x10,#2,mul vl]\n"
"prfm pldl1keep,[x5,#64]\n"
"fmopa za3.d,p0/m,p0/m,z9.d,z13.d\n"
"ld1d {z13.d},p0/z,[x10,#3,mul vl]\n"
"fmopa za4.d,p0/m,p0/m,z9.d,z14.d\n"
"ld1d {z14.d},p0/z,[x20,#0,mul vl]\n"
"prfm pldl1keep,[x5,#128]\n"
"fmopa za5.d,p0/m,p0/m,z9.d,z15.d\n"
"ld1d {z15.d},p0/z,[x20,#1,mul vl]\n"
"fmopa za6.d,p0/m,p0/m,z9.d,z16.d\n"
"ld1d {z16.d},p0/z,[x20,#2,mul vl]\n"
"prfm pldl1keep,[x5,#192]\n"
"fmopa za7.d,p0/m,p0/m,z9.d,z17.d\n"
"ld1d {z17.d},p0/z,[x20,#3,mul vl]\n"
"cbz x19,96f\n"
"prfm pldl2keep,[x19]\n"
"add x19,x19,#64\n"
"96:\n"
"add x5,x5,#256\n"
"ld1d {z9.d},p0/z,[x9]\n"
"add x9,x9,#128\n"
"add x10,x10,#256\n"
"add x20,x20,#256\n"
"sub x11,x11,#2\n"
"b 2b\n"
"3:\n"
"fmopa za0.d,p0/m,p0/m,z0.d,z1.d\n"
"prfm pldl1keep,[x5,#0]\n"
"fmopa za1.d,p0/m,p0/m,z0.d,z2.d\n"
"fmopa za2.d,p0/m,p0/m,z0.d,z3.d\n"
"prfm pldl1keep,[x5,#64]\n"
"fmopa za3.d,p0/m,p0/m,z0.d,z4.d\n"
"fmopa za4.d,p0/m,p0/m,z0.d,z5.d\n"
"prfm pldl1keep,[x5,#128]\n"
"fmopa za5.d,p0/m,p0/m,z0.d,z6.d\n"
"fmopa za6.d,p0/m,p0/m,z0.d,z7.d\n"
"prfm pldl1keep,[x5,#192]\n"
"fmopa za7.d,p0/m,p0/m,z0.d,z8.d\n"
"cbz x19,96f\n"
"prfm pldl2keep,[x19]\n"
"add x19,x19,#64\n"
"96:\n"
"add x5,x5,#256\n"
"fmopa za0.d,p0/m,p0/m,z9.d,z10.d\n"
"prfm pldl1keep,[x5,#0]\n"
"fmopa za1.d,p0/m,p0/m,z9.d,z11.d\n"
"fmopa za2.d,p0/m,p0/m,z9.d,z12.d\n"
"prfm pldl1keep,[x5,#64]\n"
"fmopa za3.d,p0/m,p0/m,z9.d,z13.d\n"
"fmopa za4.d,p0/m,p0/m,z9.d,z14.d\n"
"prfm pldl1keep,[x5,#128]\n"
"fmopa za5.d,p0/m,p0/m,z9.d,z15.d\n"
"fmopa za6.d,p0/m,p0/m,z9.d,z16.d\n"
"prfm pldl1keep,[x5,#192]\n"
"fmopa za7.d,p0/m,p0/m,z9.d,z17.d\n"
"cbz x19,96f\n"
"prfm pldl2keep,[x19]\n"
"add x19,x19,#64\n"
"96:\n"
"add x5,x5,#256\n"
"mov x13,x24\n"
"add x14,x13,#1024\n"
"add x15,x14,#1024\n"
"add x16,x15,#1024\n"
"mov x21,#512\n"
"mov w12,#0\n"
"93:\n"
"st1d {za0v.d[w12,0]},p0,[x13]\n"
"st1d {za4v.d[w12,0]},p0,[x13,x21,lsl #3]\n"
"add x13,x13,#128\n"
"st1d {za1v.d[w12,0]},p0,[x14]\n"
"st1d {za5v.d[w12,0]},p0,[x14,x21,lsl #3]\n"
"add x14,x14,#128\n"
"st1d {za2v.d[w12,0]},p0,[x15]\n"
"st1d {za6v.d[w12,0]},p0,[x15,x21,lsl #3]\n"
"add x15,x15,#128\n"
"st1d {za3v.d[w12,0]},p0,[x16]\n"
"st1d {za7v.d[w12,0]},p0,[x16,x21,lsl #3]\n"
"add x16,x16,#128\n"
"add w12,w12,#1\n"
"cmp w12,#8\n"
"b.ne 93b\n"
"add x23,x23,x17,lsl #1\n"
"add x24,x24,#8192\n"
"cbz x26,98f\n"
"ldr x9,[x28,#72]\n"
"add x26,x26,x9,lsl #1\n"
"98:\n"
"subs x27,x27,#2\n"
"b.ne 90b\n"
"smstop\n"
::[descriptor]"r"(&d):"memory","cc","x5","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}
namespace tcopt {
constexpr int MR = 16, NR = 32;

constexpr int BM = 256, BN = 128, BK = 384;
constexpr int scratch_b_k = BK > 320 ? BK : 320;
static_assert(BM % MR == 0 && BN % NR == 0 && BK > 0, "invalid macro tile");
constexpr int owner_ordinary_rows = 1792;
constexpr int scratch_c_rows = 1792;
struct Scratch {
    double *a, *b, *c;
};
static Scratch *scratch = nullptr;
static int scratch_threads = 0;
static double *tail_k_workspace = nullptr;
static int tc1_tb_k6 = 14080;
static constexpr int tc1_tb_k6_peer = 14080, tc1_tb_k6_root = 10880;
// Classical GEMM helper balance from master; active TC2 uses Strassen products.
static constexpr int tc2_tb_seed_k = 1280;
static double *block6_workspace = nullptr;

constexpr size_t tail_k_capacity = 16ul << 20;
static char *scratch_arena = nullptr;
static size_t scratch_arena_bytes = 0;
constexpr size_t scratch_slot_bytes = size_t(BM * BK + BN * scratch_b_k + scratch_c_rows * BN) * sizeof(double);
static double *packed_a_workspace = nullptr;
constexpr size_t packed_a_capacity = 64ul << 20;
static double *tb_strassen_t_workspace=nullptr, *tb_strassen_products_workspace=nullptr;
constexpr size_t tb_strassen_t_bytes=size_t(8)*25*(1ul<<20);
static int tc_t_slots=8;
static size_t tb_strassen_t_alloc=tb_strassen_t_bytes;
constexpr size_t tb_strassen_products_bytes=112ul<<20;
static double *tc2_ac_products=nullptr;
constexpr size_t tc2_ac_product_bytes=224ul<<20;
constexpr size_t tc2_ac_operand_bytes=1024ul*4000*8;
// TC1 has finished before TC2. Six A operands fit in its 200 MiB T ring;
// the seventh uses the otherwise idle 64 MiB packed-A scratch.
static_assert(6*tc2_ac_operand_bytes<=tb_strassen_t_bytes,"six AC operands fit TC1 T ring");
static_assert(tc2_ac_operand_bytes<=packed_a_capacity,"seventh AC operand fits packed A");
static double* tc2_ac_operand(int which,int m) {
    return which<6 ? tb_strassen_t_workspace+size_t(which)*m*4000 : packed_a_workspace;
}
static void *aligned_buffer(size_t n) {
    void *p = nullptr;
    if (posix_memalign(&p, 64, n) != 0) {
        fprintf(stderr, "scratch allocation failed\n");
        std::abort();
    }
    return p;
}
static void *huge_buffer(size_t bytes) {
    const size_t huge = 2ul << 20;
    if (bytes % huge)
        std::abort();
    void *mapped = mmap(nullptr, bytes + huge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapped == MAP_FAILED)
        std::abort();
    uintptr_t raw = (uintptr_t)mapped, aligned = (raw + huge - 1) & ~(huge - 1);
    if (aligned > raw)
        munmap((void *)raw, aligned - raw);
    if (raw + bytes + huge > aligned + bytes)
        munmap((void *)(aligned + bytes), raw + huge - aligned);
    if (madvise((void *)aligned, bytes, MADV_HUGEPAGE))
        std::abort();
    return (void *)aligned;
}

__attribute__((constructor)) static void prepare_scratch() {
    scratch_threads = omp_get_max_threads();

    scratch_threads = std::max(scratch_threads, 38);

    scratch = (Scratch *)calloc(scratch_threads, sizeof(Scratch));
    if (!scratch)
        std::abort();
    tail_k_workspace = (double *)huge_buffer(tail_k_capacity);
    block6_workspace = (double *)huge_buffer(tail_k_capacity);
    const size_t huge = 2ul << 20;
    scratch_arena_bytes = (scratch_threads * scratch_slot_bytes + huge - 1) & ~(huge - 1);
    void *mapped = mmap(nullptr, scratch_arena_bytes + huge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapped == MAP_FAILED)
        std::abort();
    uintptr_t raw = (uintptr_t)mapped, aligned = (raw + huge - 1) & ~(huge - 1);
    if (aligned > raw)
        munmap((void *)raw, aligned - raw);
    if (raw + scratch_arena_bytes + huge > aligned + scratch_arena_bytes)
        munmap((void *)(aligned + scratch_arena_bytes), raw + huge - aligned);
    scratch_arena = (char *)aligned;
    if (madvise(scratch_arena, scratch_arena_bytes, MADV_HUGEPAGE))
        std::abort();
    packed_a_workspace = (double *)aligned_buffer(packed_a_capacity);
    // Adopt hlh's full T storage on non-root ranks; root keeps its eight slots.
    const char* mpi_rank=getenv("OMPI_COMM_WORLD_RANK");
    if(!mpi_rank)mpi_rank=getenv("PMIX_RANK");
    tc_t_slots=mpi_rank && atoi(mpi_rank)!=0?13:8;
    tb_strassen_t_alloc=(size_t(tc_t_slots)*25*(1ul<<20)+
                        size_t(tc_t_slots-8)*10*(1ul<<20)+(2ul<<20)-1)&~((2ul<<20)-1);
    tb_strassen_t_workspace=(double*)huge_buffer(tb_strassen_t_alloc);
    tb_strassen_products_workspace=(double*)huge_buffer(tb_strassen_products_bytes);

#pragma omp parallel num_threads(38)

    {
        auto &w = scratch[omp_get_thread_num()];
        w.a = (double *)(scratch_arena + size_t(omp_get_thread_num()) * scratch_slot_bytes);
        w.b = w.a + BM * BK;
        w.c = w.b + BN * scratch_b_k;
        memset(w.a, 0, BM * BK * sizeof(double));
        memset(w.b, 0, BN * scratch_b_k * sizeof(double));
        memset(w.c, 0, scratch_c_rows * BN * sizeof(double));
#pragma omp for schedule(static)
        for (size_t i = 0; i < tail_k_capacity / sizeof(double); ++i)
            tail_k_workspace[i] = 0;
#pragma omp for schedule(static)
        for (size_t i = 0; i < packed_a_capacity / sizeof(double); ++i)
            packed_a_workspace[i] = 0;
#pragma omp for schedule(static)
        for(size_t i=0;i<tb_strassen_t_alloc/8;++i)tb_strassen_t_workspace[i]=0;
#pragma omp for schedule(static)
        for(size_t i=0;i<tb_strassen_products_bytes/8;++i)tb_strassen_products_workspace[i]=0;
    }
    FILE *maps = fopen("/proc/self/smaps", "r");
    char line[256];
    bool selected = false;
    long huge_kb = -1;
    while (maps && fgets(line, sizeof(line), maps)) {
        unsigned long start, end;
        if (sscanf(line, "%lx-%lx", &start, &end) == 2)
            selected = (uintptr_t)scratch_arena >= start && (uintptr_t)scratch_arena < end;
        if (selected && sscanf(line, "AnonHugePages: %ld kB", &huge_kb) == 1)
            break;
    }
    if (maps)
        fclose(maps);

    if (huge_kb < long(scratch_arena_bytes / 1024))
        std::abort();
}

struct Matrix {
    const double *ptr;
    long row, col;
    bool packed = false;
    bool k320 = false;
};
struct Output {
    double *ptr;
    long row, col;
    int packed_r = 0;
    int logical_rows = 0;
    bool k320 = false;
    long k_offset = 0;
    // rows >= spill_row go to spill (same row/col/batch strides) instead of ptr; after spill_tiles split-tail
    // tiles are stored, *spill_flag is raised for SDMA readers
    double *spill = nullptr;
    int spill_row = 1 << 30;
    unsigned *spill_count = nullptr;
    unsigned long *spill_flag = nullptr;
    unsigned spill_tiles = 0;
    unsigned long spill_value = 1;
};

__attribute__((always_inline)) inline void store_r_group(Output c, const double *group_c, int mc, int nc, int group_i0,
                                                         int j0, int batch, long c_batch) {
    if (c.col == 2) {
        if (mc % 32)
            std::abort();
        const int rows = mc / 2;
        for (int p = 0; p < rows; ++p) {
            const int output_row = group_i0 / 2 + p;
            if (output_row >= c.logical_rows)
                continue;
            double *destination = (output_row < c.spill_row ? c.ptr + long(output_row) * c.row
                                                            : c.spill + long(output_row - c.spill_row) * c.row) +
                                  batch * c_batch + long(j0) * 2;
            for (int j = 0; j < nc; ++j) {
                const long offset = (p / 16) * 16 * BN + (j / 32) * 16 * 32 + (j % 32) * 16 + p % 16;
                const double *values = group_c;
                const double first = values[offset], second = values[offset + rows * BN];
                destination[j * 2] = first;
                destination[j * 2 + 1] = second;
            }
        }
    } else {
        if (mc != 256 || c.col != 4)
            std::abort();
        for (int p = 0; p < 64; ++p) {
            const int output_row = group_i0 / 256 * 64 + p;
            if (output_row >= c.logical_rows)
                continue;
            double *destination = (output_row < c.spill_row ? c.ptr + long(output_row) * c.row
                                                            : c.spill + long(output_row - c.spill_row) * c.row) +
                                  batch * c_batch + long(j0) * 4;
            for (int j = 0; j < nc; ++j) {
                const long offset = (p / 16) * 16 * BN + (j / 32) * 16 * 32 + (j % 32) * 16 + p % 16;
                for (int r = 0; r < 4; ++r) {
                    const double value = group_c[offset + r * 64 * BN];
                    destination[j * 4 + r] = value;
                }
            }
        }
    }
}

struct ExternalTbTeam {
    const unsigned task_count;
    struct alignas(64) Cursor {
        std::atomic<unsigned> next{0}, finished{0};
        unsigned count = 0;
    };
    Cursor cursors[38];
    std::atomic<unsigned> ready{0};
    void *context = nullptr;
    void (*function)(void *, int) = nullptr;
    // optional callback after each item a worker runs (owner of the callback decides what to do with it)
    void *between_context = nullptr;
    void (*between)(void *, int) = nullptr;

    // block6: each tail worker also owns the block-6 K pieces of its groups, items 38*8+group after its tail pieces
    const bool block6;
    // the items of workers 32..37 go to one pool, claimed in order: their tail pieces first, then block-6 pieces
    std::array<int, 96> pool{};
    unsigned pool_size = 0;
    alignas(64) std::atomic<unsigned> pool_next{0};
    ExternalTbTeam(bool spatial = false, bool split6 = false)
        : task_count(spatial  ? 100
                     : split6 ? 128
                              : 96),
          block6(split6) {
        if (spatial && split6)
            std::abort();
        unsigned total = 0;
        for (int i = 0; i < 38; ++i) {
            cursors[i].count = i < 32 ? 2 : (spatial ? 6 : unsigned((31 - (i - 32)) / 6 + 1) * (split6 ? 2 : 1));
            total += cursors[i].count;
        }
        if (total != task_count)
            std::abort();
        for (int pass = 0; pass < 2; ++pass)
            for (unsigned n = 0; n < 12; ++n)
                for (int i = 32; i < 38; ++i) {
                    const unsigned pieces = block6 ? cursors[i].count / 2 : cursors[i].count;
                    if (pass == 0 && n < pieces)
                        pool[pool_size++] = i * 8 + int(n);
                    if (pass == 1 && n < cursors[i].count - pieces)
                        pool[pool_size++] = 38 * 8 + (i - 32) + 6 * int(n);
                }
        for (int i = 32; i < 38; ++i)
            cursors[i].count = 0;
    }
    bool claim_pool(int &item) {
        unsigned n = pool_next.load(std::memory_order_relaxed);
        while (n < pool_size) {

            if (pool_next.compare_exchange_weak(n, n + 1, std::memory_order_relaxed)) {
                item = pool[n];
                return true;
            }
        }
        return false;
    }
    bool claim(int owner, int &item) {
        auto &c = cursors[owner];
        unsigned n = c.next.load(std::memory_order_relaxed);
        while (n < c.count)
            if (c.next.compare_exchange_weak(n, n + 1, std::memory_order_relaxed)) {
                const unsigned pieces = block6 && owner >= 32 ? c.count / 2 : c.count;
                item = n >= pieces ? 38 * 8 + (owner - 32) + 6 * int(n - pieces)
                                   : owner * 8 + (owner < 32 ? (n == 0 ? 7 : 0) : n);
                return true;
            }
        return false;
    }
    bool done() const {
        unsigned sum = 0;
        for (const auto &c : cursors)
            sum += c.finished.load(std::memory_order_acquire);
        return sum == task_count;
    }
    void work(int worker) {
        unsigned completed = 0;
        while (!ready.load(std::memory_order_acquire))
            asm volatile("yield" ::: "memory");
        auto run = [&](int owner) {
            int item;
            if (!(owner < 0 ? claim_pool(item) : claim(owner, item)))
                return false;
            function(context, item);
            ++completed;
            if (between)
                between(between_context, worker);
            return true;
        };
        while (run(worker)) {
        }
        while (run(-1)) {
        }
        bool progress;
        do {
            progress = false;
            for (int d = 1; d < 38; ++d)
                if (run((worker + d) % 38))
                    progress = true;
        } while (progress);
        cursors[worker].finished.store(completed, std::memory_order_release);
    }
    template <class F> void run(F &callable) {
        if (omp_get_thread_num() != 0 || ready.load(std::memory_order_relaxed))
            std::abort();
        context = &callable;
        function = [](void *p, int item) { (*static_cast<F *>(p))(item); };
        ready.store(1, std::memory_order_release);

        work(0);
        while (!done())
            asm volatile("yield" ::: "memory");
    }
};
static ExternalTbTeam *external_tb_team = nullptr;

template <bool output_fusion, bool spatial_dispatch = true>
__attribute__((noinline)) static void gemm_r_impl(int m, int n, int k, Matrix a, Matrix b, Output c, int batches,
                                                  long b_batch, long c_batch) {
    constexpr int BK = 320;
    static_assert(0 + BK * BN * sizeof(double) <= 384 * BN * sizeof(double), "shift fits reserved B scratch");

    if (!a.packed || c.packed_r || c.k320 || !c.logical_rows)
        std::abort();
    const int padded_m = (m + MR - 1) / MR * MR;
    const double *packed_a = a.ptr;

    if (!b.packed)
        std::abort();
    const double *packed_b = b.ptr;
    const long packed_b_stride = b_batch;
    constexpr int gemm_threads = 38;
    const int dispatch_base_rows = (m + BM - 1) / BM;
    const bool split_last_k = n == 500 && batches == 8 && (m == 2048 || m == 2016) && dispatch_base_rows == 8;

    unsigned tail_k_ready[32] = {};
    const bool tc1_shape = k == 16000; // TC1 TB (k=16000) vs TC2 TB (k=8000, m=2016 or 2048)
    const int k6 = split_last_k && tc1_shape && !spatial_dispatch ? tc1_tb_k6 : k;
    unsigned block6_ready[32] = {};
    // the later of an owner (K<k6) and a tail worker (K>=k6) adds the other's block-6 partial sums and stores them
    auto finish_block6 = [&](int group, int piece, double *mine, int nc, int j0, int batch) {
        memcpy(block6_workspace + (size_t(group) * 2 + piece) * BM * BN, mine, size_t(BM) * BN * sizeof(double));
        if (__atomic_add_fetch(&block6_ready[group], 1u, 4) != 2)
            return;
        const double *other = block6_workspace + (size_t(group) * 2 + 1 - piece) * BM * BN;
        for (int e = 0; e < BM * BN; ++e)
            mine[e] += other[e];
        store_r_group(c, mine, BM, nc, 6 * BM, j0, batch, c_batch);
    };
    static_assert(BM == 256 && BN == 128, "tail K workspace layout");
    static_assert(BK == 320, "weighted owner partition follows measured K320 costs");
    if (split_last_k && gemm_threads != 38)
        std::abort();
    const int owner_work_items = split_last_k ? 38 * 8 : batches * ((n + BN - 1) / BN) * dispatch_base_rows;
    auto run_item = [&](int item, int slab = -1, bool reuse_b = false, int selected_group = -1, int row_lo = 0,
                        int row_hi = -1) {
        int block6_group = -1;
        if (k6 < k && item >= owner_work_items) {
            block6_group = item - owner_work_items;
            item = 32 * 8;
        }
        const bool spatial = item >= owner_work_items;
        if (spatial)
            item -= owner_work_items;
        // logical rows of block 6 the owners leave to the tail workers; the 2048-row TC2 TB moves half, so the tail
        // workers keep the same 9 bundles as with 2016 rows
        const int moved_logical = tc1_shape || m == 2048 ? 16 : 32;

        const int worker = item / 8, lane = item % 8;

        const bool pair_ordinary = split_last_k && worker < 32 && lane < 7;
        if (pair_ordinary && lane) {

            return;
        }
        int group = split_last_k ? (worker < 32 ? worker : worker - 32 + lane * (38 - 32)) : item / dispatch_base_rows;

        if (selected_group >= 0)
            group = selected_group;
        if (block6_group >= 0)
            group = block6_group;

        if (split_last_k && group >= 32) {

            return;
        }

        const int slot = spatial || block6_group >= 0
                             ? 6
                             : (split_last_k ? (worker < 32 ? lane : dispatch_base_rows) : item % dispatch_base_rows);
        const int batch = group / ((n + BN - 1) / BN), jtile = group % ((n + BN - 1) / BN) * BN;
        const int i0 = std::min(slot, dispatch_base_rows - 1) * BM, j0 = jtile;
        const bool split_tail = split_last_k && slot >= dispatch_base_rows - 1;
        const int tail_piece = slot == dispatch_base_rows;
        const bool fused_output = output_fusion && c.col == 4 && !split_tail && block6_group < 0;

        auto w = scratch[omp_get_thread_num()];
        if (spatial)
            w.c += 256 * BN;

        int mc = std::min(BM, m - i0), nc = std::min(BN, n - j0);
        if (pair_ordinary)
            mc = owner_ordinary_rows;
        int mp = (mc + MR - 1) / MR * MR, np = (nc + NR - 1) / NR * NR;
        static_assert(BM == 256 && BN == 128 && MR == 16 && NR == 32, "tail8 layout");
        int tail_group = mp, tail_start = i0, tail_limit = m;
        {
            if ((c.col != 2 && c.col != 4) || mc % c.col)
                std::abort();
            tail_group = mc / c.col;
            tail_start = i0 / c.col;
            tail_limit = c.logical_rows;
        }
        unsigned tail8_mask = 0, dead_mask = 0;
        if (c.col == 4)
            for (int offset = 0; offset < mp; offset += MR) {
                const int live = tail_limit - tail_start - offset % tail_group;
                if (live > 0 && live <= 8)
                    tail8_mask |= 1u << (offset / MR);
                if (live <= 0) {
                    if (offset >= 32 * MR || !split_tail)
                        std::abort();
                    dead_mask |= 1u << (offset / MR);
                }
            }

        const int k_middle = tc1_shape ? 320 : tc2_tb_seed_k;
        const int k_begin = block6_group >= 0 ? k6 : split_tail && tail_piece ? k_middle : 0;
        const int k_end = split_tail && !tail_piece ? k_middle : k;

        const int chunk_rows = pair_ordinary ? 1792 : mp;
        for (int chunk_begin = 0; chunk_begin < mp; chunk_begin += chunk_rows) {
            const int chunk_end = std::min(mp, chunk_begin + chunk_rows);
            for (int k0 = slab < 0 ? k_begin : slab; k0 < (slab < 0 ? k_end : slab + BK); k0 += BK) {

                int kc = std::min(BK, k_end - k0);

                double *current_b = w.b + 0 / sizeof(double);
                constexpr bool helper_dma = false;

                for (int istep = chunk_begin; istep < chunk_end; istep += MR) {
                    auto panel_position = [&](int step) {
                        {
                            if (fused_output && k0 + kc == k_end) {
                                if (pair_ordinary) {
                                    const int rows = 256 / c.col, local = step % 256;
                                    return long(step / 256 * 256 + (local / (MR * c.col)) * MR +
                                                ((local / MR) % c.col) * rows);
                                }
                                const int rows = mc / c.col;
                                return (step / (MR * c.col)) * MR + ((step / MR) % c.col) * rows;
                            }
                        }
                        return long(step);
                    };
                    const int ib = panel_position(istep);
                    bool first_b_panel = ib == chunk_begin;
                    if (dead_mask && ib < 32 * MR && (dead_mask >> (ib / MR) & 1)) {
                        if (first_b_panel)
                            std::abort();
                        continue;
                    }
                    if (pair_ordinary && k0 >= k6 && ib >= 6 * BM)
                        continue;

                    if (spatial_dispatch && pair_ordinary && ib >= 1536 &&
                        (ib % 256) % (256 / c.col) >= 256 / c.col - moved_logical)
                        continue;
                    if (spatial && ib % (256 / c.col) < 256 / c.col - moved_logical)
                        continue;
                    if (spatial)
                        first_b_panel = ib == 256 / c.col - moved_logical;

                    if (split_tail && worker >= 32 && row_hi >= 0) {
                        if (ib % tail_group < row_lo || ib % tail_group >= row_hi)
                            continue;
                        first_b_panel = ib == row_lo;
                    }
                    if (reuse_b)
                        first_b_panel = false;

                    auto a_pointer = [&](long row, long reduction) {
                        if (a.k320) {
                            if (!((k == 16000 && padded_m == 2048) ||
                                  (k == 8000 && (padded_m == 2016 || padded_m == 2048))) ||
                                BK != 320)
                                std::abort();
                            const long rows = std::min(256L, long(padded_m) - row / 256 * 256);
                            return packed_a + (row / 256 * 256) * k + (reduction / 320 * 320) * rows +
                                   (row % 256) * 320 + (reduction % 320) * MR;
                        }
                        return packed_a + row * k + reduction * MR;
                    };
                    const double *ap = a_pointer(i0 + ib, k0);
                    const double *future_a = nullptr;
                    if ((np == BN) && kc % 4 == 0) {
                        if (istep + MR < chunk_end)
                            future_a = a_pointer(i0 + panel_position(istep + MR), k0);
                        else if (k0 + 2 * kc <= k_end)
                            future_a = a_pointer(i0 + chunk_begin, k0 + kc);
                    }
                    SmeNPanel d{np / NR,
                                kc,
                                ap,
                                current_b,
                                w.c + ib * BN,
                                current_b,
                                future_a,
                                long(kc) * NR * 8,
                                long(kc) * NR * 8,
                                long(kc) * MR * 8 / 4,
                                k0 != k_begin};

                    {
                        if (fused_output && k0 + kc == k_end) {
                            static_assert(MR == 16 && NR == 32 && BN == 128, "direct output tile shape");
                            if ((c.col != 2 && c.col != 4) || mc % c.col)
                                std::abort();
                            const int subgroup = pair_ordinary ? ib / 256 * 256 : 0;
                            const int rows = (pair_ordinary ? 256 : mc) / c.col;
                            const int local_ib = ib - subgroup;
                            const int p = (i0 + subgroup) / c.col + local_ib % rows, plane = local_ib / rows;
                            if (rows % MR)
                                std::abort();
                            const long valid = std::max(0, std::min(MR, c.logical_rows - p));
                            if (plane == c.col - 1) {
                                double *output =
                                    valid ? c.ptr + batch * c_batch + long(p) * c.row + long(j0) * c.col : c.ptr;
                                SmeJoinedPanel jd{{d, output, c.row * 8, c.col * 8, valid, nc},
                                                  w.c + (subgroup + local_ib % rows) * BN,
                                                  long(rows) * BN * 8};
                                if (future_a) {
                                    sme_joined_r4_prefetch_a(jd);
                                } else {
                                    sme_joined_r4_pipeline(jd);
                                }
                                continue;
                            }
                        }
                    }

                    if (k0 != k_begin) {
                        if (first_b_panel && !helper_dma) {
                            d.b = packed_b + batch * packed_b_stride + long(j0) * k + long(k0) * NR;
                            d.b_stride = (long(k) * NR * 8);
                        }
                        const double *next_c =
                            istep + MR < chunk_end ? w.c + panel_position(istep + MR) * BN : w.c + chunk_begin * BN;
                        SmeCPanel cd{d, next_c};
                        {
                            if (pair_ordinary && (chunk_rows == 1792 || 0) && mp == 1792 && istep == chunk_begin + MR &&
                                ib == chunk_begin + MR && np == 128 && kc == 320 && k0 + kc < k_end && a.k320 &&
                                gemm_threads == 38 && tail8_mask == 0) {

                                const int run_end = spatial_dispatch || k0 >= k6 ? 1536 : chunk_end;
                                const double *rows[112];
                                for (int row = 0; row < (run_end - chunk_begin) / MR; ++row)
                                    rows[row] = a_pointer(i0 + chunk_begin + row * MR, k0);
                                SmeMRun md{cd, rows + 1, (run_end - chunk_begin) / MR - 2};

                                sme_mrun_accumulate(md);

                                istep = run_end - 2 * MR;
                                continue;
                            }
                        }

                        if ((split_tail || block6_group >= 0) && worker >= 32 && istep > chunk_begin && ib == istep &&

                            row_hi < 0 &&

                            kc == 320 && np == 128 && k0 + kc < k_end && a.k320) {

                            int stop = istep;
                            while (stop < chunk_end - MR && !((tail8_mask | dead_mask) & (1u << (stop / MR))))
                                stop += MR;
                            const int count = (stop - istep) / MR;
                            if (count >= 2) {
                                const double *rows[17];
                                if (count > 16)
                                    std::abort();
                                for (int row = 0; row <= count; ++row)
                                    rows[row] = a_pointer(i0 + istep + row * MR, k0);
                                SmeMRun md{cd, rows, count};
                                sme_mrun_accumulate_reset(md);
                                istep += (count - 1) * MR;
                                continue;
                            }
                        }

                        if (ib && k0 + kc < k_end && (!pair_ordinary || ib < 256) && ib < 32 * MR &&
                            (tail8_mask & (1u << (ib / MR)))) {

                            if (kc == 320 && np == 128 && (2 != 3 || future_a)) {

                                sme_m8_n64_interleaved(cd);

                            } else

                                if (future_a)
                                sme_npanel_prefetch_a_c64_spread32_m8(cd);
                            else
                                sme_npanel_pipeline_c64_spread32_m8(cd);
                            continue;
                        }
                        if (first_b_panel && !helper_dma) {
                            sme_npanel_pack_b_pipeline_c64_spread32(cd);
                        } else if (future_a)
                            sme_npanel_prefetch_a_c64_spread32(cd);
                        else
                            sme_npanel_pipeline_c64_spread32(cd);
                        continue;
                    }
                    if (first_b_panel && !helper_dma) {
                        d.b = packed_b + batch * packed_b_stride + long(j0) * k + long(k0) * NR;
                        d.b_stride = (long(k) * NR * 8);
                        sme_npanel_pack_b_pipeline(d);
                    } else if (future_a) {
                        sme_npanel_prefetch_a(d);
                    } else
                        sme_npanel_pipeline(d);
                }
                if (pair_ordinary && k0 + kc == k6)
                    finish_block6(group, 0, w.c + 6 * BM * BN, nc, j0, batch);
            }
        }

        if (slab >= 0 && slab + BK < k_end)
            return;
        if (block6_group >= 0) {
            finish_block6(group, 1, w.c, nc, j0, batch);
            return;
        }
        if (split_tail) {
            const size_t tile = size_t(batch) * 4 + j0 / BN;
            double *own = tail_k_workspace + (tile * 2 + tail_piece) * BM * BN;

            const unsigned expected = spatial_dispatch && group >= 30 ? (tc1_shape ? 3 : 4) : 2;
            if (tail_piece && row_hi >= 0) {
                for (int plane = 0; plane < c.col; ++plane)
                    for (int row = row_lo; row < row_hi; row += MR)
                        memcpy(own + (plane * tail_group + row) * BN, w.c + (plane * tail_group + row) * BN,
                               MR * BN * sizeof(double));
            } else

                memcpy(own, w.c, size_t(mc) * BN * sizeof(double));
            const unsigned completed = __atomic_add_fetch(&tail_k_ready[tile], 1u, 4);

            if (completed < expected) {

                return;
            }
            if (completed != expected)
                std::abort();

            if (tail_piece && row_hi >= 0)
                memcpy(w.c, own, size_t(mc) * BN * sizeof(double));

            const double *other = tail_k_workspace + (tile * 2 + 1 - tail_piece) * BM * BN;
            for (int element = 0; element < mc * BN; ++element)
                w.c[element] += other[element];
        }
        {

            if (!fused_output) {

                for (int subgroup = 0; subgroup < (pair_ordinary ? owner_ordinary_rows : 256); subgroup += 256) {
                    const int mc = pair_ordinary ? 256 : std::min(BM, m - i0);
                    const int group_i0 = i0 + subgroup;
                    const double *group_c = w.c + subgroup * BN;

                    if (c.col == 2 && (spatial || (pair_ordinary && subgroup == 1536))) {
                        const int begin = spatial ? 128 - moved_logical : 0;
                        const int end = spatial ? 128 : 128 - moved_logical;
                        for (int p = begin; p < end; ++p) {
                            double *dst = c.ptr + batch * c_batch + long(group_i0 / 2 + p) * c.row + long(j0) * 2;
                            for (int j = 0; j < nc; ++j) {
                                const long offset = (p / 16) * 16 * BN + (j / 32) * 16 * 32 + (j % 32) * 16 + p % 16;
                                dst[j * 2] = group_c[offset];
                                dst[j * 2 + 1] = group_c[offset + 128 * BN];
                            }
                        }
                        continue;
                    }

                    store_r_group(c, group_c, mc, nc, group_i0, j0, batch, c_batch);
                }
                if (split_tail && c.spill_flag &&
                    __atomic_add_fetch(c.spill_count, 1u, __ATOMIC_ACQ_REL) == c.spill_tiles) {
                    asm volatile("dmb oshst" ::: "memory");
                    __atomic_store_n(c.spill_flag, c.spill_value, __ATOMIC_RELEASE);
                }
            }
        }
    };
    if (!split_last_k)
        std::abort();

    auto execute_item = [&](int item) {
        if constexpr (!spatial_dispatch) {
            run_item(item);

            return;
        }
        const int worker = item / 8, lane = item % 8;
        int group = worker < 32 ? worker : worker - 32 + lane * 6;

        if (worker >= 32 && lane >= 6) {

            return;
        }
        const bool sliced = worker >= 32 && lane == 5;
        const int moved_logical = tc1_shape || m == 2048 ? 16 : 32;
        const int grain = 16 * c.col, spatial_bundles = moved_logical * c.col / grain,
                  total_bundles = (m - 1792 + moved_logical * c.col) / grain;
        const int part = (worker - 32) % 3;
        const int lo = sliced ? total_bundles * part / 3 : 0,
                  hi = sliced ? total_bundles * (part + 1) / 3 : total_bundles;
        if (sliced)
            group = 30 + (worker - 32) / 3;
        const bool has_spatial = lo<spatial_bundles, has_tail = hi> spatial_bundles;
        const int tail_lo = std::max(0, lo - spatial_bundles) * 16, tail_hi = std::max(0, hi - spatial_bundles) * 16;

        if (group >= 32 || (worker < 32 && lane != 0 && lane != 7)) {

            return;
        }
        const int middle = tc1_shape ? 320 : tc2_tb_seed_k;
        if (worker < 32)
            run_item(item);

        else
            for (int k0 = has_spatial ? 0 : middle; k0 < k; k0 += BK) {

                if (has_spatial)
                    run_item(owner_work_items + item, k0, false, group);
                if (has_tail && k0 >= middle)
                    run_item(item, k0, has_spatial, group, tail_lo, sliced ? tail_hi : -1);
            }
    };

    if (!external_tb_team)
        std::abort();

    if (n != 500 || batches != 8 || !((m == 2048 && k == 16000) || ((m == 2016 || m == 2048) && k == 8000)))
        std::abort();
    if (external_tb_team->task_count != (spatial_dispatch ? 100u : k6 < k ? 128u : 96u))
        std::abort();
    external_tb_team->run(execute_item);

    for (int group = 0; group < 32; ++group)
        if (tail_k_ready[group] != (spatial_dispatch && group >= 30 ? (tc1_shape ? 3u : 4u) : 2u) ||
            (k6 < k && block6_ready[group] != 2))
            std::abort();
}

static void gemm(int m, int n, int k, Matrix a, Matrix b, Output c, int batches, long b_batch, long c_batch) {
    const bool shape =
        omp_get_max_threads() == 38 && n == 500 && batches == 8 && c.logical_rows &&
        ((m == 2048 && k == 16000 && c.col == 4) || ((m == 2016 || m == 2048) && k == 8000 && c.col == 2));
    if (!shape)
        std::abort();
    if (c.col == 4)
        gemm_r_impl<true, false>(m, n, k, a, b, c, batches, b_batch, c_batch);
    else
        gemm_r_impl<false, true>(m, n, k, a, b, c, batches, b_batch, c_batch);
}
} // namespace tcopt

namespace tcopt {

// Generated by generate-ac-tb-strassen-store.py. Included inside tcopt.
// The original twelve K320 updates and M2048 reuse remain unchanged.
static void tc1_ac_tb_strassen_last(const double* a,const double* wire,
    double* partial,double* extra,double* packed,double* aliases,double* derived,
    int depth,int local_u){
 constexpr int kc=160,k0=3840,width=128,tile=16*128;
 static_assert(BK*BN>=kc*width+2*tile,"reuse unused final-K private panel capacity");
 if((depth!=160 && depth!=320) || local_u<0 || local_u%32 || local_u+32>depth)std::abort();
 double* temporary=packed+kc*width;
 auto cp=[&](int row){return row<1792?partial+size_t(row)*width:extra+size_t(row-1792)*width;};
 auto ap=[&](int row){return a+size_t(row/512)*512*4000+size_t(k0)*512+size_t(row%512)*kc;};
 auto row_at=[&](int group,int position){
  return (group/32+position*2)*512+group%32*16;
 };
 for(int group=0;group<64;++group){
  const int hs=group/32,p=group%32*16;
  for(int position=0;position<2;++position){
   const int row=row_at(group,position);
   const double* future=position==0?ap(row_at(group,1)):
                         group<63?ap(row_at(group+1,0)):nullptr;
   const bool first=group==0 && position==0;
   SmeNPanel n{4,kc,ap(row),first?wire+size_t(k0)*32:packed,cp(row),packed,future,
               (first?4000L:long(kc))*32*8,long(kc)*32*8,kc*16*8/4,1};
   SmeTPanel t{n,temporary+position*tile,16L*32*8};
   if(first)sme_tstore_pack_b_pipeline(t);
   else if(future)sme_tstore_prefetch_a(t);
   else sme_tstore_pipeline(t);
  }
  for(int r=0;r<2;++r){
   const size_t offset=size_t(r*512+p)*depth+size_t(local_u)*16;
   const double *x00=temporary+r*512,*x01=temporary+tile+r*512,
                *x10=temporary+(r+2)*512,*x11=temporary+tile+(r+2)*512;
   double* alias0=aliases+size_t(hs)*2048*depth+offset;
   double* alias3=aliases+size_t((hs+2)*2048+1024)*depth+offset;
   double* d0=derived+size_t(hs)*1024*320+offset;
   double* d1=derived+size_t(2+hs)*1024*320+offset;
   double* d2=derived+size_t(4+hs)*1024*320+offset;
   double* d3=derived+size_t(6+hs)*1024*320+offset;
   double* d4=derived+size_t(8+hs)*1024*320+offset;
#pragma omp simd
   for(int i=0;i<512;++i){
    const double a00=x00[i],a01=x01[i],a10=x10[i],a11=x11[i];
    alias0[i]=a00;alias3[i]=a11;
    d0[i]=a00+a11;d1[i]=a10+a11;d2[i]=a00+a01;
    d3[i]=a10-a00;d4[i]=a01-a11;
   }
  }
 }
}

template<bool task_local=false,bool direct_c=false,bool cross_s=false,bool large_ac=false,bool consumer_layout=false>
static void tc1_segmented_leaf_tb_strassen(const double* a,const double* b,
                                     double* ac,double* packed,double* output,
                                     int m,int row_start,int total_k,int local_k,double* extra,double* derived){
 constexpr int width=128,k=4000;
 static_assert(!cross_s || task_local,"cross-S AC requires task-local T layout");
 static_assert(!large_ac || (task_local && cross_s && !direct_c),"large AC uses packed cross-S panels");
 static_assert(!consumer_layout || large_ac,"consumer T layout uses large AC");
 if(m<=0 || m%16 || row_start<0 || row_start%16 ||
    (!cross_s && row_start%512+m>512) || (cross_s && row_start+m>2048) ||
    local_k<0 || local_k%32 || local_k+32>total_k)std::abort();
 if(task_local && ((!consumer_layout && (total_k!=32 || local_k!=0)) ||
                   m>(large_ac?2048:(1?544:272))))std::abort();
 if(large_ac && m>1792 && !extra)std::abort();
 auto cp=[&](int row){
   if constexpr(large_ac)if(row>=1792)return extra+size_t(row-1792)*width;
   return ac+size_t(row)*width;
 };
 auto ap=[&](int row,int l){
   const int absolute=row_start+row;
   const int kc=std::min(320,k-l);
   return a+size_t(absolute/512)*512*k+size_t(l)*512+size_t(absolute%512)*kc;
 };
 auto tptr=[&](int row){
   if constexpr(consumer_layout){
     const int absolute=row_start+row,si=absolute/512,p=absolute%512;
     constexpr int pblock=1024/4;
     return output+(size_t(si)*2048+size_t(p/pblock)*1024+(p%pblock/16)*16)*total_k+size_t(local_k)*16;
   }
   // Task-local output is [M16 task][R4][Kr32][16].  It remains in the
   // producer's bounded staging area until every TB consumer has finished.
   if constexpr(task_local)return output+size_t(row)*4*total_k;
   const int p=(row_start+row)%512;
   const long panel=long(p/64)*256+p%64;
   return output+panel*total_k+long(local_k)*16;
 };
 for(int l=0;l<k;l+=320){
  const int kc=std::min(320,k-l);
  static_assert(task_local && !direct_c && cross_s && large_ac && consumer_layout,
                "fused TB stores use the complete M2048 producer");
  if(m!=2048 || row_start!=0 || !derived)std::abort();
  if(l+kc==k){
   tc1_ac_tb_strassen_last(a,b,ac,extra,packed,output,derived,total_k,local_k);
   continue;
  }
  auto single=[&](int row){
   const double* future=row+16<m?ap(row+16,l):l+2*kc<=k?ap(0,l+kc):nullptr;
   SmeNPanel d{4,kc,ap(row,l),packed,cp(row),packed,future,
               kc*32*8,kc*32*8,kc*16*8/4,l!=0};
   if(!row || direct_c){d.b=b+size_t(l)*32;d.b_stride=k*32*8;}
   if(l+kc==k){
    SmeTPanel td{d,tptr(row),(consumer_layout?long(1024/4):task_local?16L:64L)*total_k*8};
    if(!row && !direct_c)sme_tstore_pack_b_pipeline(td);
    else if(future)sme_tstore_prefetch_a(td);
    else sme_tstore_pipeline(td);
   }else if(l){
    SmeCPanel cd{d,row+16<m?cp(row+16):cp(0)};
    if(!row && !direct_c)sme_npanel_pack_b_pipeline_c64_spread32(cd);
    else if(future)sme_npanel_prefetch_a_c64_spread32(cd);
    else sme_npanel_pipeline_c64_spread32(cd);
   }else{
    if(!row && !direct_c)sme_npanel_pack_b_pipeline(d);
    else if(future)sme_npanel_prefetch_a(d);
    else sme_npanel_pipeline(d);
   }
  };
  if(!l || l+kc==k || m<48){
   for(int row=0;row<m;row+=16)single(row);
   continue;
  }
  std::array<const double*,128> rows;
  if(m/16>int(rows.size()))std::abort();
  for(int i=0;i<m/16;++i)rows[i]=ap(i*16,l);
  single(0);
  auto run=[&](int first,int count){
    if(count<=0)return;
    SmeNPanel base{4,kc,rows[first/16],direct_c?b+size_t(l)*32:packed,cp(first),packed,rows[first/16+1],
                   (direct_c?k:kc)*32*8,kc*32*8,kc*16*8/4,1};
    SmeMRun descriptor{{base,cp(first+16)},rows.data()+first/16,count};
    sme_mrun_accumulate(descriptor);
  };
  if(large_ac && m>1792){
    run(16,110);single(1776);run(1792,(m-1792)/16-1);single(m-16);
  }else{run(16,m/16-2);single(m-16);}
 }
}

static void tc1_store_r4_tile(double* out,long out_row_stride,int p_local,
                            int p0,int q0,int ql,int batch,int nc,const double* c,int mr){
 if((nc!=32 && nc!=64 && nc!=128) || nc%32)std::abort();
 for(int p=0;p<mr;++p){
  const int row=p0+p;
  if(row>=p_local)continue;
  double* dst=out+long(row)*out_row_stride+long(batch*ql+q0)*4;
  for(int j=0;j<nc && q0+j<ql;++j){
   const long inner=long((p/16)*16*nc+(j/32)*16*32+(j%32)*16+p%16);
   for(int rr=0;rr<4;++rr)dst[j*4+rr]=c[long(rr)*mr*nc+inner];
  }
 }
}

// Included inside tcopt after tc_kernels.hpp. One Strassen level over
// TB M2048/N256/K16000. Each product retains M1024/N128 and K320/K160.
// The K halves are S={0,1} and S={2,3}; no new short reduction blocks.
// tc_kernels.hpp undefines its local clobber macro before this standalone
// harness includes the shared generated pack kernel. Restore its exact ABI.
#define TC_SME_VECTOR_CLOBBERS "z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15"

// Generated by generate-ac-sum-pack.py; included inside tcopt.
// Preserve z0-z23's K4 pipeline. z24-z27 form Cx +/- Cy before its packed
// store and first-row FMOPA. x7 is the second panel base, x6 its K cursor.
// Both C sources have the same N32 panel stride. All strides are bytes.
static_assert(0>=0 && 0%64==0 &&
              0+192<=32760,"PRFM immediate range");
struct SmeSumPanel { SmeNPanel n; const double* second_b; };
static_assert(sizeof(SmeSumPanel)==96 && offsetof(SmeSumPanel,second_b)==88,"sum descriptor");
template<bool subtract>
__attribute__((noinline,aligned(64))) static void sme_npanel_sum_pack_b_pipeline(const SmeSumPanel& d){
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "ldr x7, [x28, #88]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "mov x6, x7\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "mov x19, x25\n"
        "mov x20, #256\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z2.d, z2.d, z24.d\n"
        ".else\n"
        "fadd z2.d, z2.d, z24.d\n"
        ".endif\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z3.d, z3.d, z25.d\n"
        ".else\n"
        "fadd z3.d, z3.d, z25.d\n"
        ".endif\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z4.d, z4.d, z26.d\n"
        ".else\n"
        "fadd z4.d, z4.d, z26.d\n"
        ".endif\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z5.d, z5.d, z27.d\n"
        ".else\n"
        "fadd z5.d, z5.d, z27.d\n"
        ".endif\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z8.d, z8.d, z24.d\n"
        ".else\n"
        "fadd z8.d, z8.d, z24.d\n"
        ".endif\n"
        "st1d {z8.d}, p0, [x19]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z9.d, z9.d, z25.d\n"
        ".else\n"
        "fadd z9.d, z9.d, z25.d\n"
        ".endif\n"
        "st1d {z9.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z10.d, z10.d, z26.d\n"
        ".else\n"
        "fadd z10.d, z10.d, z26.d\n"
        ".endif\n"
        "st1d {z10.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z11.d, z11.d, z27.d\n"
        ".else\n"
        "fadd z11.d, z11.d, z27.d\n"
        ".endif\n"
        "st1d {z11.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z14.d, z14.d, z24.d\n"
        ".else\n"
        "fadd z14.d, z14.d, z24.d\n"
        ".endif\n"
        "st1d {z14.d}, p0, [x19]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z15.d, z15.d, z25.d\n"
        ".else\n"
        "fadd z15.d, z15.d, z25.d\n"
        ".endif\n"
        "st1d {z15.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z16.d, z16.d, z26.d\n"
        ".else\n"
        "fadd z16.d, z16.d, z26.d\n"
        ".endif\n"
        "st1d {z16.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z17.d, z17.d, z27.d\n"
        ".else\n"
        "fadd z17.d, z17.d, z27.d\n"
        ".endif\n"
        "st1d {z17.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z20.d, z20.d, z24.d\n"
        ".else\n"
        "fadd z20.d, z20.d, z24.d\n"
        ".endif\n"
        "st1d {z20.d}, p0, [x19]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z21.d, z21.d, z25.d\n"
        ".else\n"
        "fadd z21.d, z21.d, z25.d\n"
        ".endif\n"
        "st1d {z21.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z22.d, z22.d, z26.d\n"
        ".else\n"
        "fadd z22.d, z22.d, z26.d\n"
        ".endif\n"
        "st1d {z22.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z23.d, z23.d, z27.d\n"
        ".else\n"
        "fadd z23.d, z23.d, z27.d\n"
        ".endif\n"
        "st1d {z23.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z2.d, z2.d, z24.d\n"
        ".else\n"
        "fadd z2.d, z2.d, z24.d\n"
        ".endif\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z3.d, z3.d, z25.d\n"
        ".else\n"
        "fadd z3.d, z3.d, z25.d\n"
        ".endif\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z4.d, z4.d, z26.d\n"
        ".else\n"
        "fadd z4.d, z4.d, z26.d\n"
        ".endif\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z5.d, z5.d, z27.d\n"
        ".else\n"
        "fadd z5.d, z5.d, z27.d\n"
        ".endif\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z8.d, z8.d, z24.d\n"
        ".else\n"
        "fadd z8.d, z8.d, z24.d\n"
        ".endif\n"
        "st1d {z8.d}, p0, [x19]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z9.d, z9.d, z25.d\n"
        ".else\n"
        "fadd z9.d, z9.d, z25.d\n"
        ".endif\n"
        "st1d {z9.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z10.d, z10.d, z26.d\n"
        ".else\n"
        "fadd z10.d, z10.d, z26.d\n"
        ".endif\n"
        "st1d {z10.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z11.d, z11.d, z27.d\n"
        ".else\n"
        "fadd z11.d, z11.d, z27.d\n"
        ".endif\n"
        "st1d {z11.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z14.d, z14.d, z24.d\n"
        ".else\n"
        "fadd z14.d, z14.d, z24.d\n"
        ".endif\n"
        "st1d {z14.d}, p0, [x19]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z15.d, z15.d, z25.d\n"
        ".else\n"
        "fadd z15.d, z15.d, z25.d\n"
        ".endif\n"
        "st1d {z15.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z16.d, z16.d, z26.d\n"
        ".else\n"
        "fadd z16.d, z16.d, z26.d\n"
        ".endif\n"
        "st1d {z16.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z17.d, z17.d, z27.d\n"
        ".else\n"
        "fadd z17.d, z17.d, z27.d\n"
        ".endif\n"
        "st1d {z17.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z20.d, z20.d, z24.d\n"
        ".else\n"
        "fadd z20.d, z20.d, z24.d\n"
        ".endif\n"
        "st1d {z20.d}, p0, [x19]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z21.d, z21.d, z25.d\n"
        ".else\n"
        "fadd z21.d, z21.d, z25.d\n"
        ".endif\n"
        "st1d {z21.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z22.d, z22.d, z26.d\n"
        ".else\n"
        "fadd z22.d, z22.d, z26.d\n"
        ".endif\n"
        "st1d {z22.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z23.d, z23.d, z27.d\n"
        ".else\n"
        "fadd z23.d, z23.d, z27.d\n"
        ".endif\n"
        "st1d {z23.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 1b\n"
        "8:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z2.d, z2.d, z24.d\n"
        ".else\n"
        "fadd z2.d, z2.d, z24.d\n"
        ".endif\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z3.d, z3.d, z25.d\n"
        ".else\n"
        "fadd z3.d, z3.d, z25.d\n"
        ".endif\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z4.d, z4.d, z26.d\n"
        ".else\n"
        "fadd z4.d, z4.d, z26.d\n"
        ".endif\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z5.d, z5.d, z27.d\n"
        ".else\n"
        "fadd z5.d, z5.d, z27.d\n"
        ".endif\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "93:\n"
        "st1d {za0v.d[w12, 0]}, p0, [x13]\n"
        "st1d {za4v.d[w12, 0]}, p0, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "st1d {za1v.d[w12, 0]}, p0, [x14]\n"
        "st1d {za5v.d[w12, 0]}, p0, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "st1d {za2v.d[w12, 0]}, p0, [x15]\n"
        "st1d {za6v.d[w12, 0]}, p0, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "st1d {za3v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x7, x7, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d), [subtract]"i"(subtract), [pf0]"i"(0+0), [pf1]"i"(0+64), [pf2]"i"(0+128), [pf3]"i"(0+192) : "memory","cc","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28",TC_SME_VECTOR_CLOBBERS);
}

// First-M16 fused packing: RHS K16 requests to L2 and A K8 requests to L1.
// Both reuse the RHS remaining-K guard; demand arithmetic/order is unchanged.
static_assert(16==16,"validated TC1 source lookahead");
__attribute__((noinline, aligned(64))) static void tc1_rhs_pack_lookahead(const SmeNPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "mov x19, x25\n"
        "mov x20, #256\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "st1d {z8.d}, p0, [x19]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z9.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z10.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z11.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "st1d {z14.d}, p0, [x19]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z15.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z16.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z17.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "st1d {z20.d}, p0, [x19]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z21.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z22.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z23.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "1:\n"
        "cmp x11, #%c[required]\n"
        "b.lt 910f\n"
        "prfm pldl2keep, [x10, #%c[rhs0]]\n"
        "prfm pldl2keep, [x10, #%c[rhs1]]\n"
        "prfm pldl2keep, [x10, #%c[rhs2]]\n"
        "prfm pldl2keep, [x10, #%c[rhs3]]\n"
        "prfm pldl2keep, [x10, #%c[rhs4]]\n"
        "prfm pldl2keep, [x10, #%c[rhs5]]\n"
        "prfm pldl2keep, [x10, #%c[rhs6]]\n"
        "prfm pldl2keep, [x10, #%c[rhs7]]\n"
        "prfm pldl2keep, [x10, #%c[rhs8]]\n"
        "prfm pldl2keep, [x10, #%c[rhs9]]\n"
        "prfm pldl2keep, [x10, #%c[rhs10]]\n"
        "prfm pldl2keep, [x10, #%c[rhs11]]\n"
        "prfm pldl2keep, [x10, #%c[rhs12]]\n"
        "prfm pldl2keep, [x10, #%c[rhs13]]\n"
        "prfm pldl2keep, [x10, #%c[rhs14]]\n"
        "prfm pldl2keep, [x10, #%c[rhs15]]\n"
        "prfm pldl1keep, [x9, #1024]\n"
        "prfm pldl1keep, [x9, #1088]\n"
        "prfm pldl1keep, [x9, #1152]\n"
        "prfm pldl1keep, [x9, #1216]\n"
        "prfm pldl1keep, [x9, #1280]\n"
        "prfm pldl1keep, [x9, #1344]\n"
        "prfm pldl1keep, [x9, #1408]\n"
        "prfm pldl1keep, [x9, #1472]\n"
        "910:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "st1d {z8.d}, p0, [x19]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z9.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z10.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z11.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "st1d {z14.d}, p0, [x19]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z15.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z16.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z17.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "st1d {z20.d}, p0, [x19]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z21.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z22.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z23.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 1b\n"
        "8:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "93:\n"
        "st1d {za0v.d[w12, 0]}, p0, [x13]\n"
        "st1d {za4v.d[w12, 0]}, p0, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "st1d {za1v.d[w12, 0]}, p0, [x14]\n"
        "st1d {za5v.d[w12, 0]}, p0, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "st1d {za2v.d[w12, 0]}, p0, [x15]\n"
        "st1d {za6v.d[w12, 0]}, p0, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "st1d {za3v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d), [required]"i"(16+4), [rhs0]"i"(16*256+0), [rhs1]"i"(16*256+64), [rhs2]"i"(16*256+128), [rhs3]"i"(16*256+192), [rhs4]"i"(16*256+256), [rhs5]"i"(16*256+320), [rhs6]"i"(16*256+384), [rhs7]"i"(16*256+448), [rhs8]"i"(16*256+512), [rhs9]"i"(16*256+576), [rhs10]"i"(16*256+640), [rhs11]"i"(16*256+704), [rhs12]"i"(16*256+768), [rhs13]"i"(16*256+832), [rhs14]"i"(16*256+896), [rhs15]"i"(16*256+960) : "memory","cc","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}
// inplace: second_b is the packed panel itself (in L2), advanced by packed_stride instead of b_stride
template<bool subtract,bool inplace=false>
__attribute__((noinline,aligned(64))) static void tc1_rhs_sum_lookahead(const SmeSumPanel& d){
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "ldr x7, [x28, #88]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "mov x6, x7\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "mov x19, x25\n"
        "mov x20, #256\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z2.d, z2.d, z24.d\n"
        ".else\n"
        "fadd z2.d, z2.d, z24.d\n"
        ".endif\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z3.d, z3.d, z25.d\n"
        ".else\n"
        "fadd z3.d, z3.d, z25.d\n"
        ".endif\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z4.d, z4.d, z26.d\n"
        ".else\n"
        "fadd z4.d, z4.d, z26.d\n"
        ".endif\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z5.d, z5.d, z27.d\n"
        ".else\n"
        "fadd z5.d, z5.d, z27.d\n"
        ".endif\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z8.d, z8.d, z24.d\n"
        ".else\n"
        "fadd z8.d, z8.d, z24.d\n"
        ".endif\n"
        "st1d {z8.d}, p0, [x19]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z9.d, z9.d, z25.d\n"
        ".else\n"
        "fadd z9.d, z9.d, z25.d\n"
        ".endif\n"
        "st1d {z9.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z10.d, z10.d, z26.d\n"
        ".else\n"
        "fadd z10.d, z10.d, z26.d\n"
        ".endif\n"
        "st1d {z10.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z11.d, z11.d, z27.d\n"
        ".else\n"
        "fadd z11.d, z11.d, z27.d\n"
        ".endif\n"
        "st1d {z11.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z14.d, z14.d, z24.d\n"
        ".else\n"
        "fadd z14.d, z14.d, z24.d\n"
        ".endif\n"
        "st1d {z14.d}, p0, [x19]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z15.d, z15.d, z25.d\n"
        ".else\n"
        "fadd z15.d, z15.d, z25.d\n"
        ".endif\n"
        "st1d {z15.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z16.d, z16.d, z26.d\n"
        ".else\n"
        "fadd z16.d, z16.d, z26.d\n"
        ".endif\n"
        "st1d {z16.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z17.d, z17.d, z27.d\n"
        ".else\n"
        "fadd z17.d, z17.d, z27.d\n"
        ".endif\n"
        "st1d {z17.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z20.d, z20.d, z24.d\n"
        ".else\n"
        "fadd z20.d, z20.d, z24.d\n"
        ".endif\n"
        "st1d {z20.d}, p0, [x19]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z21.d, z21.d, z25.d\n"
        ".else\n"
        "fadd z21.d, z21.d, z25.d\n"
        ".endif\n"
        "st1d {z21.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z22.d, z22.d, z26.d\n"
        ".else\n"
        "fadd z22.d, z22.d, z26.d\n"
        ".endif\n"
        "st1d {z22.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z23.d, z23.d, z27.d\n"
        ".else\n"
        "fadd z23.d, z23.d, z27.d\n"
        ".endif\n"
        "st1d {z23.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "1:\n"
        "cmp x11, #%c[required]\n"
        "b.lt 910f\n"
        "prfm pldl2keep, [x10, #%c[rhs0]]\n"
        "prfm pldl2keep, [x10, #%c[rhs1]]\n"
        "prfm pldl2keep, [x10, #%c[rhs2]]\n"
        "prfm pldl2keep, [x10, #%c[rhs3]]\n"
        "prfm pldl2keep, [x10, #%c[rhs4]]\n"
        "prfm pldl2keep, [x10, #%c[rhs5]]\n"
        "prfm pldl2keep, [x10, #%c[rhs6]]\n"
        "prfm pldl2keep, [x10, #%c[rhs7]]\n"
        "prfm pldl2keep, [x10, #%c[rhs8]]\n"
        "prfm pldl2keep, [x10, #%c[rhs9]]\n"
        "prfm pldl2keep, [x10, #%c[rhs10]]\n"
        "prfm pldl2keep, [x10, #%c[rhs11]]\n"
        "prfm pldl2keep, [x10, #%c[rhs12]]\n"
        "prfm pldl2keep, [x10, #%c[rhs13]]\n"
        "prfm pldl2keep, [x10, #%c[rhs14]]\n"
        "prfm pldl2keep, [x10, #%c[rhs15]]\n"
        "prfm pldl2keep, [x6, #%c[rhs0]]\n"
        "prfm pldl2keep, [x6, #%c[rhs1]]\n"
        "prfm pldl2keep, [x6, #%c[rhs2]]\n"
        "prfm pldl2keep, [x6, #%c[rhs3]]\n"
        "prfm pldl2keep, [x6, #%c[rhs4]]\n"
        "prfm pldl2keep, [x6, #%c[rhs5]]\n"
        "prfm pldl2keep, [x6, #%c[rhs6]]\n"
        "prfm pldl2keep, [x6, #%c[rhs7]]\n"
        "prfm pldl2keep, [x6, #%c[rhs8]]\n"
        "prfm pldl2keep, [x6, #%c[rhs9]]\n"
        "prfm pldl2keep, [x6, #%c[rhs10]]\n"
        "prfm pldl2keep, [x6, #%c[rhs11]]\n"
        "prfm pldl2keep, [x6, #%c[rhs12]]\n"
        "prfm pldl2keep, [x6, #%c[rhs13]]\n"
        "prfm pldl2keep, [x6, #%c[rhs14]]\n"
        "prfm pldl2keep, [x6, #%c[rhs15]]\n"
        "prfm pldl1keep, [x9, #1024]\n"
        "prfm pldl1keep, [x9, #1088]\n"
        "prfm pldl1keep, [x9, #1152]\n"
        "prfm pldl1keep, [x9, #1216]\n"
        "prfm pldl1keep, [x9, #1280]\n"
        "prfm pldl1keep, [x9, #1344]\n"
        "prfm pldl1keep, [x9, #1408]\n"
        "prfm pldl1keep, [x9, #1472]\n"
        "910:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z2.d, z2.d, z24.d\n"
        ".else\n"
        "fadd z2.d, z2.d, z24.d\n"
        ".endif\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z3.d, z3.d, z25.d\n"
        ".else\n"
        "fadd z3.d, z3.d, z25.d\n"
        ".endif\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z4.d, z4.d, z26.d\n"
        ".else\n"
        "fadd z4.d, z4.d, z26.d\n"
        ".endif\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z5.d, z5.d, z27.d\n"
        ".else\n"
        "fadd z5.d, z5.d, z27.d\n"
        ".endif\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z8.d, z8.d, z24.d\n"
        ".else\n"
        "fadd z8.d, z8.d, z24.d\n"
        ".endif\n"
        "st1d {z8.d}, p0, [x19]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z9.d, z9.d, z25.d\n"
        ".else\n"
        "fadd z9.d, z9.d, z25.d\n"
        ".endif\n"
        "st1d {z9.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z10.d, z10.d, z26.d\n"
        ".else\n"
        "fadd z10.d, z10.d, z26.d\n"
        ".endif\n"
        "st1d {z10.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z11.d, z11.d, z27.d\n"
        ".else\n"
        "fadd z11.d, z11.d, z27.d\n"
        ".endif\n"
        "st1d {z11.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z14.d, z14.d, z24.d\n"
        ".else\n"
        "fadd z14.d, z14.d, z24.d\n"
        ".endif\n"
        "st1d {z14.d}, p0, [x19]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z15.d, z15.d, z25.d\n"
        ".else\n"
        "fadd z15.d, z15.d, z25.d\n"
        ".endif\n"
        "st1d {z15.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z16.d, z16.d, z26.d\n"
        ".else\n"
        "fadd z16.d, z16.d, z26.d\n"
        ".endif\n"
        "st1d {z16.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z17.d, z17.d, z27.d\n"
        ".else\n"
        "fadd z17.d, z17.d, z27.d\n"
        ".endif\n"
        "st1d {z17.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z20.d, z20.d, z24.d\n"
        ".else\n"
        "fadd z20.d, z20.d, z24.d\n"
        ".endif\n"
        "st1d {z20.d}, p0, [x19]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z21.d, z21.d, z25.d\n"
        ".else\n"
        "fadd z21.d, z21.d, z25.d\n"
        ".endif\n"
        "st1d {z21.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z22.d, z22.d, z26.d\n"
        ".else\n"
        "fadd z22.d, z22.d, z26.d\n"
        ".endif\n"
        "st1d {z22.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z23.d, z23.d, z27.d\n"
        ".else\n"
        "fadd z23.d, z23.d, z27.d\n"
        ".endif\n"
        "st1d {z23.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 1b\n"
        "8:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z2.d, z2.d, z24.d\n"
        ".else\n"
        "fadd z2.d, z2.d, z24.d\n"
        ".endif\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z3.d, z3.d, z25.d\n"
        ".else\n"
        "fadd z3.d, z3.d, z25.d\n"
        ".endif\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z4.d, z4.d, z26.d\n"
        ".else\n"
        "fadd z4.d, z4.d, z26.d\n"
        ".endif\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z5.d, z5.d, z27.d\n"
        ".else\n"
        "fadd z5.d, z5.d, z27.d\n"
        ".endif\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "93:\n"
        "st1d {za0v.d[w12, 0]}, p0, [x13]\n"
        "st1d {za4v.d[w12, 0]}, p0, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "st1d {za1v.d[w12, 0]}, p0, [x14]\n"
        "st1d {za5v.d[w12, 0]}, p0, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "st1d {za2v.d[w12, 0]}, p0, [x15]\n"
        "st1d {za6v.d[w12, 0]}, p0, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "st1d {za3v.d[w12, 0]}, p0, [x16]\n"
        "st1d {za7v.d[w12, 0]}, p0, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        ".if %c[inplace]==0\n"
        "add x7, x7, x9\n"
        ".endif\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        ".if %c[inplace]\n"
        "add x7, x7, x9\n"
        ".endif\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d), [required]"i"(16+4), [rhs0]"i"(16*256+0), [rhs1]"i"(16*256+64), [rhs2]"i"(16*256+128), [rhs3]"i"(16*256+192), [rhs4]"i"(16*256+256), [rhs5]"i"(16*256+320), [rhs6]"i"(16*256+384), [rhs7]"i"(16*256+448), [rhs8]"i"(16*256+512), [rhs9]"i"(16*256+576), [rhs10]"i"(16*256+640), [rhs11]"i"(16*256+704), [rhs12]"i"(16*256+768), [rhs13]"i"(16*256+832), [rhs14]"i"(16*256+896), [rhs15]"i"(16*256+960), [subtract]"i"(subtract), [inplace]"i"(inplace), [pf0]"i"(0+0), [pf1]"i"(0+64), [pf2]"i"(0+128), [pf3]"i"(0+192) : "memory","cc","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}

#undef TC_SME_VECTOR_CLOBBERS
namespace tbstrassen {
struct Recipe {int x,y,sign;};
inline constexpr Recipe ar[7]={{0,3,1},{2,3,1},{0,-1,0},{3,-1,0},
                              {0,1,1},{2,0,-1},{1,3,-1}};
// B wire contains X0=B00+B11, X1=B01-B11, X2=B10-B00, X3=B11.
// This table also determines which received Q128 panels each product waits for.
inline constexpr Recipe br[7]={{0,-1,0},{0,3,-1},{1,-1,0},{2,-1,0},
                              {3,-1,0},{0,1,1},{0,2,1}};
inline constexpr int derived_slot[7]={0,1,-1,-1,2,3,4};
inline constexpr int derived_product[5]={0,1,4,5,6};
inline constexpr size_t a_product=1024ul*8000,c_product=1024ul*128;
inline size_t t_index(int row,int s,int u){
 const int u0=u/320*320,k=std::min(320,4000-u0);
 return size_t(u0)*8192+size_t(s*2048+row/16*16)*k+(u-u0)*16+row%16;
}
inline size_t partial_index(int row,int q){
 return size_t(row/1024*32+q/128)*c_product+size_t(row%1024/16)*2048+
        size_t(q%128/32)*512+(q%32)*16+row%16;
}
inline size_t transformed_index(int slot,int hs,int u0,int row,int k){
 return size_t(slot)*a_product+size_t(hs*4000+u0)*1024+size_t(row)*k;
}
// Two single-quadrant products alias the immutable input; only five sums
// are materialized. Source matrices are never changed or hot-aliased.
inline const double* t_quad(const double* t,int quad,int hs,int u0,int row){
 return t+t_index(row+(quad/2)*1024,hs+(quad%2)*2,u0);
}
static void transform_panel(const double* local_t,double* target,int slot,int hs,int k){
 const auto r=ar[derived_product[slot]];
 auto source=[&](int quad){return local_t+size_t((hs+quad%2*2)*2048+quad/2*1024)*k;};
 const double* x=source(r.x),*y=source(r.y);
 if(r.sign==1){
#pragma omp simd
  for(int i=0;i<1024*k;++i)target[i]=x[i]+y[i];
 }else{
#pragma omp simd
  for(int i=0;i<1024*k;++i)target[i]=x[i]-y[i];
 }
}
static void transform(const double* t,double* derived,int task){
 const int slot=task/(2*13),hs=task/13%2,u0=task%13*320,k=std::min(320,4000-u0);
 transform_panel(t+size_t(u0)*8192,derived+transformed_index(slot,hs,u0,0,k),slot,hs,k);
}
inline const double* a_panel(const double* t,const double* derived,int which,int hs,int u0){
 const int k=std::min(320,4000-u0),slot=derived_slot[which];
 return slot<0?t_quad(t,ar[which].x,hs,u0,0):
     derived+transformed_index(slot,hs,u0,0,k);
}
inline const double* b_quad(const double* b,int quad,int hs,int u0,int q0){
 return b+size_t(q0+(quad%2)*128)*16000+size_t((hs+quad/2*2)*4000+u0)*32;
}
static void update_panel(const double* a,const double* b,double* packed,double* partial,
                         int which,int hs,int u0,int q0,bool add=true,bool inplace=false){
 const int k=std::min(320,4000-u0);
 const auto r=br[which];
 std::array<const double*,64> rows;
 for(int i=0;i<64;++i)rows[i]=a+size_t(i)*16*k;
 SmeNPanel first{4,k,rows[0],b_quad(b,r.x,hs,u0,q0),partial,packed,rows[1],
                 16000L*32*8,long(k)*32*8,k*16*8/4,add};
 {
 if(inplace){ // TC_PAIR_PACK&1 (with TC_RHS_LA&1): packed holds the partner's y basis; X0 +- packed, in place
  SmeSumPanel sum{first,packed};
  if(r.sign==1)tc1_rhs_sum_lookahead<false,true>(sum);else tc1_rhs_sum_lookahead<true,true>(sum);
 }else if(r.y<0){
  if(1&1)tc1_rhs_pack_lookahead(first);else sme_npanel_pack_b_pipeline(first);
 }else{
  SmeSumPanel sum{first,b_quad(b,r.y,hs,u0,q0)};
  if(1&1){if(r.sign==1)tc1_rhs_sum_lookahead<false>(sum);else tc1_rhs_sum_lookahead<true>(sum);}
  else if(r.sign==1)sme_npanel_sum_pack_b_pipeline<false>(sum);
  else sme_npanel_sum_pack_b_pipeline<true>(sum);
 }
 }
 {
 SmeNPanel middle{4,k,rows[1],packed,partial+16*128,packed,rows[2],
                  long(k)*32*8,long(k)*32*8,k*16*8/4,add};
 SmeMRun run{{middle,partial+32*128},rows.data()+1,62};
 sme_mrun_accumulate(run);
 SmeNPanel last{4,k,rows[63],packed,partial+1008*128,packed,nullptr,
                long(k)*32*8,long(k)*32*8,k*16*8/4,add};
 sme_npanel_pipeline(last);
 }
}
static void update(const double* t,const double* derived,const double* b,double* packed,
                   double* partial,int which,int hs,int u0,int q0,bool add=true){
 update_panel(a_panel(t,derived,which,hs,u0),b,packed,partial,which,hs,u0,q0,add);
}
static void combine_store(const double* products,double* out,int pair){
 const double* base=products+size_t(pair)*7*c_product;
 alignas(64) double tile[4][2*512];
 for(int p0=0;p0<512;p0+=16)for(int nb=0;nb<4;++nb){
  for(int r=0;r<2;++r){
   const size_t at=size_t((r*512+p0)/16)*2048+nb*512;
   const double *a=base+at,*b=a+c_product,*c=b+c_product,*d=c+c_product,
                *e=d+c_product,*f=e+c_product,*g=f+c_product;
#pragma omp simd
   for(int i=0;i<512;++i){
    const double m1=a[i],m2=b[i],m3=c[i],m4=d[i],m5=e[i],m6=f[i],m7=g[i];
    tile[0][r*512+i]=((m1+m4)-m5)+m7;
    tile[1][r*512+i]=m3+m5;
    tile[2][r*512+i]=m2+m4;
    tile[3][r*512+i]=((m1-m2)+m3)+m6;
   }
  }
  for(int pi=0;pi<16;++pi)for(int j=0;j<32;++j){
   const int qlocal=pair%2*256+nb*32+j;
   const int q=pair/2*500+qlocal;
   const int p=p0+pi;
   if(p>=500)continue;
   for(int r=0;r<2;++r){
    const size_t i=r*512+j*16+pi;
    if(qlocal<500)out[size_t(p)*16000+q*4+r]=tile[0][i];
    if(qlocal+128<500)out[size_t(p)*16000+(q+128)*4+r]=tile[1][i];
    if(qlocal<500)out[size_t(p)*16000+q*4+r+2]=tile[2][i];
    if(qlocal+128<500)out[size_t(p)*16000+(q+128)*4+r+2]=tile[3][i];
   }
  }
 }
}
static void combine(const double* products,double* partial,int pair){
 const double* p=products+size_t(pair)*7*c_product;
 double* c00=partial+size_t(pair*2)*c_product;
 double* c01=c00+c_product,*c10=c00+32*c_product,*c11=c10+c_product;
#pragma omp simd
 for(size_t i=0;i<c_product;++i){
  const double m1=p[i],m2=p[c_product+i],m3=p[2*c_product+i],
               m4=p[3*c_product+i],m5=p[4*c_product+i],
               m6=p[5*c_product+i],m7=p[6*c_product+i];
  c00[i]=((m1+m4)-m5)+m7;c01[i]=m3+m5;
  c10[i]=m2+m4;c11[i]=((m1-m2)+m3)+m6;
 }
}
static void store(const double* partial,double* out,int owner){
 const int block=owner/32,qgroup=owner%32;
 tc1_store_r4_tile(out,16000,500,block*256,qgroup%4*128,500,qgroup/4,128,
                   partial+size_t(owner)*c_product,256);
}
}

} // namespace tcopt

struct Tc1Rows {
    int own = 500, p = 500, pm = 496, rows = 2000, tail_row = 1984;
};
static Tc1Rows tc1_layout;
static void tc1_set_layout(int own, int p) {
    if (p % 4 || own > 500 || p > 512 || p < own)
        std::abort();
    const int pm = own / 16 * 16;
    tc1_layout = {own, p, pm, 4 * p, 4 * pm};
}

// One K320 slab of a TC1 AC task; cp(row) gives the accumulators of the 16-row block at row (16x128 each), which
// are contiguous below row 1792 at least.
template <class Acc>
static void tc1_unpadded_slab(const double *a, const double *b, Acc cp, double *packed, double *output, int col, int m,
                              int row_start, int l) {
    constexpr int width = 128, k = 4000;
    const Tc1Rows L = tc1_layout;
    auto ap = [&](int row, int l) {
        row += row_start;
        int kc = std::min(320, k - l);
        return a + size_t(l) * L.rows + size_t(row) * kc;
    };
    const int kc = std::min(320, k - l);
    auto single = [&](int row) {
        const double *future = row + 16 < m ? ap(row + 16, l) : l + 2 * kc <= k ? ap(0, l + kc) : nullptr;
        SmeNPanel d{4,      kc,          ap(row, l),  packed,          cp(row), packed,
                    future, kc * 32 * 8, kc * 32 * 8, kc * 16 * 8 / 4, l != 0};
        if (!row) {
            d.b = b + size_t(l) * 32;
            d.b_stride = k * 32 * 8;
        }
        const int R = row_start + row;
        if (l + kc == k && R < L.tail_row) {
            const int si = R / L.pm, p = R % L.pm;
            const long reduction = si * 4000 + col / 4;
            SmeTPanel td{d,
                         output + long(p / 64) * 256 * 16000 + (reduction / 320 * 320) * 256 + (p % 64) * 320 +
                             (reduction % 320) * 16,
                         64L * 320 * 8};
            if (!row)
                sme_tstore_pack_b_pipeline(td);
            else if (future)
                sme_tstore_prefetch_a(td);
            else
                sme_tstore_pipeline(td);
        } else if (l) {
            SmeCPanel cd{d, row + 16 < m ? cp(row + 16) : cp(0)};
            if (!row)
                sme_npanel_pack_b_pipeline_c64_spread32(cd);
            else if (future)
                sme_npanel_prefetch_a_c64_spread32(cd);
            else
                sme_npanel_pipeline_c64_spread32(cd);
        } else {
            if (!row)
                sme_npanel_pack_b_pipeline(d);
            else if (future)
                sme_npanel_prefetch_a(d);
            else
                sme_npanel_pipeline(d);
        }
    };
    if (!l || l + kc == k || m <= 32) {
        for (int row = 0; row < m; row += 16)
            single(row);
        return;
    }
    std::array<const double *, 128> rows;
    for (int i = 0; i < m / 16; ++i)
        rows[i] = ap(i * 16, l);
    auto run = [&](int first, int count) {
        SmeNPanel d{4,           kc,          rows[first / 16], packed, cp(first), packed, rows[first / 16 + 1],
                    kc * 32 * 8, kc * 32 * 8, kc * 16 * 8 / 4,  1};
        SmeMRun r{{d, cp(first + 16)}, rows.data() + first / 16, count};
        sme_mrun_accumulate(r);
    };
    single(0);
    if (m <= 1792 || cp(m - 16) == cp(0) + size_t(m - 16) * width) {
        run(16, m / 16 - 2);
        single(m - 16);
    } else {
        run(16, 110);
        single(1776);
        if (m > 1808)
            run(1792, (m - 1792) / 16 - 1);
        single(m - 16);
    }
}
// Tail blocks: accumulators are column-major 16x32 per panel; scatter into the TB layout of T
// (TB row = (p/64)*256 + r*64 + p%64, reduction = s*4000 + u, K320 slabs of 256-row groups).
template <class Acc> static void tc1_unpadded_scatter(Acc cp, double *output, int col, int m, int row_start) {
    constexpr int width = 128;
    const Tc1Rows L = tc1_layout;
    for (int R = std::max(row_start, L.tail_row); R < row_start + m; R += 16) {
        const double *block = cp(R - row_start);
        const int p0 = L.pm + (R - L.tail_row) / 4;
        for (int n = 0; n < width; ++n) {
            const int u = (col + n) / 4, r = n % 4;
            for (int i = 0; i < 16; ++i) {
                const int si = i / 4, p = p0 + i % 4;
                const long row = long(p / 64) * 256 + r * 64 + p % 64, red = long(si) * 4000 + u;
                output[(row / 256) * 256 * 16000 + (red / 320 * 320) * 256 + (row % 256 / 16 * 16) * 320 +
                       (red % 320) * 16 + row % 16] = block[(n / 32) * 512 + (n % 32) * 16 + i];
            }
        }
    }
}

static double *tc1_acc = nullptr;
constexpr size_t tc1_acc_doubles = size_t(2048) * 16000;
static void tc1_leaf_kstep(const double *a, const double *b, double *c, double *packed, double *output, int col, int m,
                           int row_start, int l) {
    auto cp = [&](int row) { return c + size_t(row) * 128; };
    tc1_unpadded_slab(a, b, cp, packed, output, col, m, row_start, l);
    if (l + 320 >= 4000)
        tc1_unpadded_scatter(cp, output, col, m, row_start);
}

// Terminal AC stores for the seven TB Strassen operands.
// The FMOPA/prefetch bodies match the previous production AC kernels.
struct SmeTc2StrassenTPanel {
    SmeTPanel base;
    double *destinations[4][7];
    long pad;
};
static_assert(offsetof(SmeTc2StrassenTPanel, destinations)==104 &&
              offsetof(SmeTc2StrassenTPanel, pad)==328 &&
              sizeof(SmeTc2StrassenTPanel)==336, "TC2 Strassen descriptor");
static int tc2_strassen_padded_p=1024;
static size_t tc2_strassen_index(int product,int p,int u) {
    const int u0=u/320*320,depth=std::min(320,4000-u0);
    return size_t(product)*tc2_strassen_padded_p*4000 +
           size_t(u0)*tc2_strassen_padded_p + size_t(p/16*16)*depth +
           (u-u0)*16+p%16;
}
__attribute__((noinline, aligned(64))) static void sme_tc2_strassen_t_pipeline(const SmeTc2StrassenTPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z6.d}, p0/z, [x9]\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z12.d}, p0/z, [x9]\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z18.d}, p0/z, [x9]\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #32\n"
        "b.lt 94f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #32\n"
        "b.ge 1b\n"
        "94:\n"
        "add x5, x24, #4096\n"
        "cmp x27, #1\n"
        "b.ne 98f\n"
        "ldr x5, [x28, #88]\n"
        "98:\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "95:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #64]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #192]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #320]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #448]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "add x5, x5, #512\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 95b\n"
        "8:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "prfm pldl1keep, [x5, #64]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "prfm pldl1keep, [x5, #192]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "prfm pldl1keep, [x5, #320]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "prfm pldl1keep, [x5, #448]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "ldr x9, [x28]\n"
        "sub x9, x9, x27\n"
        "mov x10, #56\n"
        "mul x9, x9, x10\n"
        "add x10, x28, #104\n"
        "add x10, x10, x9\n"
        "ldp x0, x1, [x10]\n"
        "ldp x2, x3, [x10, #16]\n"
        "ldp x13, x14, [x10, #32]\n"
        "ldr x15, [x10, #48]\n"
        "ldr x7, [x28, #328]\n"
        "mov z16.d, #0\n"
        "mov w12, #0\n"
        "93:\n"
        "lsl x9, x12, #3\n"
        "mova z0.d, p0/m, za0v.d[w12, 0]\n"
        "mova z1.d, p0/m, za4v.d[w12, 0]\n"
        "uzp1 z2.d, z0.d, z1.d\n"
        "uzp2 z3.d, z0.d, z1.d\n"
        "add w12, w12, #1\n"
        "mova z4.d, p0/m, za0v.d[w12, 0]\n"
        "mova z5.d, p0/m, za4v.d[w12, 0]\n"
        "uzp1 z6.d, z4.d, z5.d\n"
        "uzp2 z7.d, z4.d, z5.d\n"
        "sub w12, w12, #1\n"
        "fadd z8.d, z2.d, z7.d\n"
        "fadd z9.d, z6.d, z7.d\n"
        "fadd z10.d, z2.d, z3.d\n"
        "fsub z11.d, z6.d, z2.d\n"
        "fsub z12.d, z3.d, z7.d\n"
        "st1d {z8.d}, p0, [x0, x9, lsl #3]\n"
        "st1d {z9.d}, p0, [x1, x9, lsl #3]\n"
        "st1d {z2.d}, p0, [x2, x9, lsl #3]\n"
        "st1d {z7.d}, p0, [x3, x9, lsl #3]\n"
        "st1d {z10.d}, p0, [x13, x9, lsl #3]\n"
        "st1d {z11.d}, p0, [x14, x9, lsl #3]\n"
        "st1d {z12.d}, p0, [x15, x9, lsl #3]\n"
        "cbz x7, 200f\n"
        "add x10, x9, #8\n"
        "st1d {z16.d}, p0, [x0, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x1, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x2, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x3, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x13, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x14, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x15, x10, lsl #3]\n"
        "200:\n"
        "add x9, x9, #64\n"
        "mova z0.d, p0/m, za1v.d[w12, 0]\n"
        "mova z1.d, p0/m, za5v.d[w12, 0]\n"
        "uzp1 z2.d, z0.d, z1.d\n"
        "uzp2 z3.d, z0.d, z1.d\n"
        "add w12, w12, #1\n"
        "mova z4.d, p0/m, za1v.d[w12, 0]\n"
        "mova z5.d, p0/m, za5v.d[w12, 0]\n"
        "uzp1 z6.d, z4.d, z5.d\n"
        "uzp2 z7.d, z4.d, z5.d\n"
        "sub w12, w12, #1\n"
        "fadd z8.d, z2.d, z7.d\n"
        "fadd z9.d, z6.d, z7.d\n"
        "fadd z10.d, z2.d, z3.d\n"
        "fsub z11.d, z6.d, z2.d\n"
        "fsub z12.d, z3.d, z7.d\n"
        "st1d {z8.d}, p0, [x0, x9, lsl #3]\n"
        "st1d {z9.d}, p0, [x1, x9, lsl #3]\n"
        "st1d {z2.d}, p0, [x2, x9, lsl #3]\n"
        "st1d {z7.d}, p0, [x3, x9, lsl #3]\n"
        "st1d {z10.d}, p0, [x13, x9, lsl #3]\n"
        "st1d {z11.d}, p0, [x14, x9, lsl #3]\n"
        "st1d {z12.d}, p0, [x15, x9, lsl #3]\n"
        "cbz x7, 201f\n"
        "add x10, x9, #8\n"
        "st1d {z16.d}, p0, [x0, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x1, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x2, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x3, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x13, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x14, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x15, x10, lsl #3]\n"
        "201:\n"
        "add x9, x9, #64\n"
        "mova z0.d, p0/m, za2v.d[w12, 0]\n"
        "mova z1.d, p0/m, za6v.d[w12, 0]\n"
        "uzp1 z2.d, z0.d, z1.d\n"
        "uzp2 z3.d, z0.d, z1.d\n"
        "add w12, w12, #1\n"
        "mova z4.d, p0/m, za2v.d[w12, 0]\n"
        "mova z5.d, p0/m, za6v.d[w12, 0]\n"
        "uzp1 z6.d, z4.d, z5.d\n"
        "uzp2 z7.d, z4.d, z5.d\n"
        "sub w12, w12, #1\n"
        "fadd z8.d, z2.d, z7.d\n"
        "fadd z9.d, z6.d, z7.d\n"
        "fadd z10.d, z2.d, z3.d\n"
        "fsub z11.d, z6.d, z2.d\n"
        "fsub z12.d, z3.d, z7.d\n"
        "st1d {z8.d}, p0, [x0, x9, lsl #3]\n"
        "st1d {z9.d}, p0, [x1, x9, lsl #3]\n"
        "st1d {z2.d}, p0, [x2, x9, lsl #3]\n"
        "st1d {z7.d}, p0, [x3, x9, lsl #3]\n"
        "st1d {z10.d}, p0, [x13, x9, lsl #3]\n"
        "st1d {z11.d}, p0, [x14, x9, lsl #3]\n"
        "st1d {z12.d}, p0, [x15, x9, lsl #3]\n"
        "cbz x7, 202f\n"
        "add x10, x9, #8\n"
        "st1d {z16.d}, p0, [x0, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x1, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x2, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x3, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x13, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x14, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x15, x10, lsl #3]\n"
        "202:\n"
        "add x9, x9, #64\n"
        "mova z0.d, p0/m, za3v.d[w12, 0]\n"
        "mova z1.d, p0/m, za7v.d[w12, 0]\n"
        "uzp1 z2.d, z0.d, z1.d\n"
        "uzp2 z3.d, z0.d, z1.d\n"
        "add w12, w12, #1\n"
        "mova z4.d, p0/m, za3v.d[w12, 0]\n"
        "mova z5.d, p0/m, za7v.d[w12, 0]\n"
        "uzp1 z6.d, z4.d, z5.d\n"
        "uzp2 z7.d, z4.d, z5.d\n"
        "sub w12, w12, #1\n"
        "fadd z8.d, z2.d, z7.d\n"
        "fadd z9.d, z6.d, z7.d\n"
        "fadd z10.d, z2.d, z3.d\n"
        "fsub z11.d, z6.d, z2.d\n"
        "fsub z12.d, z3.d, z7.d\n"
        "st1d {z8.d}, p0, [x0, x9, lsl #3]\n"
        "st1d {z9.d}, p0, [x1, x9, lsl #3]\n"
        "st1d {z2.d}, p0, [x2, x9, lsl #3]\n"
        "st1d {z7.d}, p0, [x3, x9, lsl #3]\n"
        "st1d {z10.d}, p0, [x13, x9, lsl #3]\n"
        "st1d {z11.d}, p0, [x14, x9, lsl #3]\n"
        "st1d {z12.d}, p0, [x15, x9, lsl #3]\n"
        "cbz x7, 203f\n"
        "add x10, x9, #8\n"
        "st1d {z16.d}, p0, [x0, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x1, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x2, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x3, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x13, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x14, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x15, x10, lsl #3]\n"
        "203:\n"
        "add x9, x9, #64\n"
        "add w12, w12, #2\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","cc","x0","x1","x2","x3","x7","x5","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}
__attribute__((noinline, aligned(64))) static void sme_tc2_strassen_t_prefetch_a(const SmeTc2StrassenTPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "mov x19, x26\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z6.d}, p0/z, [x9]\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z12.d}, p0/z, [x9]\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z18.d}, p0/z, [x9]\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #32\n"
        "b.lt 94f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "prfm pldl2keep, [x19]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "prfm pldl2keep, [x19]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #32\n"
        "b.ge 1b\n"
        "94:\n"
        "add x5, x24, #4096\n"
        "cmp x27, #1\n"
        "b.ne 98f\n"
        "ldr x5, [x28, #88]\n"
        "98:\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "95:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #64]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #192]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #320]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #448]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "add x5, x5, #512\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 95b\n"
        "8:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "prfm pldl1keep, [x5, #64]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "prfm pldl1keep, [x5, #192]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "prfm pldl1keep, [x5, #320]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "prfm pldl1keep, [x5, #448]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "ldr x9, [x28]\n"
        "sub x9, x9, x27\n"
        "mov x10, #56\n"
        "mul x9, x9, x10\n"
        "add x10, x28, #104\n"
        "add x10, x10, x9\n"
        "ldp x0, x1, [x10]\n"
        "ldp x2, x3, [x10, #16]\n"
        "ldp x13, x14, [x10, #32]\n"
        "ldr x15, [x10, #48]\n"
        "ldr x7, [x28, #328]\n"
        "mov z16.d, #0\n"
        "mov w12, #0\n"
        "93:\n"
        "lsl x9, x12, #3\n"
        "mova z0.d, p0/m, za0v.d[w12, 0]\n"
        "mova z1.d, p0/m, za4v.d[w12, 0]\n"
        "uzp1 z2.d, z0.d, z1.d\n"
        "uzp2 z3.d, z0.d, z1.d\n"
        "add w12, w12, #1\n"
        "mova z4.d, p0/m, za0v.d[w12, 0]\n"
        "mova z5.d, p0/m, za4v.d[w12, 0]\n"
        "uzp1 z6.d, z4.d, z5.d\n"
        "uzp2 z7.d, z4.d, z5.d\n"
        "sub w12, w12, #1\n"
        "fadd z8.d, z2.d, z7.d\n"
        "fadd z9.d, z6.d, z7.d\n"
        "fadd z10.d, z2.d, z3.d\n"
        "fsub z11.d, z6.d, z2.d\n"
        "fsub z12.d, z3.d, z7.d\n"
        "st1d {z8.d}, p0, [x0, x9, lsl #3]\n"
        "st1d {z9.d}, p0, [x1, x9, lsl #3]\n"
        "st1d {z2.d}, p0, [x2, x9, lsl #3]\n"
        "st1d {z7.d}, p0, [x3, x9, lsl #3]\n"
        "st1d {z10.d}, p0, [x13, x9, lsl #3]\n"
        "st1d {z11.d}, p0, [x14, x9, lsl #3]\n"
        "st1d {z12.d}, p0, [x15, x9, lsl #3]\n"
        "cbz x7, 200f\n"
        "add x10, x9, #8\n"
        "st1d {z16.d}, p0, [x0, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x1, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x2, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x3, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x13, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x14, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x15, x10, lsl #3]\n"
        "200:\n"
        "add x9, x9, #64\n"
        "mova z0.d, p0/m, za1v.d[w12, 0]\n"
        "mova z1.d, p0/m, za5v.d[w12, 0]\n"
        "uzp1 z2.d, z0.d, z1.d\n"
        "uzp2 z3.d, z0.d, z1.d\n"
        "add w12, w12, #1\n"
        "mova z4.d, p0/m, za1v.d[w12, 0]\n"
        "mova z5.d, p0/m, za5v.d[w12, 0]\n"
        "uzp1 z6.d, z4.d, z5.d\n"
        "uzp2 z7.d, z4.d, z5.d\n"
        "sub w12, w12, #1\n"
        "fadd z8.d, z2.d, z7.d\n"
        "fadd z9.d, z6.d, z7.d\n"
        "fadd z10.d, z2.d, z3.d\n"
        "fsub z11.d, z6.d, z2.d\n"
        "fsub z12.d, z3.d, z7.d\n"
        "st1d {z8.d}, p0, [x0, x9, lsl #3]\n"
        "st1d {z9.d}, p0, [x1, x9, lsl #3]\n"
        "st1d {z2.d}, p0, [x2, x9, lsl #3]\n"
        "st1d {z7.d}, p0, [x3, x9, lsl #3]\n"
        "st1d {z10.d}, p0, [x13, x9, lsl #3]\n"
        "st1d {z11.d}, p0, [x14, x9, lsl #3]\n"
        "st1d {z12.d}, p0, [x15, x9, lsl #3]\n"
        "cbz x7, 201f\n"
        "add x10, x9, #8\n"
        "st1d {z16.d}, p0, [x0, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x1, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x2, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x3, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x13, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x14, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x15, x10, lsl #3]\n"
        "201:\n"
        "add x9, x9, #64\n"
        "mova z0.d, p0/m, za2v.d[w12, 0]\n"
        "mova z1.d, p0/m, za6v.d[w12, 0]\n"
        "uzp1 z2.d, z0.d, z1.d\n"
        "uzp2 z3.d, z0.d, z1.d\n"
        "add w12, w12, #1\n"
        "mova z4.d, p0/m, za2v.d[w12, 0]\n"
        "mova z5.d, p0/m, za6v.d[w12, 0]\n"
        "uzp1 z6.d, z4.d, z5.d\n"
        "uzp2 z7.d, z4.d, z5.d\n"
        "sub w12, w12, #1\n"
        "fadd z8.d, z2.d, z7.d\n"
        "fadd z9.d, z6.d, z7.d\n"
        "fadd z10.d, z2.d, z3.d\n"
        "fsub z11.d, z6.d, z2.d\n"
        "fsub z12.d, z3.d, z7.d\n"
        "st1d {z8.d}, p0, [x0, x9, lsl #3]\n"
        "st1d {z9.d}, p0, [x1, x9, lsl #3]\n"
        "st1d {z2.d}, p0, [x2, x9, lsl #3]\n"
        "st1d {z7.d}, p0, [x3, x9, lsl #3]\n"
        "st1d {z10.d}, p0, [x13, x9, lsl #3]\n"
        "st1d {z11.d}, p0, [x14, x9, lsl #3]\n"
        "st1d {z12.d}, p0, [x15, x9, lsl #3]\n"
        "cbz x7, 202f\n"
        "add x10, x9, #8\n"
        "st1d {z16.d}, p0, [x0, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x1, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x2, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x3, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x13, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x14, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x15, x10, lsl #3]\n"
        "202:\n"
        "add x9, x9, #64\n"
        "mova z0.d, p0/m, za3v.d[w12, 0]\n"
        "mova z1.d, p0/m, za7v.d[w12, 0]\n"
        "uzp1 z2.d, z0.d, z1.d\n"
        "uzp2 z3.d, z0.d, z1.d\n"
        "add w12, w12, #1\n"
        "mova z4.d, p0/m, za3v.d[w12, 0]\n"
        "mova z5.d, p0/m, za7v.d[w12, 0]\n"
        "uzp1 z6.d, z4.d, z5.d\n"
        "uzp2 z7.d, z4.d, z5.d\n"
        "sub w12, w12, #1\n"
        "fadd z8.d, z2.d, z7.d\n"
        "fadd z9.d, z6.d, z7.d\n"
        "fadd z10.d, z2.d, z3.d\n"
        "fsub z11.d, z6.d, z2.d\n"
        "fsub z12.d, z3.d, z7.d\n"
        "st1d {z8.d}, p0, [x0, x9, lsl #3]\n"
        "st1d {z9.d}, p0, [x1, x9, lsl #3]\n"
        "st1d {z2.d}, p0, [x2, x9, lsl #3]\n"
        "st1d {z7.d}, p0, [x3, x9, lsl #3]\n"
        "st1d {z10.d}, p0, [x13, x9, lsl #3]\n"
        "st1d {z11.d}, p0, [x14, x9, lsl #3]\n"
        "st1d {z12.d}, p0, [x15, x9, lsl #3]\n"
        "cbz x7, 203f\n"
        "add x10, x9, #8\n"
        "st1d {z16.d}, p0, [x0, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x1, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x2, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x3, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x13, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x14, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x15, x10, lsl #3]\n"
        "203:\n"
        "add x9, x9, #64\n"
        "add w12, w12, #2\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","cc","x0","x1","x2","x3","x7","x5","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}
__attribute__((noinline, aligned(64))) static void sme_tc2_strassen_t_pack_b_pipeline(const SmeTc2StrassenTPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "mov x19, x25\n"
        "mov x20, #256\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "st1d {z8.d}, p0, [x19]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z9.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z10.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z11.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "st1d {z14.d}, p0, [x19]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z15.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z16.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z17.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "st1d {z20.d}, p0, [x19]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z21.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z22.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z23.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #32\n"
        "b.lt 94f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "st1d {z8.d}, p0, [x19]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z9.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z10.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z11.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "st1d {z14.d}, p0, [x19]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z15.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z16.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z17.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "st1d {z20.d}, p0, [x19]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z21.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z22.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z23.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #32\n"
        "b.ge 1b\n"
        "94:\n"
        "add x5, x24, #4096\n"
        "cmp x27, #1\n"
        "b.ne 98f\n"
        "ldr x5, [x28, #88]\n"
        "98:\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "95:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #64]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #192]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "st1d {z8.d}, p0, [x19]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z9.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z10.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z11.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #320]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "st1d {z14.d}, p0, [x19]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z15.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z16.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z17.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "prfm pldl1keep, [x5, #448]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "st1d {z20.d}, p0, [x19]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z21.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z22.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z23.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "add x5, x5, #512\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 95b\n"
        "8:\n"
        "prfm pldl1keep, [x5, #0]\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "prfm pldl1keep, [x5, #64]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "prfm pldl1keep, [x5, #128]\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "prfm pldl1keep, [x5, #192]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "prfm pldl1keep, [x5, #256]\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "prfm pldl1keep, [x5, #320]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "prfm pldl1keep, [x5, #384]\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "prfm pldl1keep, [x5, #448]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "ldr x9, [x28]\n"
        "sub x9, x9, x27\n"
        "mov x10, #56\n"
        "mul x9, x9, x10\n"
        "add x10, x28, #104\n"
        "add x10, x10, x9\n"
        "ldp x0, x1, [x10]\n"
        "ldp x2, x3, [x10, #16]\n"
        "ldp x13, x14, [x10, #32]\n"
        "ldr x15, [x10, #48]\n"
        "ldr x7, [x28, #328]\n"
        "mov z16.d, #0\n"
        "mov w12, #0\n"
        "93:\n"
        "lsl x9, x12, #3\n"
        "mova z0.d, p0/m, za0v.d[w12, 0]\n"
        "mova z1.d, p0/m, za4v.d[w12, 0]\n"
        "uzp1 z2.d, z0.d, z1.d\n"
        "uzp2 z3.d, z0.d, z1.d\n"
        "add w12, w12, #1\n"
        "mova z4.d, p0/m, za0v.d[w12, 0]\n"
        "mova z5.d, p0/m, za4v.d[w12, 0]\n"
        "uzp1 z6.d, z4.d, z5.d\n"
        "uzp2 z7.d, z4.d, z5.d\n"
        "sub w12, w12, #1\n"
        "fadd z8.d, z2.d, z7.d\n"
        "fadd z9.d, z6.d, z7.d\n"
        "fadd z10.d, z2.d, z3.d\n"
        "fsub z11.d, z6.d, z2.d\n"
        "fsub z12.d, z3.d, z7.d\n"
        "st1d {z8.d}, p0, [x0, x9, lsl #3]\n"
        "st1d {z9.d}, p0, [x1, x9, lsl #3]\n"
        "st1d {z2.d}, p0, [x2, x9, lsl #3]\n"
        "st1d {z7.d}, p0, [x3, x9, lsl #3]\n"
        "st1d {z10.d}, p0, [x13, x9, lsl #3]\n"
        "st1d {z11.d}, p0, [x14, x9, lsl #3]\n"
        "st1d {z12.d}, p0, [x15, x9, lsl #3]\n"
        "cbz x7, 200f\n"
        "add x10, x9, #8\n"
        "st1d {z16.d}, p0, [x0, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x1, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x2, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x3, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x13, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x14, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x15, x10, lsl #3]\n"
        "200:\n"
        "add x9, x9, #64\n"
        "mova z0.d, p0/m, za1v.d[w12, 0]\n"
        "mova z1.d, p0/m, za5v.d[w12, 0]\n"
        "uzp1 z2.d, z0.d, z1.d\n"
        "uzp2 z3.d, z0.d, z1.d\n"
        "add w12, w12, #1\n"
        "mova z4.d, p0/m, za1v.d[w12, 0]\n"
        "mova z5.d, p0/m, za5v.d[w12, 0]\n"
        "uzp1 z6.d, z4.d, z5.d\n"
        "uzp2 z7.d, z4.d, z5.d\n"
        "sub w12, w12, #1\n"
        "fadd z8.d, z2.d, z7.d\n"
        "fadd z9.d, z6.d, z7.d\n"
        "fadd z10.d, z2.d, z3.d\n"
        "fsub z11.d, z6.d, z2.d\n"
        "fsub z12.d, z3.d, z7.d\n"
        "st1d {z8.d}, p0, [x0, x9, lsl #3]\n"
        "st1d {z9.d}, p0, [x1, x9, lsl #3]\n"
        "st1d {z2.d}, p0, [x2, x9, lsl #3]\n"
        "st1d {z7.d}, p0, [x3, x9, lsl #3]\n"
        "st1d {z10.d}, p0, [x13, x9, lsl #3]\n"
        "st1d {z11.d}, p0, [x14, x9, lsl #3]\n"
        "st1d {z12.d}, p0, [x15, x9, lsl #3]\n"
        "cbz x7, 201f\n"
        "add x10, x9, #8\n"
        "st1d {z16.d}, p0, [x0, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x1, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x2, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x3, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x13, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x14, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x15, x10, lsl #3]\n"
        "201:\n"
        "add x9, x9, #64\n"
        "mova z0.d, p0/m, za2v.d[w12, 0]\n"
        "mova z1.d, p0/m, za6v.d[w12, 0]\n"
        "uzp1 z2.d, z0.d, z1.d\n"
        "uzp2 z3.d, z0.d, z1.d\n"
        "add w12, w12, #1\n"
        "mova z4.d, p0/m, za2v.d[w12, 0]\n"
        "mova z5.d, p0/m, za6v.d[w12, 0]\n"
        "uzp1 z6.d, z4.d, z5.d\n"
        "uzp2 z7.d, z4.d, z5.d\n"
        "sub w12, w12, #1\n"
        "fadd z8.d, z2.d, z7.d\n"
        "fadd z9.d, z6.d, z7.d\n"
        "fadd z10.d, z2.d, z3.d\n"
        "fsub z11.d, z6.d, z2.d\n"
        "fsub z12.d, z3.d, z7.d\n"
        "st1d {z8.d}, p0, [x0, x9, lsl #3]\n"
        "st1d {z9.d}, p0, [x1, x9, lsl #3]\n"
        "st1d {z2.d}, p0, [x2, x9, lsl #3]\n"
        "st1d {z7.d}, p0, [x3, x9, lsl #3]\n"
        "st1d {z10.d}, p0, [x13, x9, lsl #3]\n"
        "st1d {z11.d}, p0, [x14, x9, lsl #3]\n"
        "st1d {z12.d}, p0, [x15, x9, lsl #3]\n"
        "cbz x7, 202f\n"
        "add x10, x9, #8\n"
        "st1d {z16.d}, p0, [x0, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x1, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x2, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x3, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x13, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x14, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x15, x10, lsl #3]\n"
        "202:\n"
        "add x9, x9, #64\n"
        "mova z0.d, p0/m, za3v.d[w12, 0]\n"
        "mova z1.d, p0/m, za7v.d[w12, 0]\n"
        "uzp1 z2.d, z0.d, z1.d\n"
        "uzp2 z3.d, z0.d, z1.d\n"
        "add w12, w12, #1\n"
        "mova z4.d, p0/m, za3v.d[w12, 0]\n"
        "mova z5.d, p0/m, za7v.d[w12, 0]\n"
        "uzp1 z6.d, z4.d, z5.d\n"
        "uzp2 z7.d, z4.d, z5.d\n"
        "sub w12, w12, #1\n"
        "fadd z8.d, z2.d, z7.d\n"
        "fadd z9.d, z6.d, z7.d\n"
        "fadd z10.d, z2.d, z3.d\n"
        "fsub z11.d, z6.d, z2.d\n"
        "fsub z12.d, z3.d, z7.d\n"
        "st1d {z8.d}, p0, [x0, x9, lsl #3]\n"
        "st1d {z9.d}, p0, [x1, x9, lsl #3]\n"
        "st1d {z2.d}, p0, [x2, x9, lsl #3]\n"
        "st1d {z7.d}, p0, [x3, x9, lsl #3]\n"
        "st1d {z10.d}, p0, [x13, x9, lsl #3]\n"
        "st1d {z11.d}, p0, [x14, x9, lsl #3]\n"
        "st1d {z12.d}, p0, [x15, x9, lsl #3]\n"
        "cbz x7, 203f\n"
        "add x10, x9, #8\n"
        "st1d {z16.d}, p0, [x0, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x1, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x2, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x3, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x13, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x14, x10, lsl #3]\n"
        "st1d {z16.d}, p0, [x15, x10, lsl #3]\n"
        "203:\n"
        "add x9, x9, #64\n"
        "add w12, w12, #2\n"
        "cmp w12, #8\n"
        "b.ne 93b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","cc","x0","x1","x2","x3","x7","x5","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}

// Packed A rows are (p,t). If P is not M16-aligned, the last AC microtile
// also zeroes the upper eight p lanes in every Strassen operand.
static int tc2_rows_stride = 2000, tc2_zero_row = 1984;

// TC2: advance ready AC tasks by one K320 slab, retaining each accumulator.
static void tc2_leaf_kstep(int m, int width, const double *a, const double *b, double *c, double *packed, double *t,
                           int col, int row_start, int first_k) {
    constexpr int k = 8000, b_full_k = 8000;
    const int full_rows = tc2_rows_stride;
    auto ap = [&](int ib, int l) { return a + size_t(l) * full_rows + size_t(row_start + ib) * std::min(320, k - l); };
    for (int l = first_k; l < first_k + 320; l += 320) {
        const int kc = std::min(320, k - l);
        auto single = [&](int ib) {
            const double *future = ib + 16 < m ? ap(ib + 16, l) : l + 2 * kc <= k ? ap(0, l + kc) : nullptr;
            SmeNPanel d{width / 32, kc,          ap(ib, l),   packed,          c + ib * width, packed,
                        future,     kc * 32 * 8, kc * 32 * 8, kc * 16 * 8 / 4, l != 0};
            if (!ib) {
                d.b = b + size_t(l) * 32;
                d.b_stride = b_full_k * 32 * 8;
            }
            SmeCPanel cd{d, c + (ib + 16 < m ? ib + 16 : 0) * width};
            if (l + kc == k) {
                const int p=(row_start+ib)/2;
                SmeTc2StrassenTPanel td{{d,nullptr,0},{},row_start+ib==tc2_zero_row};
                td.base.output=ib+16<m?c+(ib+16)*width:nullptr;
                for(int panel=0;panel<width/32;++panel)
                    for(int product=0;product<7;++product)
                        td.destinations[panel][product]=t+tc2_strassen_index(product,p,(col+panel*32)/2);
                if(!ib) sme_tc2_strassen_t_pack_b_pipeline(td);
                else if(future) sme_tc2_strassen_t_prefetch_a(td);
                else sme_tc2_strassen_t_pipeline(td);
                return;
            }
            if (l) {
                if (!ib)
                    sme_npanel_pack_b_pipeline_c64_spread32(cd);
                else if (future)
                    sme_npanel_prefetch_a_c64_spread32(cd);
                else
                    sme_npanel_pipeline_c64_spread32(cd);
            } else {
                if (!ib)
                    sme_npanel_pack_b_pipeline(d);
                else if (future)
                    sme_npanel_prefetch_a(d);
                else
                    sme_npanel_pipeline(d);
            }
        };
        {

            single(0);
        }
        {

        if (!l || l + kc == k || width != 128 || m <= 32) {
            for (int ib = 16; ib < m; ib += 16) single(ib);
            continue;
        }
        std::array<const double *, 128> rows;
        if (m / 16 > int(rows.size()))
            std::abort();
        for (int i = 0; i < m / 16; ++i)
            rows[i] = ap(i * 16, l);
        SmeNPanel base{4,           kc,          rows[1],         packed, c + 16 * 128, packed, rows[2],
                       kc * 32 * 8, kc * 32 * 8, kc * 16 * 8 / 4, 1};
        SmeMRun run{{base, c + 32 * 128}, rows.data() + 1, m / 16 - 2};
        sme_mrun_accumulate(run);
        single(m - 16);
        }
    }
}

struct Tc2KFirstQueue {
    struct alignas(64) Bank {
        std::atomic<uint64_t> bits[2]{};
    };
    Bank banks[25];
    std::atomic<int> remaining{2500}, final_claimed{0};
    Tc2KFirstQueue() {
        for (auto &bank : banks)
            for (auto &bits : bank.bits)
                bits.store(0, std::memory_order_relaxed);
        banks[0].bits[0].store(~uint64_t(0));
        banks[0].bits[1].store((uint64_t(1) << 36) - 1);
    }
    template <class F, class Ready> void drain(const unsigned *order, F execute, Ready ready, int step_limit = 25) {
        while (final_claimed.load(std::memory_order_acquire) < 100) {
            bool claimed = false;
            for (int k = 0; k < std::min(25, step_limit) && !claimed; ++k)
                for (int half = 0; half < 2 && !claimed; ++half) {
                    auto &bank = banks[k].bits[half];
                    uint64_t word = bank.load(std::memory_order_relaxed);
                    while (word) {
                        uint64_t eligible = word;
                        while (eligible && k == 0 && !ready(order[half * 64 + __builtin_ctzll(eligible)]))
                            eligible &= eligible - 1;
                        if (!eligible)
                            break;
                        const unsigned bit = __builtin_ctzll(eligible);
                        const uint64_t mask = uint64_t(1) << bit;
                        if (!bank.compare_exchange_weak(word, word & ~mask, std::memory_order_acquire,
                                                        std::memory_order_relaxed))
                            continue;
                        const unsigned priority = half * 64 + bit;
                        if (priority >= 100)
                            std::abort();
                        if (k == 24)
                            final_claimed.fetch_add(1, std::memory_order_release);
                        execute(order[priority], k);
                        if (k < 24)
                            banks[k + 1].bits[half].fetch_or(mask, std::memory_order_release);
                        remaining.fetch_sub(1, std::memory_order_release);
                        claimed = true;
                        break;
                    }
                }
            if (!claimed)
                asm volatile("yield" ::: "memory");
        }
    }
    template <class F> void drain(const unsigned *order, F execute) {
        drain(order, execute, [](unsigned) { return true; });
    }
};

// Tc2KFirstQueue for up to 64*Words tasks and 32 steps: step k+1 of a task becomes claimable once step k ran;
// workers take the lowest claimable step, tasks in priority order (ready() gates step 0).
template <int Words> struct KFirstQueue {
    struct alignas(64) Bank {
        std::atomic<uint64_t> bits[Words]{};
    };
    Bank banks[32];
    const int tasks, steps;
    std::atomic<int> final_claimed{0};
    KFirstQueue(int task_count, int step_count) : tasks(task_count), steps(step_count) {
        if (steps > 32 || tasks > 64 * Words)
            std::abort();
        for (int i = 0; i < Words; ++i) {
            const int n = std::min(64, std::max(0, tasks - 64 * i));
            banks[0].bits[i].store(n == 64 ? ~uint64_t(0) : (uint64_t(1) << n) - 1, std::memory_order_relaxed);
        }
    }
    template <class F, class Ready> void drain(const unsigned *order, F execute, Ready ready, int step_limit = 32) {
        while (final_claimed.load(std::memory_order_acquire) < tasks) {
            bool claimed = false;
            for (int k = 0; k < std::min(steps, step_limit) && !claimed; ++k)
                for (int h = 0; h < Words && !claimed; ++h) {
                    auto &bank = banks[k].bits[h];
                    uint64_t word = bank.load(std::memory_order_relaxed);
                    while (word) {
                        uint64_t eligible = word;
                        while (eligible && k == 0 && !ready(order[h * 64 + __builtin_ctzll(eligible)]))
                            eligible &= eligible - 1;
                        if (!eligible)
                            break;
                        const unsigned bit = __builtin_ctzll(eligible);
                        const uint64_t mask = uint64_t(1) << bit;
                        if (!bank.compare_exchange_weak(word, word & ~mask, std::memory_order_acquire,
                                                        std::memory_order_relaxed))
                            continue;
                        if (k == steps - 1)
                            final_claimed.fetch_add(1, std::memory_order_release);
                        execute(order[h * 64 + bit], k);
                        if (k < steps - 1)
                            banks[k + 1].bits[h].fetch_or(mask, std::memory_order_release);
                        claimed = true;
                        break;
                    }
                }
            if (!claimed)
                asm volatile("yield" ::: "memory");
        }
    }
    template <class F, class Ready> void drain_terminal(const unsigned *order, F execute, Ready ready, int step_limit = 32) {
        while (final_claimed.load(std::memory_order_acquire) < tasks) {
            bool claimed = false;
            for (int k = 0; k < std::min(steps, step_limit) && !claimed; ++k)
                for (int h = 0; h < Words && !claimed; ++h) {
                    auto &bank = banks[k].bits[h];
                    uint64_t word = bank.load(std::memory_order_relaxed);
                    while (word) {
                        uint64_t eligible = word;
                        while (eligible && !ready(order[h * 64 + __builtin_ctzll(eligible)],k))
                            eligible &= eligible - 1;
                        if (!eligible)
                            break;
                        const unsigned bit = __builtin_ctzll(eligible);
                        const uint64_t mask = uint64_t(1) << bit;
                        if (!bank.compare_exchange_weak(word, word & ~mask, std::memory_order_acquire,
                                                        std::memory_order_relaxed))
                            continue;
                        if (k == steps - 1)
                            final_claimed.fetch_add(1, std::memory_order_release);
                        execute(order[h * 64 + bit], k);
                        if (k < steps - 1)
                            banks[k + 1].bits[h].fetch_or(mask, std::memory_order_release);
                        claimed = true;
                        break;
                    }
                }
            if (!claimed)
                asm volatile("yield" ::: "memory");
        }
    }
};

struct MergedAcCut {
    int col, first, rows;
};
static constexpr MergedAcCut merged_ac_cuts[36][3] = {
    {{0, 0, 2000}, {4608, 0, 672}, {6912, 0, 800}},          {{128, 0, 2000}, {4736, 0, 976}, {6912, 800, 496}},
    {{256, 0, 2000}, {4864, 0, 1216}, {6912, 1296, 256}},    {{384, 0, 2000}, {4992, 0, 1024}, {6912, 1552, 448}},
    {{512, 0, 2000}, {5120, 0, 896}, {7040, 0, 576}},        {{640, 0, 2000}, {5248, 0, 976}, {7040, 576, 496}},
    {{768, 0, 2000}, {5376, 0, 1008}, {7040, 1072, 464}},    {{896, 0, 2000}, {5504, 0, 1008}, {7040, 1536, 464}},
    {{1024, 0, 2000}, {5632, 0, 1072}, {7168, 0, 400}},      {{1152, 0, 2000}, {5760, 0, 1024}, {7168, 400, 448}},
    {{1280, 0, 2000}, {5888, 0, 816}, {7168, 848, 656}},     {{1408, 0, 2000}, {6016, 0, 976}, {7168, 1504, 496}},
    {{1536, 0, 2000}, {6144, 0, 1296}, {7296, 0, 176}},      {{1664, 0, 2000}, {6272, 0, 1024}, {7296, 176, 448}},
    {{1792, 0, 2000}, {6400, 0, 784}, {7296, 624, 688}},     {{1920, 0, 2000}, {6528, 0, 784}, {7296, 1312, 688}},
    {{2048, 0, 2000}, {6656, 0, 768}, {7424, 0, 704}},       {{2176, 0, 2000}, {6784, 0, 768}, {7424, 704, 704}},
    {{2304, 0, 2000}, {4608, 672, 1328}, {7424, 1408, 144}}, {{2432, 0, 2000}, {4736, 976, 1024}, {7424, 1552, 448}},
    {{2560, 0, 2000}, {4864, 1216, 784}, {7552, 0, 688}},    {{2688, 0, 2000}, {4992, 1024, 976}, {7552, 688, 496}},
    {{2816, 0, 2000}, {5120, 896, 1104}, {7552, 1184, 368}}, {{2944, 0, 2000}, {5248, 976, 1024}, {7552, 1552, 448}},
    {{3072, 0, 2000}, {5376, 1008, 992}, {7680, 0, 480}},    {{3200, 0, 2000}, {5504, 1008, 992}, {7680, 480, 480}},
    {{3328, 0, 2000}, {5632, 1072, 928}, {7680, 960, 544}},  {{3456, 0, 2000}, {5760, 1024, 976}, {7680, 1504, 496}},
    {{3584, 0, 2000}, {5888, 816, 1184}, {7808, 0, 288}},    {{3712, 0, 2000}, {6016, 976, 1024}, {7808, 288, 448}},
    {{3840, 0, 2000}, {6144, 1296, 704}, {7808, 736, 768}},  {{3968, 0, 2000}, {6272, 1024, 976}, {7808, 1504, 496}},
    {{4096, 0, 2000}, {6400, 784, 1216}, {7936, 0, 528}},    {{4224, 0, 2000}, {6528, 784, 1216}, {7936, 528, 512}},
    {{4352, 0, 2000}, {6656, 768, 1232}, {7936, 1040, 480}}, {{4480, 0, 2000}, {6784, 768, 1232}, {7936, 1520, 480}},
};

namespace tcopt {

// shared: p received from donor ranks as [slot][s][t][4], local p = L.own + 4*slot + i

constexpr int tc1_apack_chunks = 63;
static int tc1_apack_tasks() { return 4 * tc1_apack_chunks + (tc1_layout.p - tc1_layout.own) / 4; }
static void pack_tc1_rows_task(const double *a, const double *shared, double *out, int task) {
    constexpr int m = 500, k = 4000;
    const Tc1Rows L = tc1_layout;
    const svbool_t pg = svptrue_b64(), p4 = svwhilelt_b64(0, 4);
    if (task < 4 * tc1_apack_chunks) {
        const int s = task / tc1_apack_chunks, t0 = task % tc1_apack_chunks * 64, t1 = std::min(k, t0 + 64);
        const int l = t0 / 320 * 320, kc = std::min(320, k - l);
        double *slab = out + size_t(l) * L.rows;
        double *own_blocks = slab + size_t(s) * L.pm * kc; // block b of s starts at (s*pm/16+b)*16*kc
        double *tail =
            slab + size_t(L.tail_row) * kc + s * 4; // own tail block j holds p=pm+4j.. of s at columns 4s..4s+3
        for (int t = t0; t < t1; ++t) {
            const double *src = a + (size_t(s) * k + t) * m;
            const size_t row = size_t(t - l) * 16;
            for (int b = 0; b < L.pm / 16; ++b) {
                double *d = own_blocks + size_t(b) * 16 * kc + row;
                svst1_vnum(pg, d, 0, svld1_vnum(pg, src + 16 * b, 0));
                svst1_vnum(pg, d, 1, svld1_vnum(pg, src + 16 * b, 1));
            }
            for (int p = L.pm; p < L.own; p += 4)
                svst1(p4, tail + size_t(p - L.pm) / 4 * 16 * kc + row, svld1(p4, src + p));
        }
        return;
    }
    const int slot = task - 4 * tc1_apack_chunks, block = L.tail_row / 16 + (L.own - L.pm) / 4 + slot;
    for (int s = 0; s < 4; ++s)
        for (int t = 0; t < k; ++t) {
            const int l = t / 320 * 320, kc = std::min(320, k - l);
            svst1(p4, out + size_t(l) * L.rows + size_t(block) * 16 * kc + size_t(t - l) * 16 + s * 4,
                  svld1(p4, shared + size_t(slot) * 64000 + (size_t(s) * k + t) * 4));
        }
}

} // namespace tcopt

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

struct chn_ioe_info {
    u32 ch_err_status;
    u32 ch_cqe_sqeid;
    u32 ch_cqe_status;
};

struct hisi_sdma_chn_num {
    u32 total_chn_num;
    u32 share_chn_num;
};

struct hisi_sdma_umem_info {
    uintptr_t vma;
    u32 size;
    u64 cookie;
};

struct hisi_sdma_sq_entry {
    __le32 opcode : 8;
    __le32 sssv : 1;
    __le32 dssv : 1;
    __le32 sns : 1;
    __le32 dns : 1;
    __le32 sro : 1;
    __le32 dro : 1;
    __le32 stride : 2;
    __le32 ie : 1;
    __le32 comp_en : 1;
    __le32 reserved0 : 14;

    __le32 sqe_id : 16;
    __le32 mpam_partid : 8;
    __le32 mpamns : 1;
    __le32 pmg : 2;
    __le32 qos : 4;
    __le32 reserved1 : 1;

    __le32 src_streamid : 16;
    __le32 src_substreamid : 16;
    __le32 dst_streamid : 16;
    __le32 dst_substreamid : 16;

    __le32 src_addr_l : 32;
    __le32 src_addr_h : 32;
    __le32 dst_addr_l : 32;
    __le32 dst_addr_h : 32;

    __le32 length_move : 32;

    __le32 src_stride_len : 32;
    __le32 dst_stride_len : 32;
    __le32 stride_num : 32;
    __le32 reserved2 : 32;
    __le32 reserved3 : 32;
    __le32 reserved4 : 32;
    __le32 reserved5 : 32;
};

struct hisi_sdma_queue_info {
    u16 sq_head;
    u16 sq_tail;
    u16 cq_head;
    u16 cq_tail;
    u32 cq_vld;
    int lock;
    u32 lock_pid;
    int err_cnt;
    int cqe_err[(1U << 10)];
    u32 round_cnt[(1U << 10)];
    struct chn_ioe_info ioe;
    bool is_ncob_enable;
    bool is_numa;
};

struct hisi_sdma_share_chn {
    u16 chn_idx;
    bool init_flag;
};

struct hisi_sdma_reg_info {
    u32 chn;
    int type;
    u32 reg_value;
};

struct hisi_sdma_pid_info {
    u32 num;
    uintptr_t pid_list_addr;
};

typedef void (*sdma_task_callback)(int task_status, void *task_data);

struct hisi_sdma_sqe_task {
    u64 src_addr;
    u64 dst_addr;
    u32 src_process_id;
    u32 dst_process_id;
    u32 src_stride_len;
    u32 dst_stride_len;
    u32 stride_num;
    u32 length;
    u8 opcode;
    u8 mpam_partid;
    u8 pmg : 2;
    u8 resvd1 : 6;
    u8 qos : 4;
    u8 resvd2 : 4;
    sdma_task_callback task_cb;
    void *task_data;
    struct sdma_sqe_task *next_sqe;
};

struct hisi_sdma_task_info {
    u32 chn;
    u32 req_cnt;
    u32 task_cnt;
    uintptr_t task_addr;
};
inline double tc_seconds() {
    timespec t;
    clock_gettime(4, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
inline void sdma_check(bool good, const char *operation) {
    if (!good)
        throw std::runtime_error(std::string(operation) + ": " + strerror((*__errno_location())));
}

struct Sdma {
    int fd = -1;
    bool safe = true, shared = false, locked = false;
    u32 channel = (4294967295U), process = 0, stream = 0;
    hisi_sdma_chn_num channels{};
    hisi_sdma_sq_entry *sq = nullptr;
    volatile u32 *cq = nullptr;
    volatile u32 *regs = nullptr;
    volatile hisi_sdma_queue_info *sync = nullptr;
    u32 tail = 0, head = 0, valid = 1, outstanding = 0;
    size_t sync_size = 0;
    std::vector<u64> cookies;
    std::vector<u16> expected;
    size_t expected_head = 0;
    Sdma(const Sdma &) = delete;
    Sdma &operator=(const Sdma &) = delete;
    u32 reg(unsigned long command, u32 offset) {
        if (regs)
            return regs[offset / 4];
        hisi_sdma_reg_info info{channel, 1, 0};
        sdma_check(ioctl(fd, command, &info) == 0, "read channel register");
        return info.reg_value;
    }
    void *map(size_t bytes, u32 kind) {
        void *p = mmap(nullptr, bytes, 0x1 | 0x2, 0x01, fd,
                       off_t(channel + kind * channels.total_chn_num) * sysconf(_SC_PAGESIZE));
        sdma_check(p != ((void *)-1), "mmap SDMA queue");
        return p;
    }
    explicit Sdma(int device, int preferred_channel = -1, int channel_stride = 1) {
        char path[32];
        snprintf(path, sizeof(path), "/dev/sdma%d", device);
        fd = open(path, 02 | 02000000);
        sdma_check(fd >= 0, "open SDMA");
        sdma_check(ioctl(fd,
                         (((2U) << (((0 + 8) + 8) + 14)) | ((('s')) << (0 + 8)) | (((20)) << 0) |
                          ((((sizeof(bool)))) << ((0 + 8) + 8))),
                         &safe) == 0,
                   "get mode");
        sdma_check(ioctl(fd,
                         (((2U) << (((0 + 8) + 8) + 14)) | ((('s')) << (0 + 8)) | (((9)) << 0) |
                          ((((sizeof(struct hisi_sdma_chn_num)))) << ((0 + 8) + 8))),
                         &channels) == 0,
                   "get channel count");

        const bool all_shared = channels.total_chn_num > 0 && channels.share_chn_num == channels.total_chn_num;
        if (all_shared || ioctl(fd,
                                (((2U) << (((0 + 8) + 8) + 14)) | ((('s')) << (0 + 8)) | (((2)) << 0) |
                                 ((((sizeof(u32)))) << ((0 + 8) + 8))),
                                &channel) != 0) {
            sdma_check((all_shared || (*__errno_location()) == 28) && channels.share_chn_num > 0, "allocate channel");
            sdma_check(channel_stride > 0 && channels.share_chn_num % channel_stride == 0,
                       "shared channel partition stride");
            shared = true;
            channel = (preferred_channel >= 0 ? u32(preferred_channel) : u32(getpid())) % channels.share_chn_num;
            hisi_sdma_share_chn claim{u16(channel), true};
            sdma_check(ioctl(fd,
                             (((1U) << (((0 + 8) + 8) + 14)) | ((('s')) << (0 + 8)) | (((11)) << 0) |
                              ((((sizeof(struct hisi_sdma_share_chn)))) << ((0 + 8) + 8))),
                             &claim) == 0,
                       "claim shared channel");
        }
        sdma_check(ioctl(fd,
                         (((2U) << (((0 + 8) + 8) + 14)) | ((('s')) << (0 + 8)) | (((1)) << 0) |
                          ((((sizeof(u32)))) << ((0 + 8) + 8))),
                         &process) == 0,
                   "get process ID");
        sdma_check(ioctl(fd,
                         (((2U) << (((0 + 8) + 8) + 14)) | ((('s')) << (0 + 8)) | (((4)) << 0) |
                          ((((sizeof(u32)))) << ((0 + 8) + 8))),
                         &stream) == 0,
                   "get stream ID");
        sync_size = (sizeof(hisi_sdma_queue_info) + 4095) & ~size_t(4095);

        bool accepted = false;
        for (u32 attempt = 0; attempt < (shared ? channels.share_chn_num / channel_stride : 1); ++attempt) {
            sync = static_cast<volatile hisi_sdma_queue_info *>(map(sync_size, 3));
            cq = static_cast<volatile u32 *>(map(16384, 1));
            if (!safe) {
                sq = static_cast<hisi_sdma_sq_entry *>(map(65536, 0));
                regs = static_cast<volatile u32 *>(map(4096, 2));
            }
            if (try_acquire()) {
                tail = reg((((2U | 1U) << (((0 + 8) + 8) + 14)) | ((('s')) << (0 + 8)) | (((15)) << 0) |
                            ((((sizeof(struct hisi_sdma_reg_info)))) << ((0 + 8) + 8))),
                           0x4c);
                head = reg((((2U | 1U) << (((0 + 8) + 8) + 14)) | ((('s')) << (0 + 8)) | (((16)) << 0) |
                            ((((sizeof(struct hisi_sdma_reg_info)))) << ((0 + 8) + 8))),
                           0x90);
                valid = sync->cq_vld;
                bool bounds = tail < 1024 && head < 1024 && valid < 2;
                bool idle = bounds && head == sync->cq_head && tail == sync->sq_tail &&
                            reg((((2U | 1U) << (((0 + 8) + 8) + 14)) | ((('s')) << (0 + 8)) | (((14)) << 0) |
                                 ((((sizeof(struct hisi_sdma_reg_info)))) << ((0 + 8) + 8))),
                                0x50) == tail;
                bool empty = idle && ((cq[head * 4 + 3] >> 16) & 1) != valid;
                if (empty) {
                    accepted = true;
                    break;
                }
                fprintf(stderr,
                        "SDMA reject inconsistent/nonempty channel=%u sq_tail=%u sync_sq_tail=%u cq_head=%u "
                        "sync_cq_head=%u phase=%u\n",
                        channel, tail, u32(sync->sq_tail), head, u32(sync->cq_head), valid);
                release();
            }
            if (sq) {
                munmap(sq, 65536);
                sq = nullptr;
            }
            if (regs) {
                munmap(const_cast<u32 *>(regs), 4096);
                regs = nullptr;
            }
            munmap(const_cast<u32 *>(cq), 16384);
            cq = nullptr;
            munmap(const_cast<hisi_sdma_queue_info *>(sync), sync_size);
            sync = nullptr;
            if (!shared)
                break;
            hisi_sdma_share_chn old_claim{u16(channel), false};
            sdma_check(ioctl(fd,
                             (((1U) << (((0 + 8) + 8) + 14)) | ((('s')) << (0 + 8)) | (((11)) << 0) |
                              ((((sizeof(struct hisi_sdma_share_chn)))) << ((0 + 8) + 8))),
                             &old_claim) == 0,
                       "release rejected shared channel");
            channel = (channel + channel_stride) % channels.share_chn_num;
            hisi_sdma_share_chn next_claim{u16(channel), true};
            sdma_check(ioctl(fd,
                             (((1U) << (((0 + 8) + 8) + 14)) | ((('s')) << (0 + 8)) | (((11)) << 0) |
                              ((((sizeof(struct hisi_sdma_share_chn)))) << ((0 + 8) + 8))),
                             &next_claim) == 0,
                       "claim next shared channel");
        }
        sdma_check(accepted, "no idle empty SDMA channel");
        expected.reserve(1024);

        release();
    }
    bool try_acquire() {
        if (!shared || locked)
            return true;
        int unlocked = 0;

        if (!__atomic_compare_exchange_n(&sync->lock, &unlocked, 1, false, 2, 0))
            return false;

        locked = true;
        sync->lock_pid = getpid();

        tail = reg((((2U | 1U) << (((0 + 8) + 8) + 14)) | ((('s')) << (0 + 8)) | (((15)) << 0) |
                    ((((sizeof(struct hisi_sdma_reg_info)))) << ((0 + 8) + 8))),
                   0x4c);

        head = sync->cq_head;
        valid = sync->cq_vld;

        return true;
    }
    void acquire() {

        double deadline = tc_seconds() + 10;

        while (!try_acquire())
            if (tc_seconds() > deadline)
                throw std::runtime_error("shared SDMA lock timeout");
    }
    void release() {
        if (locked) {
            sync->lock_pid = 0;
            __atomic_store_n(&sync->lock, 0, 3);
            locked = false;
        }
    }
    void pin(void *p, size_t bytes) {
        sdma_check(bytes > 0 && bytes < (4294967295U), "pin size");
        hisi_sdma_umem_info info{reinterpret_cast<uintptr_t>(p), u32(bytes), 0};
        sdma_check(ioctl(fd,
                         (((2U | 1U) << (((0 + 8) + 8) + 14)) | ((('s')) << (0 + 8)) | (((5)) << 0) |
                          ((((sizeof(struct hisi_sdma_umem_info)))) << ((0 + 8) + 8))),
                         &info) == 0,
                   "pin buffer");
        cookies.push_back(info.cookie);
    }

    void authorize(const std::vector<u32> &pids) {

        if (!safe) {
            for (u32 id : pids)
                sdma_check(id <= 65535, "fast peer PASID width");
            return;
        }
        hisi_sdma_pid_info info{u32(pids.size()), reinterpret_cast<uintptr_t>(pids.data())};
        sdma_check(ioctl(fd,
                         (((1U) << (((0 + 8) + 8) + 14)) | ((('s')) << (0 + 8)) | (((12)) << 0) |
                          ((((sizeof(struct hisi_sdma_pid_info)))) << ((0 + 8) + 8))),
                         &info) == 0,
                   "authorize peer processes");
    }

    void submit_fast_tasks(const hisi_sdma_sqe_task *tasks, u32 count) {
        sdma_check(!safe && sq && regs && count && count <= 64 && outstanding + count < 1000,
                   "fast generic batch bounds");
        for (u32 i = 0; i < count; ++i) {
            const auto &t = tasks[i];
            sdma_check(t.length && t.opcode == 0 && t.src_process_id <= 65535 && t.dst_process_id <= 65535 &&
                           (t.src_process_id == process || t.dst_process_id == process) && !t.task_cb,
                       "fast copy descriptor");
        }

        if (!outstanding)
            acquire();

        for (u32 i = 0; i < count; ++i) {
            const auto &t = tasks[i];
            hisi_sdma_sq_entry e{};
            e.sssv = e.dssv = e.sns = e.dns = e.comp_en = e.mpamns = 1;
            e.src_streamid = e.dst_streamid = stream;
            e.src_substreamid = t.src_process_id;
            e.dst_substreamid = t.dst_process_id;
            e.src_addr_l = u32(t.src_addr);
            e.src_addr_h = u32(t.src_addr >> 32);
            e.dst_addr_l = u32(t.dst_addr);
            e.dst_addr_h = u32(t.dst_addr >> 32);
            e.length_move = t.length;
            e.stride = t.stride_num ? 1 : 0;
            e.stride_num = t.stride_num;
            e.src_stride_len = t.src_stride_len;
            e.dst_stride_len = t.dst_stride_len;
            e.mpam_partid = t.mpam_partid;
            e.pmg = t.pmg;
            e.qos = t.qos;
            e.sqe_id = tail;
            expected.push_back(tail);
            sq[tail] = e;
            tail = (tail + 1) & 1023;
        }
        asm volatile("dmb oshst" ::: "memory");
        regs[0x4c / 4] = tail;
        sync->sq_tail = tail;
        outstanding += count;
    }
    void submit_tasks(const hisi_sdma_sqe_task *tasks, u32 count) {
        if (!safe) {
            submit_fast_tasks(tasks, count);
            return;
        }
        sdma_check(count && count <= 64 && outstanding + count < 1000, "safe generic batch bounds");
        for (u32 i = 0; i < count; ++i)
            sdma_check(tasks[i].length && !tasks[i].task_cb, "safe copy batch descriptor");
        if (!outstanding)
            acquire();
        hisi_sdma_task_info info{channel, count, count, reinterpret_cast<uintptr_t>(tasks)};
        asm volatile("dmb oshst" ::: "memory");
        sdma_check(ioctl(fd,
                         (((2U | 1U) << (((0 + 8) + 8) + 14)) | ((('s')) << (0 + 8)) | (((13)) << 0) |
                          ((((sizeof(struct hisi_sdma_task_info)))) << ((0 + 8) + 8))),
                         &info) == 0,
                   "submit generic batch");
        for (u32 i = 0; i < count; ++i) {
            expected.push_back(tail);
            tail = (tail + 1) & 1023;
        }
        sync->sq_tail = tail;
        outstanding += count;
    }
    void submit_task(hisi_sdma_sqe_task task) {
        if (!safe) {
            submit_fast_tasks(&task, 1);
            return;
        }
        sdma_check(safe && task.length && outstanding < 1000, "generic task bounds");
        if (!outstanding)
            acquire();
        hisi_sdma_task_info info{channel, 1, 1, reinterpret_cast<uintptr_t>(&task)};
        asm volatile("dmb oshst" ::: "memory");
        sdma_check(ioctl(fd,
                         (((2U | 1U) << (((0 + 8) + 8) + 14)) | ((('s')) << (0 + 8)) | (((13)) << 0) |
                          ((((sizeof(struct hisi_sdma_task_info)))) << ((0 + 8) + 8))),
                         &info) == 0,
                   "submit generic task");
        expected.push_back(tail);
        tail = (tail + 1) & 1023;
        sync->sq_tail = tail;
        ++outstanding;
    }

    // Drain only CQEs already visible. A pending batch retains its queue lease;
    // callers must use a different Sdma object for independent C transfers.
    bool poll() {
        if(!outstanding)return true;
        while(outstanding){
            const u32 state=cq[head*4+3];
            if(((state>>16)&1)!=valid)return false;
            asm volatile("dmb oshld":::"memory");
            const u32 id=(cq[head*4+2]>>16)&65535;
            if((state>>17)!=0 || id!=expected[expected_head])
                throw std::runtime_error("SDMA poll completion status or ID mismatch");
            sync->cqe_err[id]=0;++sync->round_cnt[id];
            ++expected_head;--outstanding;
            if(++head==1024){head=0;valid^=1;}
        }
        asm volatile("dmb osh":::"memory");
        if(regs)regs[0x90/4]=head;
        else {hisi_sdma_reg_info info{channel,2,head};sdma_check(ioctl(fd,_IOWR('s',16,struct hisi_sdma_reg_info),&info)==0,"acknowledge polled CQ");}
        sync->cq_head=head;sync->cq_vld=valid;sync->sq_head=tail;sync->cq_tail=head;
        expected.clear();expected_head=0;release();return true;
    }
    void wait() {
        double deadline = tc_seconds() + 10;
        while (outstanding) {
            u32 state = cq[head * 4 + 3];
            if (((state >> 16) & 1) != valid) {
                if (tc_seconds() > deadline)
                    throw std::runtime_error("SDMA completion timeout");
                continue;
            }
            asm volatile("dmb oshld" ::: "memory");
            u32 id = (cq[head * 4 + 2] >> 16) & 65535;
            if ((state >> 17) != 0 || id != expected[expected_head]) {
                fprintf(stderr, "bad completion state=%08x id=%u expected=%u\n", state, id, expected[expected_head]);
                throw std::runtime_error("SDMA completion status or ID mismatch");
            }
            sync->cqe_err[id] = 0;
            ++sync->round_cnt[id];
            ++expected_head;
            --outstanding;
            if (++head == 1024) {
                head = 0;
                valid ^= 1;
            }
        }
        asm volatile("dmb osh" ::: "memory");

        if (regs)
            regs[0x90 / 4] = head;
        else {
            hisi_sdma_reg_info info{channel, 2, head};
            sdma_check(ioctl(fd,
                             (((2U | 1U) << (((0 + 8) + 8) + 14)) | ((('s')) << (0 + 8)) | (((16)) << 0) |
                              ((((sizeof(struct hisi_sdma_reg_info)))) << ((0 + 8) + 8))),
                             &info) == 0,
                       "acknowledge CQ");
        }

        sync->cq_head = head;
        sync->cq_vld = valid;
        sync->sq_head = tail;
        sync->cq_tail = head;
        expected.clear();
        expected_head = 0;

        release();
    }
    ~Sdma() {

        if (outstanding) {
            fprintf(stderr, "fatal: outstanding SDMA at teardown\n");
            std::_Exit(3);
        }
        for (u64 cookie : cookies)
            ioctl(fd,
                  (((1U) << (((0 + 8) + 8) + 14)) | ((('s')) << (0 + 8)) | (((6)) << 0) |
                   ((((sizeof(u64)))) << ((0 + 8) + 8))),
                  &cookie);
        if (sq)
            munmap(sq, 65536);
        if (cq)
            munmap(const_cast<u32 *>(cq), 16384);
        if (regs)
            munmap(const_cast<u32 *>(regs), 4096);
        if (sync)
            munmap(const_cast<hisi_sdma_queue_info *>(sync), sync_size);
        if (shared) {
            hisi_sdma_share_chn claim{u16(channel), false};
            ioctl(fd,
                  (((1U) << (((0 + 8) + 8) + 14)) | ((('s')) << (0 + 8)) | (((11)) << 0) |
                   ((((sizeof(struct hisi_sdma_share_chn)))) << ((0 + 8) + 8))),
                  &claim);
        } else if (channel != (4294967295U))
            ioctl(fd,
                  (((1U) << (((0 + 8) + 8) + 14)) | ((('s')) << (0 + 8)) | (((3)) << 0) |
                   ((((sizeof(u32)))) << ((0 + 8) + 8))),
                  &channel);
        if (fd >= 0)
            close(fd);
    }
};

__attribute__((noinline, aligned(64))) static void sme_pack_transpose32(long k, const double *source, long lda,
                                                                        double *destination) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x19, %[source]\n"
        "mov x13, %[destination]\n"
        "lsl x17, %[lda], #3\n"
        "lsr x20, %[k], #3\n"
        "smstart\n"
        "ptrue p0.d\n"
        "zero {za}\n"
        "1:\n"
        "mov x9, x19\n"
        "add x10, x9, x17, lsl #3\n"
        "add x11, x10, x17, lsl #3\n"
        "add x14, x11, x17, lsl #3\n"
        "mov w12, #0\n"
        "2:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x10]\n"
        "ld1d {z2.d}, p0/z, [x11]\n"
        "ld1d {z3.d}, p0/z, [x14]\n"
        "mova za0h.d[w12, 0], p0/m, z0.d\n"
        "mova za1h.d[w12, 0], p0/m, z1.d\n"
        "mova za2h.d[w12, 0], p0/m, z2.d\n"
        "mova za3h.d[w12, 0], p0/m, z3.d\n"
        "add x9, x9, x17\n"
        "add x10, x10, x17\n"
        "add x11, x11, x17\n"
        "add x14, x14, x17\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 2b\n"
        "mov w12, #0\n"
        "3:\n"
        "mova z0.d, p0/m, za0v.d[w12, 0]\n"
        "mova z1.d, p0/m, za1v.d[w12, 0]\n"
        "mova z2.d, p0/m, za2v.d[w12, 0]\n"
        "mova z3.d, p0/m, za3v.d[w12, 0]\n"
        "st1d {z0.d}, p0, [x13]\n"
        "st1d {z1.d}, p0, [x13, #1, mul vl]\n"
        "st1d {z2.d}, p0, [x13, #2, mul vl]\n"
        "st1d {z3.d}, p0, [x13, #3, mul vl]\n"
        "add x13, x13, #256\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 3b\n"
        "add x19, x19, #64\n"
        "subs x20, x20, #1\n"
        "b.ne 1b\n"
        "smstop\n"
        :
        :[source]"r"(source),[destination]"r"(destination),[lda]"r"(lda),[k]"r"(k)
        :"memory","cc","x9","x10","x11","x12","x13","x14","x17","x19","x20",
         "z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31",
         "p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}

__attribute__((noinline, aligned(64))) static void pack_b_rows_fused(long rows, const double *src, double *dst,
                                                                     long panel_stride_bytes) {
    if (!rows)
        return;
    asm volatile(
  ".arch_extension sme\nmov x9,%[src]\nmov x10,%[dst]\nmov x11,%[rows]\nmov x14,%[stride]\n"
  "smstart\nptrue p0.d\nptrue p1.d,vl4\ndup z3.d,#0\n"
  "1:\nmov x12,x10\nmov x13,#15\n"
  "2:\n"
  "ld1d {z0.d},p0/z,[x9]\nst1d {z0.d},p0,[x12]\n"
  "ld1d {z1.d},p0/z,[x9,#1,mul vl]\nst1d {z1.d},p0,[x12,#1,mul vl]\n"
  "ld1d {z2.d},p0/z,[x9,#2,mul vl]\nst1d {z2.d},p0,[x12,#2,mul vl]\n"
  "ld1d {z4.d},p0/z,[x9,#3,mul vl]\nst1d {z4.d},p0,[x12,#3,mul vl]\n"
  "add x9,x9,#256\nadd x12,x12,x14\nsubs x13,x13,#1\nb.ne 2b\n"
  "ld1d {z0.d},p0/z,[x9]\nst1d {z0.d},p0,[x12]\n"
  "ld1d {z1.d},p0/z,[x9,#1,mul vl]\nst1d {z1.d},p0,[x12,#1,mul vl]\n"
  "ld1d {z2.d},p1/z,[x9,#2,mul vl]\nst1d {z2.d},p0,[x12,#2,mul vl]\n"
  "st1d {z3.d},p0,[x12,#3,mul vl]\n"
  "add x9,x9,#160\nadd x10,x10,#256\nsubs x11,x11,#1\nb.ne 1b\nsmstop\n"
  ::[src]"r"(src),[dst]"r"(dst),[rows]"r"(rows),[stride]"r"(panel_stride_bytes)
  :"memory","cc","x9","x10","x11","x12","x13","x14",
   "z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15",
   "z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31",
   "p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}

namespace tcopt {

struct CommPool {
    static constexpr size_t large = 512ul << 20, small = 64ul << 20;
    static constexpr size_t temp_bytes = 256ul << 20;
    double *c, *b_source, *b_full, *temp;

    Sdma *dma;
    Sdma *other_binding;

    static double *blank(size_t bytes) {
        void *p = mmap(nullptr, bytes, 0x1 | 0x2, 0x02 | 0x20, -1, 0);
        sdma_check(p != ((void *)-1), "allocate communication pool");
#pragma omp parallel for schedule(static)
        for (size_t offset = 0; offset < bytes; offset += 65536)
            memset((char *)p + offset, 0, std::min(size_t(65536), bytes - offset));
        return (double *)p;
    }
    CommPool() {
        c = blank(large);
        b_source = blank(small);
        b_full = blank(large);
        temp = blank(temp_bytes);
        const char *rank_text = getenv("OMPI_COMM_WORLD_RANK");
        sdma_check(rank_text != nullptr, "SDMA rank partition requires the MPI rank environment");
        int local_rank = atoi(rank_text);
        sdma_check(local_rank >= 0 && local_rank < 8, "SDMA rank partition expects eight local ranks");
        dma = new Sdma(2, 17 * local_rank, 8);
        other_binding = new Sdma(3, 17 * local_rank, 8);
        other_binding->pin(c, large);
        other_binding->pin(b_source, small);
        other_binding->pin(b_full, large);
        dma->pin(c, large);
        dma->pin(b_source, small);
        dma->pin(b_full, large);
    }
    ~CommPool() {

        delete other_binding;
        delete dma;
        munmap(c, large);
        munmap(b_source, small);
        munmap(b_full, large);
        munmap(temp, temp_bytes);
    }
};
static CommPool *comm_pool = nullptr;
__attribute__((constructor)) static void prepare_comm_pool() {
    try {
        comm_pool = new CommPool;

    } catch (const std::exception &e) {
        fprintf(stderr, "communication pool: %s\n", e.what());
        std::abort();
    }
}
__attribute__((destructor)) static void release_comm_pool() { delete comm_pool; }
static void parallel_copy(double *dst, const double *src, size_t bytes) {
#pragma omp parallel for schedule(static)
    for (size_t offset = 0; offset < bytes; offset += 65536)
        memcpy((char *)dst + offset, (const char *)src + offset, std::min(size_t(65536), bytes - offset));
};
static void pack_tc2_b_task(const double *input, int index) {
    constexpr int k = 4000, n = 500, batches = 2, padded_n = 512;
    const int batch = index / (padded_n / 32), jb = index % (padded_n / 32) * 32;
    double *target = comm_pool->b_source + size_t(jb) * batches * k + size_t(batch) * k * 32;
    const double *source = input + (size_t(batch) * n + jb) * k;
    if (jb + 32 <= n)
        sme_pack_transpose32(k, source, k, target);
    else
        for (int l = 0; l < k; ++l)
            for (int j = 0; j < 32; ++j)
                target[size_t(l) * 32 + j] = jb + j < n ? source[size_t(j) * k + l] : 0;
}

// Same 2 x 512 x 4000 B wire and gather, but each local Q256 pair contains
// X0=B00+B11, X1=B01-B11, X2=B10-B00, X3=B11 in its four original slots.
// One task transposes/formulates N32 x K320/160 for all four quadrants.
struct TbRhsPackDescriptor {
    const double* source[4];
    double* output[4];
    long k,right_rows;
};
static_assert(offsetof(TbRhsPackDescriptor,k)==64,"TB RHS descriptor");
static_assert(offsetof(TbRhsPackDescriptor,right_rows)==72,"TB RHS mask");
template<bool Tail>
__attribute__((noinline,aligned(64))) static void sme_tb_rhs_pack(const TbRhsPackDescriptor &d) {
    // Tail masks source ROWS before any load, including the final T1/Q500 edge.
    // The shared assembly body uses a compile-time branch outside the hot loop
    // through an assembler constant, so full panels have no per-row mask test.
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28,%[d]\nmov x23,#32000\nmov x25,#0\n"
        "ldr x27,[x28,#72]\nsmstart\nptrue p0.d\ndup z7.d,#0\n"
        "1:\n"
        "ldr x19,[x28]\nldr x20,[x28,#8]\nldr x21,[x28,#16]\nldr x22,[x28,#24]\n"
        "lsl x26,x25,#3\n"
        "madd x19,x26,x23,x19\nmadd x20,x26,x23,x20\n"
        "madd x21,x26,x23,x21\nmadd x22,x26,x23,x22\n"
        "ldr x14,[x28,#64]\nlsr x14,x14,#3\nmov x24,#0\n"
        "2:\nmov x9,x19\nmov x10,x20\nmov x11,x21\nmov x13,x22\nmov w12,#0\n"
        "3:\n"
        "ld1d {za0h.d[w12,0]},p0/z,[x9]\n"
        "ld1d {za2h.d[w12,0]},p0/z,[x11]\n"
        ".if %c[tail]\n"
        "add x26,x12,x25,lsl #3\ncmp x26,x27\nb.ge 5f\n"
        ".endif\n"
        "ld1d {za1h.d[w12,0]},p0/z,[x10]\n"
        "ld1d {za3h.d[w12,0]},p0/z,[x13]\n"
        ".if %c[tail]\nb 6f\n5:\n"
        "mova za1h.d[w12,0],p0/m,z7.d\n"
        "mova za3h.d[w12,0],p0/m,z7.d\n6:\n.endif\n"
        "add x9,x9,x23\nadd x10,x10,x23\nadd x11,x11,x23\nadd x13,x13,x23\n"
        "add w12,w12,#1\ncmp w12,#8\nb.ne 3b\n"
        "add x26,x24,x25,lsl #6\n"
        "ldr x9,[x28,#32]\nldr x10,[x28,#40]\nldr x11,[x28,#48]\nldr x13,[x28,#56]\n"
        "add x9,x9,x26\nadd x10,x10,x26\nadd x11,x11,x26\nadd x13,x13,x26\n"
        "mov w12,#0\n4:\n"
        "mova z0.d,p0/m,za0v.d[w12,0]\nmova z1.d,p0/m,za1v.d[w12,0]\n"
        "mova z2.d,p0/m,za2v.d[w12,0]\nmova z3.d,p0/m,za3v.d[w12,0]\n"
        "fadd z4.d,z0.d,z3.d\nst1d {z4.d},p0,[x9]\n"
        "fsub z4.d,z1.d,z3.d\nst1d {z4.d},p0,[x10]\n"
        "fsub z4.d,z2.d,z0.d\nst1d {z4.d},p0,[x11]\n"
        "st1d {z3.d},p0,[x13]\n"
        "add x9,x9,#256\nadd x10,x10,#256\nadd x11,x11,#256\nadd x13,x13,#256\n"
        "add w12,w12,#1\ncmp w12,#8\nb.ne 4b\n"
        "add x19,x19,#64\nadd x20,x20,#64\nadd x21,x21,#64\nadd x22,x22,#64\n"
        "add x24,x24,#2048\nsubs x14,x14,#1\nb.ne 2b\n"
        "add x25,x25,#1\ncmp x25,#4\nb.ne 1b\nsmstop\n"
        ::[d]"r"(&d),[tail]"i"(Tail?1:0)
        :"memory","cc","x9","x10","x11","x12","x13","x14",
         "x19","x20","x21","x22","x23","x24","x25","x26","x27","x28",
         "z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15",
         "z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31",
         "p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}
static constexpr int tc2_tb_rhs_tasks=2*4*13;
static void pack_tc2_tb_rhs_task(const double* input,int index) {
    const int pair=index/52,n=(index%52)/13*32,step=index%13,u=step*320;
    const int k=std::min(320,4000-u),q=pair*256+n,right=std::min(32,500-q-128);
    TbRhsPackDescriptor d{{},{},k,right};
    for(int z=0;z<4;++z){
        const int qq=q+(z%2)*128,t=z/2;
        d.source[z]=input+(size_t(t)*500+qq)*4000+u;
        d.output[z]=comm_pool->b_source+size_t(qq)*8000+size_t(t*4000+u)*32;
    }
    if(right<32)sme_tb_rhs_pack<true>(d);else sme_tb_rhs_pack<false>(d);
}

constexpr size_t c_ready_chunk = 16ul << 20;
static unsigned long c_epoch = 1;

// pinned C pool addresses of all ranks, once known
static std::array<uintptr_t, 8> tree_pools{};
static bool tree_pools_known = false;

struct CReadyTask {
    int jb, l0, kc;
    size_t begin, end;
};
struct alignas(64) CReadyCounter {
    std::atomic<int> left{0};
};

static void c_tree_ready_transfer(double *dst, size_t bytes, size_t metadata, Sdma &dma, const std::vector<u32> &pids,
                                  uintptr_t root, uintptr_t relay, uintptr_t relay1, int rank, MPI_Comm comm,
                                  void (*progress)(void*)=nullptr, void* context=nullptr) {
    const size_t chunks = (bytes + c_ready_chunk - 1) / c_ready_chunk;
    if (rank == 0)
        std::abort();
    const int owner = rank == 2 || rank == 3 ? 1 : rank == 7 ? 6 : rank >= 5 ? 4 : 0;
    if ((owner == 6 || owner == 2) && !tree_pools_known)
        std::abort();
    const uintptr_t source = owner == 1                 ? relay1
                             : owner == 4               ? relay
                             : owner == 6 || owner == 2 ? tree_pools[owner]
                                                        : root;
    auto *snapshot = (u64 *)((char *)dst + metadata + chunks * 64);
    for (size_t c = 0; c < chunks; ++c) {
        Sdma &active = dma;

        const double deadline = tc_seconds() + 30;
        for (;;) {
            if(progress)progress(context);
            hisi_sdma_sqe_task flag{};
            flag.src_addr = source + metadata + c * 64;
            flag.dst_addr = (uintptr_t)snapshot;
            flag.src_process_id = pids[owner];
            flag.dst_process_id = active.process;
            flag.length = 64;
            active.submit_task(flag);
            active.wait();

            const u64 ready = *(volatile u64 *)snapshot;
            if (ready == c_epoch) {
                break;
            }
            if (ready > c_epoch)
                MPI_Abort(comm, 5);
            if (tc_seconds() > deadline)
                MPI_Abort(comm, 5);
        }

        const size_t offset = c * c_ready_chunk;
        hisi_sdma_sqe_task task{};
        task.src_addr = source + offset;
        task.dst_addr = (uintptr_t)dst + offset;
        task.src_process_id = pids[owner];
        task.dst_process_id = active.process;
        task.length = std::min(c_ready_chunk, bytes - offset);
        active.submit_task(task);

        active.wait();
        {
            asm volatile("dmb oshst" ::: "memory");
            __atomic_store_n((u64 *)((char *)dst + metadata + c * 64), u64(c_epoch), 3);
        }
    }
}

class PersistentGather {
    Sdma &dma;
    MPI_Comm comm;
    bool finished = false;
    static Sdma& select_dma(MPI_Comm communicator){
        int rank;MPI_Comm_rank(communicator,&rank);
        sdma_check(comm_pool->dma->process==comm_pool->other_binding->process,"B/C queue process IDs");
        return *((224>>rank)&1?comm_pool->dma:comm_pool->other_binding);
    }

  public:
    template <class Buffer>
    PersistentGather(const std::vector<double> &local, Buffer &full, int width, MPI_Comm communicator)
        : dma(select_dma(communicator)), comm(communicator) {
        if (local.size() * 8 > CommPool::small || full.size() * 8 > CommPool::large ||
            full.data() != comm_pool->b_full || size_t(width) * sizeof(double) > CommPool::small)
            MPI_Abort(comm, 4);
        double *source = comm_pool->b_source;
        int size;
        MPI_Comm_size(comm, &size);
        std::vector<u32> pids(size);
        std::vector<unsigned long> addresses(size);
        unsigned long address = (uintptr_t)source;

        MPI_Allgather(&dma.process, 1, MPI_UNSIGNED, pids.data(), 1, MPI_UNSIGNED, comm);

        MPI_Allgather(&address, 1, MPI_UNSIGNED_LONG, addresses.data(), 1, MPI_UNSIGNED_LONG, comm);

        dma.authorize(pids);

        MPI_Barrier(comm);

        for (int step = 0; step < size; ++step) {

            const int peer = step;

            hisi_sdma_sqe_task task{};
            task.src_addr = addresses[peer];
            task.dst_addr = (uintptr_t)full.data() + size_t(peer) * width * sizeof(double);
            task.src_process_id = pids[peer];
            task.dst_process_id = dma.process;
            task.length = size_t(width) * sizeof(double);
            task.dst_stride_len = size_t(full.size() - width) * sizeof(double);
            task.stride_num = 1;
            dma.submit_task(task);
        }
    }
    // Only the main OpenMP worker polls the separate B queue.
    bool progress(){
        if(!finished && dma.poll())finished=true;
        return finished;
    }
    void finish() {
        if (!finished) {
            dma.wait();
            finished = true;
        }
    }
    ~PersistentGather() {
        if (!finished)
            std::_Exit(3);
        MPI_Barrier(comm);
    }
};
} // namespace tcopt

namespace tcopt {
struct WorkBuffer {
    double *pointer;
    size_t count;
    WorkBuffer(size_t n, bool is_b) : count(n) {
        if (n * sizeof(double) > (is_b ? CommPool::large : CommPool::temp_bytes))
            std::abort();
        pointer = is_b ? comm_pool->b_full : comm_pool->temp;
    }
    double *data() { return pointer; }
    size_t size() const { return count; }
};
;

struct alignas(64) AcWorker {
    unsigned ac = 0;
};
struct PrepAcTask {
    int owner = 0, panel = 0, first = 0, rows = 0;
};
struct AcWorkload {
    int task_count = 0;
    AcWorker workers[38]{};
    PrepAcTask tasks[162]{};
};
struct PrepAcQueue {
    struct alignas(64) Cursor {
        std::atomic<unsigned> next{0}, finished{0};
        unsigned end = 0;
    };
    Cursor cursors[38];
    unsigned total = 0;

    unsigned order[162]{};

    void prepare_longest(const PrepAcTask *tasks) {
        if (total != 100)
            std::abort();
        for (unsigned i = 0; i < total; ++i)
            order[i] = i;

        auto weight = [&](unsigned i) { return tasks[i].panel == 62 ? 1556 : tasks[i].rows; };
        std::sort(order, order + total, [&](unsigned a, unsigned b) {
            int x = weight(a), y = weight(b);
            return x != y ? x > y : a < b;
        });
    }

    void prepare_longest_tc1(const PrepAcTask *tasks) {
        if (total > 162)
            std::abort();
        for (unsigned i = 0; i < total; ++i)
            order[i] = i;

        std::sort(order, order + total, [&](unsigned a, unsigned b) {
            return tasks[a].rows != tasks[b].rows ? tasks[a].rows > tasks[b].rows : a < b;
        });
    }

    void set(int owner, unsigned begin, unsigned end) {
        cursors[owner].next.store(begin, std::memory_order_relaxed);
        cursors[owner].end = end;
        total += end - begin;
    }
    bool claim(int owner, unsigned &task) {
        auto &c = cursors[owner];
        unsigned n = c.next.load(std::memory_order_relaxed);
        while (n < c.end) {
            if (c.next.compare_exchange_weak(n, n + 1, std::memory_order_relaxed)) {
                task = n;
                return true;
            }
        }
        return false;
    }
    template <class F, class Publish> void drain_with_publish(int worker, F task, Publish publish) {
        unsigned id, completed = 0;
        auto run = [&](int owner) {
            if (!claim(owner, id))
                return false;
            task(owner, id);
            ++completed;
            return true;
        };
        while (run(worker)) {
        }
        bool progress;
        do {
            progress = false;
            for (int d = 1; d < 38; ++d)
                if (run((worker + d) % 38))
                    progress = true;
        } while (progress);

        publish();
        cursors[worker].finished.store(completed, std::memory_order_release);
    }
    template <class F> void drain(int worker, F task) {
        drain_with_publish(worker, task, []() {});
    }
    bool done() const {
        unsigned sum = 0;
        for (const auto &c : cursors)
            sum += c.finished.load(std::memory_order_acquire);
        return sum == total;
    }
};

static constexpr int tc_cpack_panels() { return 64; }
static constexpr int tc_cpack_rows() { return 32; }
static constexpr unsigned tc1_cpack_tasks() {

    return unsigned((500 + tc_cpack_panels() - 1) / tc_cpack_panels()) *
           unsigned((4000 + tc_cpack_rows() - 1) / tc_cpack_rows());
}
// Fixed root C pack tasks and the number of releases for each 16 MiB ready chunk.
// Both plans are constant-initialized: no startup constructor or heap allocation.
template <size_t TaskCount> struct CPackPlan {
    std::array<CReadyTask, TaskCount> tasks{};
    std::array<int, 32> pending{};
};
template <bool transposed> static constexpr auto build_cpack_plan() {
    CPackPlan<transposed ? 250u : tc1_cpack_tasks()> plan{};
    auto &tasks = plan.tasks;
    auto *pending = plan.pending.data();
    size_t task_count = 0;
    const int k = transposed ? 8000 : 4000, n = transposed ? 8000 : 16000, grain = transposed ? k : 128;
    if constexpr (!transposed) {
        const int panels = tc_cpack_panels(), rows = tc_cpack_rows();
        for (int jb = 0; jb < n; jb += panels * 32)
            for (int l0 = 0; l0 < k; l0 += rows) {
                const int kc = std::min(rows, k - l0);
                tasks[task_count++] = {jb, l0, kc, 0, 0};
                const int q_end = std::min(n / 32, jb / 32 + panels);
                for (int q = jb / 32; q < q_end; ++q) {
                    const size_t begin = (size_t(q) * 32 * k + size_t(l0) * 32) * 8, end = begin + size_t(kc) * 32 * 8;
                    for (size_t c = begin / c_ready_chunk; c <= (end - 1) / c_ready_chunk; ++c)
                        ++pending[c];
                }
            }
    } else
        for (int jb = 0; jb < n; jb += 32)
            for (int l0 = 0; l0 < k; l0 += grain) {
                int kc = std::min(grain, k - l0);
                size_t begin = (size_t(jb) * k + size_t(l0) * 32) * 8, end = begin + size_t(kc) * 32 * 8;
                tasks[task_count++] = {jb, l0, kc, begin, end};
                for (size_t c = begin / c_ready_chunk; c <= (end - 1) / c_ready_chunk; ++c)
                    ++pending[c];
            }
    if (task_count != tasks.size())
        throw "incorrect C pack task count";
    return plan;
}
static constexpr auto tc1_cpack_plan = build_cpack_plan<false>();
static constexpr auto tc2_cpack_plan = build_cpack_plan<true>();
// C++17 read-only view over the differently sized static task arrays.
struct CPackTaskView {
    const CReadyTask *data;
    size_t count;
    constexpr size_t size() const { return count; }
    constexpr const CReadyTask &operator[](size_t i) const { return data[i]; }
};
struct PrepAcBroadcast {
    Sdma &dma;
    std::vector<u32> pids;
    unsigned long root = 0, relay = 0, relay1 = 0;
    const bool transposed;

    static constexpr size_t bytes = 512000000, metadata = bytes, chunks = (bytes + c_ready_chunk - 1) / c_ready_chunk;
    std::array<CReadyCounter, 32> counters;
    const CPackTaskView tasks;
    PrepAcBroadcast(MPI_Comm comm, int rank, bool transpose = false)
        : dma(*(((224 >> rank) & 1) ? comm_pool->other_binding : comm_pool->dma)), pids(8), transposed(transpose),
          tasks(transpose ? CPackTaskView{tc2_cpack_plan.tasks.data(), tc2_cpack_plan.tasks.size()}
                          : CPackTaskView{tc1_cpack_plan.tasks.data(), tc1_cpack_plan.tasks.size()}) {
        auto &pool = *comm_pool;
        sdma_check(pool.dma->process == pool.other_binding->process, "pipeline process IDs");

        // process IDs and pool addresses never change: exchange them once, let every rank's pool be read through both
        // SDMA devices, then only advance the flag epoch
        static std::vector<u32> known_pids;
        static unsigned long epoch = 0;
        if (!epoch) {
            unsigned long mine[2] = {dma.process, (uintptr_t)pool.c}, all[16];
            MPI_Allgather(mine, 2, MPI_UNSIGNED_LONG, all, 2, MPI_UNSIGNED_LONG, comm);
            known_pids.resize(8);
            for (int r = 0; r < 8; ++r) {
                known_pids[r] = u32(all[2 * r]);
                tree_pools[r] = all[2 * r + 1];
            }
            pool.dma->authorize(known_pids);
            pool.other_binding->authorize(known_pids);
            tree_pools_known = true;
            MPI_Barrier(comm);
        }
        c_epoch = ++epoch;
        pids = known_pids;
        root = tree_pools[0];
        relay = tree_pools[4];
        relay1 = tree_pools[1];

        if (rank == 0) {
            const auto &pending = transposed ? tc2_cpack_plan.pending : tc1_cpack_plan.pending;
            for (size_t c = 0; c < chunks; ++c)
                counters[c].left.store(pending[c], std::memory_order_relaxed);
        }
    }
    void release_piece(size_t begin, size_t end) {
        for (size_t c = begin / c_ready_chunk; c <= (end - 1) / c_ready_chunk; ++c)
            if (counters[c].left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                asm volatile("dmb oshst" ::: "memory");
                __atomic_store_n((u64 *)((char *)comm_pool->c + metadata + c * 64), u64(c_epoch), 3);
            }
    }
    void pack(unsigned id, const double *input) {
        const auto &t = tasks[id];
        if (!transposed) {
            const int p0 = t.jb / 32, np = std::min(tc_cpack_panels(), 500 - p0);
            const svbool_t pg = svptrue_b64();
            for (int l = t.l0; l < t.l0 + t.kc; ++l) {
                const double *src = input + size_t(l) * 16000 + t.jb;
                for (int q = 0; q < np; ++q, src += 32) {
                    double *dst = comm_pool->c + (size_t(p0 + q) * 4000 + l) * 32;
                    const svfloat64_t x0 = svld1_vnum(pg, src, 0), x1 = svld1_vnum(pg, src, 1),
                                      x2 = svld1_vnum(pg, src, 2), x3 = svld1_vnum(pg, src, 3);
                    svst1_vnum(pg, dst, 0, x0);
                    svst1_vnum(pg, dst, 1, x1);
                    svst1_vnum(pg, dst, 2, x2);
                    svst1_vnum(pg, dst, 3, x3);
                }
            }
            for (int q = 0; q < np; ++q) {
                const size_t begin = (size_t(p0 + q) * 32 * 4000 + size_t(t.l0) * 32) * 8;
                release_piece(begin, begin + size_t(t.kc) * 32 * 8);
            }
            return;
        }
        sme_pack_transpose32(8000, input + size_t(t.jb) * 8000, 8000, (double *)((char *)comm_pool->c + t.begin));
        for (size_t c = t.begin / c_ready_chunk; c <= (t.end - 1) / c_ready_chunk; ++c)
            if (counters[c].left.fetch_sub(1, std::memory_order_acq_rel) == 1) {

                asm volatile("dmb oshst" ::: "memory");
                __atomic_store_n((u64 *)((char *)comm_pool->c + metadata + c * 64), u64(c_epoch), 3);
            }
    }
    void receive(int rank, MPI_Comm comm, void (*progress)(void*)=nullptr, void* context=nullptr) {
        c_tree_ready_transfer(comm_pool->c, bytes, metadata, dma, pids, root, relay, relay1, rank, comm, progress, context);
    }
};

static constexpr int tc1_share_groups[3] = {5, 0, 0};

struct ShareRole {
    int give = 0, helper[12]{}, slot[12]{}; // donor: group g (its top p, own+size*g..) goes to this helper slot
    int take = 0, donor[3]{}, group[3]{};   // helper: slot k (local p after its own) holds this donor group
};

// groups[d]: groups given by rank 0, 1, 4; each helper holds up to capacity groups
static ShareRole share_plan(const int *groups, int capacity, int rank) {
    const int donors[3] = {0, 1, 4};
    int load[8]{}, helper[3][12]{}, slot[3][12]{};
    auto assign = [&](int d, int g, std::initializer_list<int> order) {
        int best = -1;
        for (int h : order)
            if (load[h] < capacity && (best < 0 || load[h] < load[best]))
                best = h;
        if (best < 0)
            std::abort();
        helper[d][g] = best;
        slot[d][g] = load[best]++;
    };
    for (int g = 0; g < groups[1]; ++g)
        assign(1, g, {2, 3});
    for (int g = 0; g < groups[2]; ++g)
        assign(2, g, {7, 6, 5});
    for (int g = 0; g < groups[0]; ++g) {
        assign(0, g, {2, 3, 5, 6, 7});
    }
    ShareRole s;
    for (int d = 0; d < 3; ++d)
        for (int g = 0; g < groups[d]; ++g) {
            if (donors[d] == rank) {
                s.give = groups[d];
                s.helper[g] = helper[d][g];
                s.slot[g] = slot[d][g];
            }
            if (helper[d][g] == rank) {
                s.donor[slot[d][g]] = donors[d];
                s.group[slot[d][g]] = g;
                ++s.take;
            }
        }
    return s;
}
// Share exchange through the pinned C pools (receiver-pulled SDMA, as in the C tree): the donor gathers the
// donated A columns into its pool and raises a flag; the helper polls the flag and pulls its group. At the end the
// helper's TB writes the donated output rows into its own pool and raises a flag; the donor pulls them.
// Flags hold the testcase number, so TC2 never mistakes a TC1 flag for its own. A TC1 group is 512000 bytes of A
// and of output, a TC2 group 1024000 bytes of A and 512000 of output.
struct ShareArea {
    static constexpr size_t base = 512032768, group = 512000;
    // a_out holds up to 12 TC1 groups or 11 TC2 groups (twice the size), a_in 3 slots of either
    static constexpr size_t a_out = base, a_flag = a_out + 22 * group, a_in = a_flag + 65536, o_rows = a_in + 6 * group,
                            o_flag = o_rows + 3 * group, o_in = o_flag + 65536, snapshot = o_in + 12 * group,
                            end = snapshot + 64;
    static_assert(end <= CommPool::large && base >= PrepAcBroadcast::metadata + (PrepAcBroadcast::chunks + 2) * 64,
                  "share area inside the C pool");
    static void *at(size_t offset) { return (char *)comm_pool->c + offset; }
    static void wait_flag(Sdma &dma, u32 pid, uintptr_t pool, size_t flag, unsigned long expect, MPI_Comm comm) {
        auto *snapshot = (volatile unsigned long *)at(ShareArea::snapshot);
        const double deadline = tc_seconds() + 30;
        for (;;) {
            hisi_sdma_sqe_task poll{};
            poll.src_addr = pool + flag;
            poll.dst_addr = (uintptr_t)snapshot;
            poll.src_process_id = pid;
            poll.dst_process_id = dma.process;
            poll.length = 64;
            dma.submit_task(poll);
            dma.wait();
            const unsigned long ready = *snapshot;
            if (ready == expect)
                break;
            if (ready > expect || tc_seconds() > deadline)
                MPI_Abort(comm, 5);
        }
    }
    static bool flag_raised(Sdma &dma, u32 pid, uintptr_t pool, size_t flag, unsigned long expect) {
        auto *snapshot = (volatile unsigned long *)at(ShareArea::snapshot);
        hisi_sdma_sqe_task poll{};
        poll.src_addr = pool + flag;
        poll.dst_addr = (uintptr_t)snapshot;
        poll.src_process_id = pid;
        poll.dst_process_id = dma.process;
        poll.length = 64;
        dma.submit_task(poll);
        dma.wait();
        return *snapshot == expect;
    }
    static void submit_copy(Sdma &dma, u32 pid, uintptr_t pool, size_t data, size_t bytes, void *dst) {
        hisi_sdma_sqe_task task{};
        task.src_addr = pool + data;
        task.dst_addr = (uintptr_t)dst;
        task.src_process_id = pid;
        task.dst_process_id = dma.process;
        task.length = bytes;
        dma.submit_task(task);
    }
    static void pull(Sdma &dma, u32 pid, uintptr_t pool, size_t flag, unsigned long expect, size_t data, size_t bytes,
                     void *dst, MPI_Comm comm) {
        wait_flag(dma, pid, pool, flag, expect, comm);
        submit_copy(dma, pid, pool, data, bytes, dst);
        dma.wait();
    }
};
static constexpr int tc2_share_groups[3] = {8, 0, 0}, tc2_share_cap = 2;

// pinned C pool addresses of all ranks (exchanged once) and one-time SDMA read authorization of a helper's pool
static std::array<uintptr_t, 8> share_pools(MPI_Comm comm) {
    if (tree_pools_known)
        return tree_pools;
    static std::array<uintptr_t, 8> pools{};
    static bool ready = false;
    if (!ready) {
        const uintptr_t mine = (uintptr_t)comm_pool->c;
        MPI_Allgather(&mine, 1, MPI_UNSIGNED_LONG, pools.data(), 1, MPI_UNSIGNED_LONG, comm);
        ready = true;
    }
    return pools;
}
static void share_authorize(const std::vector<u32> &pids) {
    static bool done = false;
    if (!done) {
        comm_pool->dma->authorize(pids);
        done = true;
    }
}
// Release unused staging pages before the driver gathers the output.
static void release_temp(size_t retained) {
    if (madvise((char *)comm_pool->temp + retained, CommPool::temp_bytes - retained, MADV_DONTNEED))
        std::abort();
}

// Included inside tcopt, after master preparation and communication helpers.
__attribute__((noinline)) static void sme_pack_a_rows16(long k,const double* source,long lda,double* destination,long rows){
 asm volatile(
  ".arch_extension sme\nmov x9,%[src]\nmov x10,%[dst]\nmov x11,%[k]\nlsl x12,%[lda],#3\nsub x13,%[rows],#8\nsmstart\nptrue p0.d\nwhilelt p1.d,xzr,%[rows]\nwhilelt p2.d,xzr,x13\n1:\n"
  "ld1d {z0.d},p1/z,[x9]\nld1d {z1.d},p2/z,[x9,#1,mul vl]\nst1d {z0.d},p0,[x10]\nst1d {z1.d},p0,[x10,#1,mul vl]\n"
  "add x9,x9,x12\nadd x10,x10,#128\nsubs x11,x11,#1\nb.ne 1b\nsmstop\n"
  ::[src]"r"(source),[dst]"r"(destination),[k]"r"(k),[lda]"r"(lda),[rows]"r"(rows)
  :"memory","cc","x9","x10","x11","x12","x13",
   "z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15",
   "z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31",
   "p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}

__attribute__((noinline,aligned(64))) static void sme_pack_rows32(long k,const double* source,long lda,double* destination){
    asm volatile(
        ".arch_extension sme\nmov x9,%[src]\nmov x10,%[dst]\nmov x11,%[k]\nlsl x17,%[lda],#3\nsmstart\nptrue p0.d\n1:\n"
        "ld1d {z0.d},p0/z,[x9]\nld1d {z1.d},p0/z,[x9,#1,mul vl]\nld1d {z2.d},p0/z,[x9,#2,mul vl]\nld1d {z3.d},p0/z,[x9,#3,mul vl]\n"
        "st1d {z0.d},p0,[x10]\nst1d {z1.d},p0,[x10,#1,mul vl]\nst1d {z2.d},p0,[x10,#2,mul vl]\nst1d {z3.d},p0,[x10,#3,mul vl]\n"
        "add x9,x9,x17\nadd x10,x10,#256\nsubs x11,x11,#1\nb.ne 1b\nsmstop\n"
        ::[src]"r"(source),[dst]"r"(destination),[k]"r"(k),[lda]"r"(lda)
        :"memory", "cc", "x9", "x10", "x11", "x17", "x19", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15");
}
static void pack_tc1_four_flat_task(const double* a,double* out,int task){
    constexpr int m=500,mp=512,k=4000;
    int s=task/32,ib=task%32*16;
    for(int l=0;l<k;l+=320){
        int kc=std::min(320,k-l);
        sme_pack_a_rows16(kc,a+size_t(s)*m*k+size_t(l)*m+ib,m,
            out+size_t(s)*mp*k+size_t(l)*mp+size_t(ib)*kc,std::min(16,m-ib));
    }
}

// TC1 B is already [S*U][local Q500]. Form four RHS bases during its
// existing row packing, preserving [Q32][S*U][32] wire slots and Q128 flags.
struct Tc1TbRhsDescriptor {
    const double* source[4];
    double* output[4];
    long k,right_columns;
};
static_assert(offsetof(Tc1TbRhsDescriptor,k)==64,"TC1 RHS descriptor");
static_assert(offsetof(Tc1TbRhsDescriptor,right_columns)==72,"TC1 RHS tail mask");
__attribute__((noinline,aligned(64))) static void sme_tc1_tb_rhs_pack(const Tc1TbRhsDescriptor &d) {
    asm volatile(
        ".arch_extension sme\n"
        "mov x28,%[d]\n"
        "ldp x9,x10,[x28]\n"
        "ldp x11,x12,[x28,#16]\n"
        "ldp x13,x14,[x28,#32]\n"
        "ldp x15,x16,[x28,#48]\n"
        "ldr x17,[x28,#64]\n"
        "ldr x19,[x28,#72]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "whilelt p1.d,xzr,x19\n"
        "mov x20,#8\n"
        "whilelt p2.d,x20,x19\n"
        "mov x20,#16\n"
        "whilelt p3.d,x20,x19\n"
        "mov x20,#24\n"
        "whilelt p4.d,x20,x19\n"
        "1:\n"
        "ld1d {z0.d},p0/z,[x9]\n"
        "ld1d {z1.d},p1/z,[x10]\n"
        "ld1d {z2.d},p0/z,[x11]\n"
        "ld1d {z3.d},p1/z,[x12]\n"
        "ld1d {z4.d},p0/z,[x9,#1,mul vl]\n"
        "ld1d {z5.d},p2/z,[x10,#1,mul vl]\n"
        "ld1d {z6.d},p0/z,[x11,#1,mul vl]\n"
        "ld1d {z7.d},p2/z,[x12,#1,mul vl]\n"
        "ld1d {z8.d},p0/z,[x9,#2,mul vl]\n"
        "ld1d {z9.d},p3/z,[x10,#2,mul vl]\n"
        "ld1d {z10.d},p0/z,[x11,#2,mul vl]\n"
        "ld1d {z11.d},p3/z,[x12,#2,mul vl]\n"
        "ld1d {z12.d},p0/z,[x9,#3,mul vl]\n"
        "ld1d {z13.d},p4/z,[x10,#3,mul vl]\n"
        "ld1d {z14.d},p0/z,[x11,#3,mul vl]\n"
        "ld1d {z15.d},p4/z,[x12,#3,mul vl]\n"
        "fadd z16.d,z0.d,z3.d\n"
        "fsub z17.d,z1.d,z3.d\n"
        "fsub z18.d,z2.d,z0.d\n"
        "st1d {z16.d},p0,[x13]\n"
        "st1d {z17.d},p0,[x14]\n"
        "st1d {z18.d},p0,[x15]\n"
        "st1d {z3.d},p0,[x16]\n"
        "fadd z16.d,z4.d,z7.d\n"
        "fsub z17.d,z5.d,z7.d\n"
        "fsub z18.d,z6.d,z4.d\n"
        "st1d {z16.d},p0,[x13,#1,mul vl]\n"
        "st1d {z17.d},p0,[x14,#1,mul vl]\n"
        "st1d {z18.d},p0,[x15,#1,mul vl]\n"
        "st1d {z7.d},p0,[x16,#1,mul vl]\n"
        "fadd z16.d,z8.d,z11.d\n"
        "fsub z17.d,z9.d,z11.d\n"
        "fsub z18.d,z10.d,z8.d\n"
        "st1d {z16.d},p0,[x13,#2,mul vl]\n"
        "st1d {z17.d},p0,[x14,#2,mul vl]\n"
        "st1d {z18.d},p0,[x15,#2,mul vl]\n"
        "st1d {z11.d},p0,[x16,#2,mul vl]\n"
        "fadd z16.d,z12.d,z15.d\n"
        "fsub z17.d,z13.d,z15.d\n"
        "fsub z18.d,z14.d,z12.d\n"
        "st1d {z16.d},p0,[x13,#3,mul vl]\n"
        "st1d {z17.d},p0,[x14,#3,mul vl]\n"
        "st1d {z18.d},p0,[x15,#3,mul vl]\n"
        "st1d {z15.d},p0,[x16,#3,mul vl]\n"
        "add x9,x9,#4000\n"
        "add x10,x10,#4000\n"
        "add x11,x11,#4000\n"
        "add x12,x12,#4000\n"
        "add x13,x13,#256\n"
        "add x14,x14,#256\n"
        "add x15,x15,#256\n"
        "add x16,x16,#256\n"
        "subs x17,x17,#1\n"
        "b.ne 1b\n"
        "smstop\n"
        ::[d]"r"(&d)
        :"memory","cc","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}
static constexpr int tc1_tb_rhs_tasks=2*4*25;
static void pack_tc1_tb_rhs_task(const double* input,int index) {
    const int pair=index/100,n=index%100/25*32,k0=index%25*320,q=pair*256+n;
    Tc1TbRhsDescriptor d{{},{},320,std::min(32,500-q-128)};
    for(int z=0;z<4;++z){
        const int qq=q+(z%2)*128,row=k0+(z/2)*8000;
        d.source[z]=input+size_t(row)*500+qq;
        d.output[z]=comm_pool->b_source+size_t(qq)*16000+size_t(row)*32;
    }
    sme_tc1_tb_rhs_pack(d);
}

// Included inside tcopt after the C-chunk readiness constants. No OpenMP team
// barrier is used here: the communication worker may still be draining C.
struct TC1CoopReady {
    struct Group { int owner,begin,count; };
    static constexpr int workers=38,full_tasks=1280,parts=4,batches=13;
    static constexpr int tb_n=1?128:32,tb_m=1?1024:128;
    static constexpr int p_block=tb_m/4,m_groups=512/p_block,q_groups=4096/tb_n,output_tiles=m_groups*q_groups;
    static constexpr int tb_owners=1?output_tiles:workers;
    static constexpr int tb_updates_per_batch=1?112*2:tb_owners*parts;
    static constexpr size_t t_elems=128*320,c_tile=tb_m*tb_n,c_window=2*c_tile;
    std::vector<Group> groups[2];
    int begin[2][workers],owner[2][full_tasks];
    std::array<std::atomic<int>,46> ac_claim;
    std::array<std::array<std::atomic<bool>,20>,batches> wide_claim{};
    std::array<std::atomic<int>,batches> wide_left{};
    std::array<std::atomic<int>,tb_owners> tb_next;
    static_assert(8==8 && 2048==2048 && 1 &&
                  1 && 1 && 320==320 &&
                  1024==1024 && 128==128 && !0,
                  "TB Strassen requires the existing full-row eight-slot pipeline");
    std::array<std::atomic<int>,112> product_next{};
    std::array<std::array<std::atomic<int>,10>,batches> t_sum_state{};
    std::array<std::atomic<int>,16> product_pair_left{};
    struct alignas(64) Cursor { int next=0; };
    std::array<Cursor,workers> cursors;
    alignas(64) std::atomic<int> epoch{0}; // even=AC, odd=TB; 26=done
    alignas(64) std::atomic<int> left{46};
    alignas(64) std::atomic<bool> b_ready{false};
    std::array<std::atomic<bool>,q_groups> b_panels{};
    std::array<std::atomic<int>,batches> panel_tb_left{};
    std::array<std::atomic<bool>,batches> tb_finished{};
    alignas(64) std::atomic<int> ac_jobs_left{125};
    alignas(64) std::atomic<int> tb_k160_left{tb_updates_per_batch*25};
    int producer_quota()const{
        const int jobs=ac_jobs_left.load(std::memory_order_relaxed);
        if(!jobs)return 0;
        // M2048 AC / K160 TB FLOPs = 50. Measured AC/TB rates are
        // 146.97/142.88 GF/s, giving 48.61 TB time units per AC job.
        const int ac=(1?47:49)*jobs,tb=tb_k160_left.load(std::memory_order_relaxed);
        const int share=(workers*ac+(ac+tb)/2)/(ac+tb);
        // A full U320 batch needs ten long AC kernels. Preserve enough
        // preferred workers to launch one batch, including the five-job tail.
        return std::min(workers-1,std::max(std::min(10,jobs),share));
    }
    static constexpr int slots=8?8:1;
    static constexpr size_t partial_elems=output_tiles*c_tile,stage_elems=full_tasks*16*32*4;
    static_assert(slots>=1 && (partial_elems+(slots-(2048?0:1))*stage_elems)*sizeof(double)<=(256ul<<20),
                  "T ring and partials must fit the existing temp pool");
    // consume_batch is the retired prefix. Panel-ready consumers may advance
    // independently; slot reuse still requires every previous reader to retire.
    alignas(64) std::atomic<int> produce_batch{0},consume_batch{0};
    alignas(64) std::atomic<int> ac_left{46},tb_left{tb_owners*parts};
    std::array<std::atomic<bool>,batches> ac_finished{};
    const double *a,*c,*b;
    double *out,*partial;

    TC1CoopReady(const double* aa,const double* cc,const double* bb,double* oo,double* pp)
        :a(aa),c(cc),b(bb),out(oo),partial(pp){
        for(auto& x:ac_claim)x.store(-1,std::memory_order_relaxed);
        for(auto& x:b_panels)x.store(false,std::memory_order_relaxed);
        for(auto& x:panel_tb_left)x.store(tb_updates_per_batch,std::memory_order_relaxed);
        for(auto& x:tb_finished)x.store(false,std::memory_order_relaxed);
        for(auto& x:ac_finished)x.store(false,std::memory_order_relaxed);
        for(auto& x:tb_next)x.store(0,std::memory_order_relaxed);
        for(auto& x:product_next)x.store(0,std::memory_order_relaxed);
        for(auto& batch:t_sum_state)for(auto& x:batch)x.store(0,std::memory_order_relaxed);
        for(auto& x:product_pair_left)x.store(7,std::memory_order_relaxed);
        for(int w=0;w<workers;++w)cursors[w].next=w;
        for(int shape=0;shape<2;++shape){
            const int tasks=shape?640:full_tasks;
            for(int t=0;t<tasks;t+=2048/16)
                groups[shape].push_back({int(groups[shape].size())%workers,t,2048/16});
        }
        for(int batch=0;batch<batches;++batch){
            for(auto& x:wide_claim[batch])x.store(false,std::memory_order_relaxed);
            wide_left[batch].store(int(groups[batch==12].size()),std::memory_order_relaxed);
        }
    }
    bool done()const{
        return consume_batch.load(std::memory_order_acquire)==batches;
    }
    double* staged_at(int batch,int task){
        const int slot=batch%slots;
        return partial+partial_elems+size_t(slot)*stage_elems+size_t(task)*16*32*4;
    }
    void retire(int phase){
        const int batch=phase/2,next=batch+1;
        if(phase%2==0)ac_jobs_left.fetch_sub(1,std::memory_order_relaxed);
        else tb_k160_left.fetch_sub(batch==12?1:2,std::memory_order_relaxed);
        if(phase%2==0){
            if(wide_left[batch].fetch_sub(1,std::memory_order_acq_rel)!=1)return;
            ac_finished[batch].store(true,std::memory_order_release);
        }else{
            if(panel_tb_left[batch].fetch_sub(1,std::memory_order_acq_rel)!=1)return;
            tb_finished[batch].store(true,std::memory_order_release);
            int prefix=consume_batch.load(std::memory_order_acquire);
            while(prefix<batches && tb_finished[prefix].load(std::memory_order_acquire)){
                if(consume_batch.compare_exchange_weak(prefix,prefix+1,std::memory_order_acq_rel))++prefix;
            }
        }
    }
    bool try_produce(int worker,int batch,const std::array<std::atomic<bool>,128>& a_ready){
        const int shape=batch==12;
        const auto& jobs=groups[shape];
        for(int own=1;own>=0;--own)for(size_t j=0;j<jobs.size();++j){
            const auto& job=jobs[j];
            if((job.owner==worker)!=bool(own))continue;
            bool previous=false;
            if(wide_claim[batch][j].load(std::memory_order_relaxed))continue;
            const int row=job.begin%128,u0=batch*320+job.begin/128*32;
            bool ready=true;
            for(int p=row;p<row+job.count;++p)
                if(!a_ready[p].load(std::memory_order_acquire)){ready=false;break;}
            const size_t lo=size_t(u0)*4*4000*8,hi=lo+size_t(128)*4000*8;
            for(size_t chunk=lo/c_ready_chunk;ready && chunk<=(hi-1)/c_ready_chunk;++chunk)
                if(__atomic_load_n((const uint64_t*)((const char*)c+512000000+chunk*64),__ATOMIC_ACQUIRE)!=c_epoch)ready=false;
            if(!ready)continue;
            if(!wide_claim[batch][j].compare_exchange_strong(previous,true,std::memory_order_acq_rel))continue;
            {
                auto& w=scratch[worker];
                tc1_segmented_leaf_tb_strassen<true,false,true,true,true>(a,c+size_t(u0)*4*4000,
                    w.c,w.b,staged_at(batch,0),job.count*16,row*16,batch==12?160:320,
                    job.begin/128*32,w.a,sum_at(batch,0,0));
            }
            retire(batch*2);return true;
        }
        return false;
    }
    double* sum_at(int batch,int slot,int hs){
        // Reserve full K320 strides for every ring slot, including K160 tail.
        return tb_strassen_t_workspace+size_t(batch%slots)*10*1024*320+
               size_t(slot*2+hs)*1024*320;
    }
    bool try_sum(int batch,int slot,int hs){
        int expected=0;
        if(!t_sum_state[batch][slot*2+hs].compare_exchange_strong(expected,1,std::memory_order_acq_rel))return false;
        {
            tbstrassen::transform_panel(staged_at(batch,0),sum_at(batch,slot,hs),slot,hs,batch==12?160:320);
        }
        t_sum_state[batch][slot*2+hs].store(2,std::memory_order_release);
        return true;
    }
    bool try_consume(int worker,int batch){
        for(int offset=0;offset<112;++offset){
            const int destination=(cursors[worker].next+offset)%112;
            int part=product_next[destination].load(std::memory_order_acquire);
            if(part<0 || part>=26)continue;
            const int task_batch=part/2,hs=part%2,which=destination%7,pair=destination/7;
            if(!ac_finished[task_batch].load(std::memory_order_acquire))continue;
            const auto recipe=tbstrassen::br[which];
            if(!b_panels[pair*2+recipe.x%2].load(std::memory_order_acquire) ||
               (recipe.y>=0 && !b_panels[pair*2+recipe.y%2].load(std::memory_order_acquire)))continue;
            const int slot=tbstrassen::derived_slot[which],depth=task_batch==12?160:320;
            if(!product_next[destination].compare_exchange_strong(part,-part-1,std::memory_order_acq_rel))continue;
            {
                const int quad=tbstrassen::ar[which].x;
                const double* source=slot>=0?sum_at(task_batch,slot,hs):
                    staged_at(task_batch,0)+size_t((hs+quad%2*2)*2048+quad/2*1024)*depth;
                double* target=tb_strassen_products_workspace+size_t(destination)*tbstrassen::c_product;
                if(part==0)memset(target,0,tbstrassen::c_product*sizeof(double));
                tbstrassen::update_panel(source,b,scratch[worker].b,target,which,hs,task_batch*320,pair*256);
                if(part==25 && product_pair_left[pair].fetch_sub(1,std::memory_order_acq_rel)==1){
                    tbstrassen::combine_store(tb_strassen_products_workspace,out,pair);
                }
            }
            product_next[destination].store(part+1,std::memory_order_release);
            cursors[worker].next=(destination+1)%112;
            retire(task_batch*2+1);return true;
        }
        return false;
    }
    bool try_compute(int worker,const std::array<std::atomic<bool>,128>& a_ready){
        // Let rank's main thread keep progressing B before it joins compute.
        if(worker==0 && !b_ready.load(std::memory_order_acquire))return false;
        const int consumer=consume_batch.load(std::memory_order_acquire);
        if(consumer==batches)return false;
        auto produce=[&](){
            for(int producer=consumer;producer<std::min(batches,consumer+slots);++producer)
                if(!ac_finished[producer].load(std::memory_order_acquire) && try_produce(worker,producer,a_ready))return true;
            return false;
        };
        // A soft preference, not a concurrency cap. The main worker keeps
        // its communication role; either class can take the other's work.
        const bool prefer_ac=worker>0 && worker<=producer_quota();
        if(prefer_ac && produce())return true;
        if(try_consume(worker,consumer))return true;
        if(prefer_ac)return false; // Already checked the same producer window.
        return produce();
        return false;
    }
};

static int tc1_coop_c_pack_workers=28;

// Each A panel and each B pack task is claimed once. B completion never gates
// AC; only the main worker polls the asynchronous B queue.
struct TC1ReadyPipeline {
    TC1CoopReady* cooperative;
    const double *a_input,*b_input;
    const int pack_workers;
    alignas(64) std::atomic<int> next_a{0},next_b{0},b_left{tc1_tb_rhs_tasks};
    std::array<std::atomic<bool>,128> a_ready{};
    void* b_context=nullptr;
    void (*b_start)(void*)=nullptr;
    static constexpr size_t b_flag=CommPool::small-128;
    TC1ReadyPipeline(const double* a,const double* b,TC1CoopReady* coop)
        :cooperative(coop),a_input(a),b_input(b),pack_workers(tc1_coop_c_pack_workers){
        if(pack_workers<1 || pack_workers>37)std::abort();
        for(auto& ready:a_ready)ready.store(false,std::memory_order_relaxed);
    }
    bool packing_worker(int worker)const{return worker>=38-pack_workers;}
    void start_b_if_ready(){
        if(__atomic_load_n((u64*)((char*)comm_pool->b_source+b_flag),__ATOMIC_ACQUIRE)==1)
            b_start(b_context);
    }
    bool prepare_one(){
        int index=next_a.load(std::memory_order_relaxed);
        if(index<128)index=next_a.fetch_add(1,std::memory_order_relaxed);
        if(index<128){
            pack_tc1_four_flat_task(a_input,packed_a_workspace,index);
            a_ready[index].store(true,std::memory_order_release);
            return true;
        }
        index=next_b.load(std::memory_order_relaxed);
        if(index<tc1_tb_rhs_tasks)index=next_b.fetch_add(1,std::memory_order_relaxed);
        if(index>=tc1_tb_rhs_tasks)return false;
        pack_tc1_tb_rhs_task(b_input,index);
        if(b_left.fetch_sub(1,std::memory_order_acq_rel)==1){
            asm volatile("dmb oshst":::"memory");
            __atomic_store_n((u64*)((char*)comm_pool->b_source+b_flag),u64(1),__ATOMIC_RELEASE);
        }
        return true;
    }
    bool try_compute(int worker){return cooperative->try_compute(worker,a_ready);}
    bool done()const{return cooperative->done();}
};
static TC1ReadyPipeline* tc1_pipeline=nullptr;

// Only the MPI/main worker progresses this separate B DMA queue.
class TC1ReadyGather {
    Sdma& dma;MPI_Comm comm;bool finished=false,started=false;
    std::vector<u32> pids;std::vector<unsigned long> addresses;
    double* destination=nullptr;size_t width=0,total=0;
    size_t checked_peer=0;
    size_t published_panels=0;
    double flag_deadline=0,bulk_deadline=0;
    uint64_t trace_begin=0;
    static Sdma& select_dma(MPI_Comm communicator){
        int rank;MPI_Comm_rank(communicator,&rank);
        sdma_check(comm_pool->dma->process==comm_pool->other_binding->process,"B routing requires equal process IDs");
        return *((224>>rank)&1?comm_pool->dma:comm_pool->other_binding);
    }
public:
    template<class Buffer>
    TC1ReadyGather(const std::vector<double>& local,Buffer& full,int count,MPI_Comm communicator):
        dma(select_dma(communicator)),comm(communicator),destination(full.data()),width(size_t(count)*8),total(full.size()*8){
        if(!tc1_pipeline || local.size()*8>CommPool::small || total>CommPool::large || full.data()!=comm_pool->b_full)MPI_Abort(comm,4);
        if(width>TC1ReadyPipeline::b_flag)MPI_Abort(comm,4);
        memset((char*)comm_pool->b_source+TC1ReadyPipeline::b_flag,0,128);
        int size;MPI_Comm_size(comm,&size);pids.resize(size);addresses.resize(size);
        unsigned long address=(uintptr_t)comm_pool->b_source;
        MPI_Allgather(&dma.process,1,MPI_UNSIGNED,pids.data(),1,MPI_UNSIGNED,comm);
        MPI_Allgather(&address,1,MPI_UNSIGNED_LONG,addresses.data(),1,MPI_UNSIGNED_LONG,comm);
        // A receiver may use either device, so every source authorizes both.
        comm_pool->dma->authorize(pids);comm_pool->other_binding->authorize(pids);
        MPI_Barrier(comm);
        tc1_pipeline->b_context=this;
        tc1_pipeline->b_start=[](void* context){
            auto* gather=static_cast<TC1ReadyGather*>(context);
            {
                auto& ready=tc1_pipeline->cooperative->b_ready;
                if(ready.load(std::memory_order_acquire))return;
                if(gather->progress())ready.store(true,std::memory_order_release);
                return;
            }
        };
    }
    bool progress(){
        if(finished)return true;
        if(!flag_deadline){
            flag_deadline=tc_seconds()+30;
        }
        if(!started){
            auto* snapshot=(u64*)((char*)comm_pool->b_source+TC1ReadyPipeline::b_flag+64);
            while(checked_peer<pids.size()){
                hisi_sdma_sqe_task flag{};
                flag.src_addr=addresses[checked_peer]+TC1ReadyPipeline::b_flag;flag.dst_addr=(uintptr_t)snapshot;
                flag.src_process_id=pids[checked_peer];flag.dst_process_id=dma.process;flag.length=64;
                dma.submit_task(flag);dma.wait(); // Only this 64-byte probe is outstanding.
                const u64 value=*(volatile u64*)snapshot;
                if(value!=1){
                    if(value!=0 || tc_seconds()>flag_deadline)MPI_Abort(comm,5);
                    return false; // Continue C progress instead of spinning on a peer.
                }
                ++checked_peer;
            }
            for(size_t peer=0;peer<pids.size();++peer){
                constexpr size_t panels_per_peer=512/128;
                sdma_check(pids.size()==8 && width==16000ul*512*8 && total==width*8,
                           "TC1 B panel shape");
                const size_t panel_bytes=width/panels_per_peer;
                for(size_t panel=0;panel<panels_per_peer;++panel){
                    const size_t offset=panel*panel_bytes;
                    hisi_sdma_sqe_task task{};
                    task.src_addr=addresses[peer]+offset;
                    task.dst_addr=(uintptr_t)destination+peer*width+offset;
                    task.src_process_id=pids[peer];task.dst_process_id=dma.process;task.length=panel_bytes;
                    task.dst_stride_len=total-panel_bytes;task.stride_num=1;dma.submit_task(task);
                }
            }
            started=true;bulk_deadline=tc_seconds()+10;
        }
        const bool complete=dma.poll();
        // poll may drain a prefix and keep the queue leased. Completion IDs
        // are checked there; outstanding remains valid after its final reset.
        constexpr size_t panel_count=TC1CoopReady::q_groups;
        const size_t completed=panel_count-dma.outstanding;
        sdma_check(completed>=published_panels && completed<=panel_count,"B panel completion prefix");
        if(completed>published_panels)asm volatile("dmb osh":::"memory");
        while(published_panels<completed){
            tc1_pipeline->cooperative->b_panels[published_panels].store(true,std::memory_order_release);
            ++published_panels;
        }
        if(!complete){
            if(tc_seconds()>bulk_deadline)throw std::runtime_error("asynchronous B completion timeout");
            return false;
        }
        finished=true;
        return true;
    }
    void finish(){
        while(!progress())asm volatile("yield":::"memory");
    }
    ~TC1ReadyGather(){if(!finished)std::_Exit(3);MPI_Barrier(comm);}
};

static void tc1_strassen_ready(std::vector<double>& a,std::vector<double>& b,
                              std::vector<double>& c,std::vector<double>& out,MPI_Comm comm,int rank){
    if(omp_get_max_threads()!=38 || svcntd()!=8 || a.size()!=8000000 ||
       b.size()!=8000000 || c.size()!=64000000 || out.size()!=8000000)std::abort();
    // This also advances C epochs and authorizes the current master DMA tree.
    PrepAcBroadcast exchange(comm,rank);
    WorkBuffer all_b(16000ul*512*8,true);
    TC1CoopReady fused(packed_a_workspace,comm_pool->c,all_b.data(),out.data(),comm_pool->temp);
    TC1ReadyPipeline pipeline(a.data(),b.data(),&fused);
    tc1_pipeline=&pipeline;
    TC1ReadyGather pending_b(b,all_b,16000*512,comm);
    std::atomic<int> next_pack{0};
#pragma omp parallel num_threads(38)
    {
        const int worker=omp_get_thread_num();
        if(omp_get_num_threads()!=38)std::abort();
        if(rank!=0 && worker==0){
            exchange.receive(rank,comm,[](void* context){
                static_cast<TC1ReadyPipeline*>(context)->start_b_if_ready();
            },&pipeline);
        }
        for(;;){
            if(worker==0)pipeline.start_b_if_ready();
            if(rank!=0){
                if(worker && pipeline.prepare_one())continue;
                if(pipeline.try_compute(worker))continue;
            }else{
                if(!pipeline.packing_worker(worker) && pipeline.prepare_one())continue;
                const bool packing=next_pack.load(std::memory_order_relaxed)<int(exchange.tasks.size());
                if((!pipeline.packing_worker(worker) || !packing) && pipeline.try_compute(worker))continue;
                if(packing){
                    const int index=next_pack.fetch_add(1,std::memory_order_relaxed);
                    if(index<int(exchange.tasks.size())){
                        exchange.pack(index,c.data());
                        continue;
                    }
                }
                if(pipeline.prepare_one())continue;
            }
            if(pipeline.done())break;
            asm volatile("yield":::"memory");
        }
    }
    pending_b.finish();
    tc1_pipeline=nullptr;
}

__attribute__((noinline, aligned(64))) static void pack_a_rows64(const double *source, double *destination) {
    asm volatile(
  "mov x9,%[src]\nmov x10,%[dst]\nmov x11,#16000\nmov x12,#64\nptrue p0.d\n"
  "1:\nmov x13,x9\nmov x14,x10\nmov x15,#125\n"
  "2:\nld1d {z0.d},p0/z,[x13]\nld1d {z1.d},p0/z,[x13,#1,mul vl]\n"
  "st1d {z0.d},p0,[x14]\nst1d {z1.d},p0,[x14,#1,mul vl]\n"
  "add x13,x13,#128\nadd x14,x14,#10,lsl #12\nsubs x15,x15,#1\nb.ne 2b\n"
  "add x9,x9,x11\nadd x10,x10,#128\nsubs x12,x12,#1\nb.ne 1b\n"
  ::[src]"r"(source),[dst]"r"(destination)
  :"memory","cc","x9","x10","x11","x12","x13","x14","x15","z0","z1","p0");
}

// Existing N128/K160 FMOPA body; final M1 stays in ZA during reconstruction.
// output[] permits a P16 tile to straddle the donated-row boundary at P1000.
struct SmeTbTerminalPanel {
    SmeNPanel base;
    const double* other[6];
    double* output[16];
    long rows,right_columns;
};
static_assert(offsetof(SmeTbTerminalPanel,other)==88 &&
              offsetof(SmeTbTerminalPanel,output)==136 &&
              offsetof(SmeTbTerminalPanel,rows)==264 && sizeof(SmeTbTerminalPanel)==280,"TB terminal layout");
__attribute__((noinline, aligned(64))) static void sme_tb_terminal_plain(const SmeTbTerminalPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cbz x11, 9f\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z6.d}, p0/z, [x9]\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z12.d}, p0/z, [x9]\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z18.d}, p0/z, [x9]\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 1b\n"
        "8:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "ldr x17, [x28]\n"
        "sub x17, x17, x27\n"
        "lsl x6, x17, #12\n"
        "ldr x9, [x28, #88]\n"
        "add x9, x9, x6\n"
        "ldr x10, [x28, #96]\n"
        "add x10, x10, x6\n"
        "ldr x11, [x28, #104]\n"
        "add x11, x11, x6\n"
        "ldr x13, [x28, #112]\n"
        "add x13, x13, x6\n"
        "ldr x14, [x28, #120]\n"
        "add x14, x14, x6\n"
        "ldr x15, [x28, #128]\n"
        "add x15, x15, x6\n"
        "ldr x20, [x28, #264]\n"
        "whilelt p1.d, xzr, x20\n"
        "sub x20, x20, #8\n"
        "whilelt p2.d, xzr, x20\n"
        "ldr x21, [x28, #272]\n"
        "sub x21, x21, x17, lsl #5\n"
        "add x16, x28, #136\n"
        "ld1d {z24.d}, p0/z, [x16]\n"
        "ld1d {z25.d}, p0/z, [x16, #1, mul vl]\n"
        "lsl x17, x17, #9\n"
        "dup z28.d, x17\n"
        "add z24.d, z24.d, z28.d\n"
        "add z25.d, z25.d, z28.d\n"
        "mov x17, #2048\n"
        "dup z28.d, x17\n"
        "add z26.d, z24.d, z28.d\n"
        "add z27.d, z25.d, z28.d\n"
        "mov x19, #0\n"
        "mov w12, #0\n"
        "400:\n"
        "mova z0.d, p0/m, za0v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p1, [z24.d]\n"
        "st1d {z18.d}, p1, [z24.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 401f\n"
        "st1d {z17.d}, p1, [z26.d]\n"
        "st1d {z19.d}, p1, [z26.d, #8]\n"
        "401:\n"
        "mova z0.d, p0/m, za4v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p2, [z25.d]\n"
        "st1d {z18.d}, p2, [z25.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 402f\n"
        "st1d {z17.d}, p2, [z27.d]\n"
        "st1d {z19.d}, p2, [z27.d, #8]\n"
        "402:\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add z24.d, z24.d, #16\n"
        "add z25.d, z25.d, #16\n"
        "add z26.d, z26.d, #16\n"
        "add z27.d, z27.d, #16\n"
        "add x19, x19, #1\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 400b\n"
        "mov w12, #0\n"
        "410:\n"
        "mova z0.d, p0/m, za1v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p1, [z24.d]\n"
        "st1d {z18.d}, p1, [z24.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 411f\n"
        "st1d {z17.d}, p1, [z26.d]\n"
        "st1d {z19.d}, p1, [z26.d, #8]\n"
        "411:\n"
        "mova z0.d, p0/m, za5v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p2, [z25.d]\n"
        "st1d {z18.d}, p2, [z25.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 412f\n"
        "st1d {z17.d}, p2, [z27.d]\n"
        "st1d {z19.d}, p2, [z27.d, #8]\n"
        "412:\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add z24.d, z24.d, #16\n"
        "add z25.d, z25.d, #16\n"
        "add z26.d, z26.d, #16\n"
        "add z27.d, z27.d, #16\n"
        "add x19, x19, #1\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 410b\n"
        "mov w12, #0\n"
        "420:\n"
        "mova z0.d, p0/m, za2v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p1, [z24.d]\n"
        "st1d {z18.d}, p1, [z24.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 421f\n"
        "st1d {z17.d}, p1, [z26.d]\n"
        "st1d {z19.d}, p1, [z26.d, #8]\n"
        "421:\n"
        "mova z0.d, p0/m, za6v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p2, [z25.d]\n"
        "st1d {z18.d}, p2, [z25.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 422f\n"
        "st1d {z17.d}, p2, [z27.d]\n"
        "st1d {z19.d}, p2, [z27.d, #8]\n"
        "422:\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add z24.d, z24.d, #16\n"
        "add z25.d, z25.d, #16\n"
        "add z26.d, z26.d, #16\n"
        "add z27.d, z27.d, #16\n"
        "add x19, x19, #1\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 420b\n"
        "mov w12, #0\n"
        "430:\n"
        "mova z0.d, p0/m, za3v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p1, [z24.d]\n"
        "st1d {z18.d}, p1, [z24.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 431f\n"
        "st1d {z17.d}, p1, [z26.d]\n"
        "st1d {z19.d}, p1, [z26.d, #8]\n"
        "431:\n"
        "mova z0.d, p0/m, za7v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p2, [z25.d]\n"
        "st1d {z18.d}, p2, [z25.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 432f\n"
        "st1d {z17.d}, p2, [z27.d]\n"
        "st1d {z19.d}, p2, [z27.d, #8]\n"
        "432:\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add z24.d, z24.d, #16\n"
        "add z25.d, z25.d, #16\n"
        "add z26.d, z26.d, #16\n"
        "add z27.d, z27.d, #16\n"
        "add x19, x19, #1\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 430b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","x7","x6","x5","x4","cc","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}
__attribute__((noinline, aligned(64))) static void sme_tb_terminal_prefetch(const SmeTbTerminalPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "mov x19, x26\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cbz x11, 9f\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z6.d}, p0/z, [x9]\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z12.d}, p0/z, [x9]\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z18.d}, p0/z, [x9]\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "prfm pldl2keep, [x19]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "prfm pldl2keep, [x19]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 1b\n"
        "8:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "ldr x17, [x28]\n"
        "sub x17, x17, x27\n"
        "lsl x6, x17, #12\n"
        "ldr x9, [x28, #88]\n"
        "add x9, x9, x6\n"
        "ldr x10, [x28, #96]\n"
        "add x10, x10, x6\n"
        "ldr x11, [x28, #104]\n"
        "add x11, x11, x6\n"
        "ldr x13, [x28, #112]\n"
        "add x13, x13, x6\n"
        "ldr x14, [x28, #120]\n"
        "add x14, x14, x6\n"
        "ldr x15, [x28, #128]\n"
        "add x15, x15, x6\n"
        "ldr x20, [x28, #264]\n"
        "whilelt p1.d, xzr, x20\n"
        "sub x20, x20, #8\n"
        "whilelt p2.d, xzr, x20\n"
        "ldr x21, [x28, #272]\n"
        "sub x21, x21, x17, lsl #5\n"
        "add x16, x28, #136\n"
        "ld1d {z24.d}, p0/z, [x16]\n"
        "ld1d {z25.d}, p0/z, [x16, #1, mul vl]\n"
        "lsl x17, x17, #9\n"
        "dup z28.d, x17\n"
        "add z24.d, z24.d, z28.d\n"
        "add z25.d, z25.d, z28.d\n"
        "mov x17, #2048\n"
        "dup z28.d, x17\n"
        "add z26.d, z24.d, z28.d\n"
        "add z27.d, z25.d, z28.d\n"
        "mov x19, #0\n"
        "mov w12, #0\n"
        "400:\n"
        "mova z0.d, p0/m, za0v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p1, [z24.d]\n"
        "st1d {z18.d}, p1, [z24.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 401f\n"
        "st1d {z17.d}, p1, [z26.d]\n"
        "st1d {z19.d}, p1, [z26.d, #8]\n"
        "401:\n"
        "mova z0.d, p0/m, za4v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p2, [z25.d]\n"
        "st1d {z18.d}, p2, [z25.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 402f\n"
        "st1d {z17.d}, p2, [z27.d]\n"
        "st1d {z19.d}, p2, [z27.d, #8]\n"
        "402:\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add z24.d, z24.d, #16\n"
        "add z25.d, z25.d, #16\n"
        "add z26.d, z26.d, #16\n"
        "add z27.d, z27.d, #16\n"
        "add x19, x19, #1\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 400b\n"
        "mov w12, #0\n"
        "410:\n"
        "mova z0.d, p0/m, za1v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p1, [z24.d]\n"
        "st1d {z18.d}, p1, [z24.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 411f\n"
        "st1d {z17.d}, p1, [z26.d]\n"
        "st1d {z19.d}, p1, [z26.d, #8]\n"
        "411:\n"
        "mova z0.d, p0/m, za5v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p2, [z25.d]\n"
        "st1d {z18.d}, p2, [z25.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 412f\n"
        "st1d {z17.d}, p2, [z27.d]\n"
        "st1d {z19.d}, p2, [z27.d, #8]\n"
        "412:\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add z24.d, z24.d, #16\n"
        "add z25.d, z25.d, #16\n"
        "add z26.d, z26.d, #16\n"
        "add z27.d, z27.d, #16\n"
        "add x19, x19, #1\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 410b\n"
        "mov w12, #0\n"
        "420:\n"
        "mova z0.d, p0/m, za2v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p1, [z24.d]\n"
        "st1d {z18.d}, p1, [z24.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 421f\n"
        "st1d {z17.d}, p1, [z26.d]\n"
        "st1d {z19.d}, p1, [z26.d, #8]\n"
        "421:\n"
        "mova z0.d, p0/m, za6v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p2, [z25.d]\n"
        "st1d {z18.d}, p2, [z25.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 422f\n"
        "st1d {z17.d}, p2, [z27.d]\n"
        "st1d {z19.d}, p2, [z27.d, #8]\n"
        "422:\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add z24.d, z24.d, #16\n"
        "add z25.d, z25.d, #16\n"
        "add z26.d, z26.d, #16\n"
        "add z27.d, z27.d, #16\n"
        "add x19, x19, #1\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 420b\n"
        "mov w12, #0\n"
        "430:\n"
        "mova z0.d, p0/m, za3v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p1, [z24.d]\n"
        "st1d {z18.d}, p1, [z24.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 431f\n"
        "st1d {z17.d}, p1, [z26.d]\n"
        "st1d {z19.d}, p1, [z26.d, #8]\n"
        "431:\n"
        "mova z0.d, p0/m, za7v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p2, [z25.d]\n"
        "st1d {z18.d}, p2, [z25.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 432f\n"
        "st1d {z17.d}, p2, [z27.d]\n"
        "st1d {z19.d}, p2, [z27.d, #8]\n"
        "432:\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add z24.d, z24.d, #16\n"
        "add z25.d, z25.d, #16\n"
        "add z26.d, z26.d, #16\n"
        "add z27.d, z27.d, #16\n"
        "add x19, x19, #1\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 430b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","x7","x6","x5","x4","cc","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}
__attribute__((noinline, aligned(64))) static void sme_tb_terminal_run(const SmeTbTerminalPanel* panels,long count) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "mov x3, %[count]\n"
        "smstart\n"
        "900:\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "mov x19, x26\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cbz x11, 9f\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z6.d}, p0/z, [x9]\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z12.d}, p0/z, [x9]\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z18.d}, p0/z, [x9]\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "prfm pldl2keep, [x19]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "prfm pldl2keep, [x19]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 1b\n"
        "8:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "ldr x17, [x28]\n"
        "sub x17, x17, x27\n"
        "lsl x6, x17, #12\n"
        "ldr x9, [x28, #88]\n"
        "add x9, x9, x6\n"
        "ldr x10, [x28, #96]\n"
        "add x10, x10, x6\n"
        "ldr x11, [x28, #104]\n"
        "add x11, x11, x6\n"
        "ldr x13, [x28, #112]\n"
        "add x13, x13, x6\n"
        "ldr x14, [x28, #120]\n"
        "add x14, x14, x6\n"
        "ldr x15, [x28, #128]\n"
        "add x15, x15, x6\n"
        "ldr x20, [x28, #264]\n"
        "whilelt p1.d, xzr, x20\n"
        "sub x20, x20, #8\n"
        "whilelt p2.d, xzr, x20\n"
        "ldr x21, [x28, #272]\n"
        "sub x21, x21, x17, lsl #5\n"
        "add x16, x28, #136\n"
        "ld1d {z24.d}, p0/z, [x16]\n"
        "ld1d {z25.d}, p0/z, [x16, #1, mul vl]\n"
        "lsl x17, x17, #9\n"
        "dup z28.d, x17\n"
        "add z24.d, z24.d, z28.d\n"
        "add z25.d, z25.d, z28.d\n"
        "mov x17, #2048\n"
        "dup z28.d, x17\n"
        "add z26.d, z24.d, z28.d\n"
        "add z27.d, z25.d, z28.d\n"
        "mov x19, #0\n"
        "mov w12, #0\n"
        "400:\n"
        "mova z0.d, p0/m, za0v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p1, [z24.d]\n"
        "st1d {z18.d}, p1, [z24.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 401f\n"
        "st1d {z17.d}, p1, [z26.d]\n"
        "st1d {z19.d}, p1, [z26.d, #8]\n"
        "401:\n"
        "mova z0.d, p0/m, za4v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p2, [z25.d]\n"
        "st1d {z18.d}, p2, [z25.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 402f\n"
        "st1d {z17.d}, p2, [z27.d]\n"
        "st1d {z19.d}, p2, [z27.d, #8]\n"
        "402:\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add z24.d, z24.d, #16\n"
        "add z25.d, z25.d, #16\n"
        "add z26.d, z26.d, #16\n"
        "add z27.d, z27.d, #16\n"
        "add x19, x19, #1\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 400b\n"
        "mov w12, #0\n"
        "410:\n"
        "mova z0.d, p0/m, za1v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p1, [z24.d]\n"
        "st1d {z18.d}, p1, [z24.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 411f\n"
        "st1d {z17.d}, p1, [z26.d]\n"
        "st1d {z19.d}, p1, [z26.d, #8]\n"
        "411:\n"
        "mova z0.d, p0/m, za5v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p2, [z25.d]\n"
        "st1d {z18.d}, p2, [z25.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 412f\n"
        "st1d {z17.d}, p2, [z27.d]\n"
        "st1d {z19.d}, p2, [z27.d, #8]\n"
        "412:\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add z24.d, z24.d, #16\n"
        "add z25.d, z25.d, #16\n"
        "add z26.d, z26.d, #16\n"
        "add z27.d, z27.d, #16\n"
        "add x19, x19, #1\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 410b\n"
        "mov w12, #0\n"
        "420:\n"
        "mova z0.d, p0/m, za2v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p1, [z24.d]\n"
        "st1d {z18.d}, p1, [z24.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 421f\n"
        "st1d {z17.d}, p1, [z26.d]\n"
        "st1d {z19.d}, p1, [z26.d, #8]\n"
        "421:\n"
        "mova z0.d, p0/m, za6v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p2, [z25.d]\n"
        "st1d {z18.d}, p2, [z25.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 422f\n"
        "st1d {z17.d}, p2, [z27.d]\n"
        "st1d {z19.d}, p2, [z27.d, #8]\n"
        "422:\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add z24.d, z24.d, #16\n"
        "add z25.d, z25.d, #16\n"
        "add z26.d, z26.d, #16\n"
        "add z27.d, z27.d, #16\n"
        "add x19, x19, #1\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 420b\n"
        "mov w12, #0\n"
        "430:\n"
        "mova z0.d, p0/m, za3v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p1, [z24.d]\n"
        "st1d {z18.d}, p1, [z24.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 431f\n"
        "st1d {z17.d}, p1, [z26.d]\n"
        "st1d {z19.d}, p1, [z26.d, #8]\n"
        "431:\n"
        "mova z0.d, p0/m, za7v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "st1d {z16.d}, p2, [z25.d]\n"
        "st1d {z18.d}, p2, [z25.d, #8]\n"
        "cmp x19, x21\n"
        "b.ge 432f\n"
        "st1d {z17.d}, p2, [z27.d]\n"
        "st1d {z19.d}, p2, [z27.d, #8]\n"
        "432:\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add z24.d, z24.d, #16\n"
        "add z25.d, z25.d, #16\n"
        "add z26.d, z26.d, #16\n"
        "add z27.d, z27.d, #16\n"
        "add x19, x19, #1\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 430b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "subs x3, x3, #1\n"
        "b.eq 901f\n"
        "add x28, x28, #280\n"
        "b 900b\n"
        "901:\n"
        "smstop\n"
        : : [descriptor]"r"(panels),[count]"r"(count) : "memory","x3","x7","x6","x5","x4","cc","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}

// Included inside tcopt. A terminal AC tile forms seven operands over (r,t).
// TB retains the existing N128/K320 microkernel and its K160 final panel.
struct Tc2StrassenTeam {
    const double *a,*b;
    Output out;
    int m;
    size_t product_size;
    std::atomic<int> next{0},remaining{112},ready{0},combine_remaining{0};
    std::atomic<int> pair_left[16];
    std::atomic<unsigned> combine_bits[16];
    std::atomic<int> pair_slices[16];
    int slices;
    // A negative step leases the product exclusively until this run ends. Partial sums
    // never have concurrent writers; K steps stay in their original order.
    struct alignas(64) ProductProgress {std::atomic<int> step{0};};
    std::array<ProductProgress,112> progress;
    struct alignas(64) Cursor {int next=0;};
    std::array<Cursor,38> cursors;
    static constexpr int follower[7]={-1,-1,5,6,1,-1,-1};
    static constexpr bool is_follower[7]={false,true,false,false,false,true,true};
    static constexpr uint32_t panel_mask(int step){
        const int lo=step*320/128;
        const int hi=(step==12?3999:(step+1)*320-1)/128;
        return uint32_t(((uint64_t(1)<<(hi-lo+1))-1)<<lo);
    }
    Tc2StrassenTeam(const double* aa,const double* bb,Output oo,int mm)
      :a(aa),b(bb),out(oo),m(mm),product_size(size_t(mm)*128),slices((oo.logical_rows+127)/128) {
        if(112*product_size*8>tb_strassen_products_bytes)std::abort();
        combine_remaining.store(16*slices,std::memory_order_relaxed);
        for(int w=0;w<38;++w)cursors[w].next=w;
        for(int i=0;i<16;++i){
            pair_left[i].store(7,std::memory_order_relaxed);
            combine_bits[i].store(0,std::memory_order_relaxed);
            pair_slices[i].store(slices,std::memory_order_relaxed);
        }
    }
    const double* bquad(int quad,int u,int q) const {
        return b+size_t(q+(quad%2)*128)*8000+size_t((quad/2)*4000+u)*32;
    }
    void product_steps(int item,double* packed,int begin,int end,bool inplace=false) {
        const int pair=item/7,which=item%7;

        static constexpr tbstrassen::Recipe basis_recipe[7]={
            {0,-1,0},{0,3,-1},{1,-1,0},{2,-1,0},{3,-1,0},{0,1,1},{0,2,1}};
        const auto r=basis_recipe[which];
        double* c=tb_strassen_products_workspace+size_t(item)*product_size;
        for(int u=begin*320;u<end*320;u+=320){
            const int k=std::min(320,4000-u),row_count=m/16;

            std::array<const double*,64> rows;
            for(int i=0;i<row_count;++i)rows[i]=a+tc2_strassen_index(which,i*16,u);
            SmeNPanel first{4,k,rows[0],bquad(r.x,u,pair*256),c,packed,rows[1],
                           8000L*32*8,long(k)*32*8,k*16*8/4,u!=0};
            {

            if(inplace){
                SmeSumPanel sum{first,packed};
                if(r.sign==1)tc1_rhs_sum_lookahead<false,true>(sum);
                else tc1_rhs_sum_lookahead<true,true>(sum);
            }else if(r.y<0)tc1_rhs_pack_lookahead(first);
            else {
                SmeSumPanel sum{first,bquad(r.y,u,pair*256)};
                if(r.sign==1)tc1_rhs_sum_lookahead<false>(sum);
                else tc1_rhs_sum_lookahead<true>(sum);
            }
            }
            {

            SmeNPanel mid{4,k,rows[1],packed,c+16*128,packed,rows[2],
                         long(k)*32*8,long(k)*32*8,k*16*8/4,u!=0};
            SmeMRun run{{mid,c+32*128},rows.data()+1,row_count-2};
            sme_mrun_accumulate(run);
            SmeNPanel last{4,k,rows[row_count-1],packed,c+(m-16)*128,packed,nullptr,
                          long(k)*32*8,long(k)*32*8,k*16*8/4,u!=0};
            sme_npanel_pipeline(last);
            }
        }
    }
    void product(int item,double* packed){product_steps(item,packed,0,13);}
    bool try_product(int worker,uint32_t panels,bool ac_done){
        if(!ready.load(std::memory_order_acquire))return false;
        int first=cursors[worker].next;
        if(ac_done){
            // Remaining K-chain lengths differ after panel overlap. Start the
            // longest available chain first, retaining its worker for up to 1 steps.
            int best=-1,least=13;
            for(int offset=0;offset<112;++offset){
                const int item=(first+offset)%112;
                if(is_follower[item%7])continue;
                const int step=progress[item].step.load(std::memory_order_acquire);
                if(step>=0 && step<least){best=item;least=step;}
            }
            if(best<0)return false;
            first=best;
        }
        // A lost longest-chain claim returns to the outer cooperative loop;
        // do not silently substitute a short chain from the cyclic order.
        for(int offset=0;offset<(ac_done?1:112);++offset){
            const int item=(first+offset)%112;
            if(is_follower[item%7])continue;
            const int child=follower[item%7],products=child<0?1:2;
            auto& state=progress[item].step;
            int step=state.load(std::memory_order_acquire);
            if(step<0 || step>=13 || (panels&panel_mask(step))!=panel_mask(step))continue;
            if(!state.compare_exchange_strong(step,-step-1,std::memory_order_acq_rel,
                                               std::memory_order_acquire))continue;
            // Before AC is complete, yield after one K320/K160 step to give
            // newly ready AC work priority. Afterwards retain up to 1 K
            // steps, then release the product for longest-chain scheduling.
            int end=step+1;
            if(ac_done)while(end<std::min(13,step+1) && (panels&panel_mask(end))==panel_mask(end))++end;
            // One exclusive lease owns both products. Consume the leader's
            // packed RHS immediately, before the next K step overwrites it.
            if(child<0)product_steps(item,scratch[worker].b,step,end);
            else for(int k=step;k<end;++k){
                product_steps(item,scratch[worker].b,k,k+1);
                product_steps(item-item%7+child,scratch[worker].b,k,k+1,true);
            }
            if(child>=0)progress[item-item%7+child].step.store(end,std::memory_order_release);
            state.store(end,std::memory_order_release);
            if(end==13){
                if(pair_left[item/7].fetch_sub(products,std::memory_order_acq_rel)==products)
                    combine_bits[item/7].store((1u<<slices)-1,std::memory_order_release);
                remaining.fetch_sub(products,std::memory_order_release);
            }
            cursors[worker].next=end==13?(item+1)%112:item;
            return true;
        }
        return false;
    }

    void combine(int pair,int slice) {
        const int lo=slice*128,hi=std::min(lo+128,out.logical_rows);

        const double* p=tb_strassen_products_workspace+size_t(pair)*7*product_size;
        const int qbase=(pair/2)*500+(pair%2)*256;
        const int nright=pair%2?116:128;
        std::array<SmeTbTerminalPanel,8> descriptors;
        int count=0;
        for(int row=lo;row<hi;row+=16){
            const int rows=std::min(16,hi-row);
            const double* input=a+tc2_strassen_index(0,row,3840);
            double* packed=scratch[omp_get_thread_num()].b;
            const int depth=0;
            const double* future=row+16<m?input+16*160:input;
            SmeNPanel n{4,depth,input,packed,const_cast<double*>(p)+size_t(row)*128,packed,future,
                        160L*32*8,160L*32*8,160*16*8/4,1};
            auto& d=descriptors[count++];d=SmeTbTerminalPanel{n,{},{},rows,nright};
            for(int z=0;z<6;++z)d.other[z]=p+size_t(z+1)*product_size+size_t(row)*128;
            for(int i=0;i<rows;++i){
                const int at=row+i;
                d.output[i]=(at<out.spill_row?out.ptr+size_t(at)*8000:
                             out.spill+size_t(at-out.spill_row)*8000)+qbase*2;
            }
        }
        sme_tb_terminal_run(descriptors.data(),count);
        if(pair_slices[pair].fetch_sub(1,std::memory_order_acq_rel)==1 &&
           out.spill_flag && __atomic_add_fetch(out.spill_count,1u,__ATOMIC_ACQ_REL)==16){
            asm volatile("dmb oshst":::"memory");
            __atomic_store_n(out.spill_flag,out.spill_value,__ATOMIC_RELEASE);
        }
        combine_remaining.fetch_sub(1,std::memory_order_release);
    }
    bool help_combine(int worker) {
        for(int i=0;i<16;++i){
            const int pair=(worker+i)%16;
            auto& bank=combine_bits[pair];
            unsigned bits=bank.load(std::memory_order_acquire);
            while(bits){
                const int slice=__builtin_ctz(bits);
                if(bank.compare_exchange_weak(bits,bits&(bits-1),std::memory_order_acq_rel,
                                               std::memory_order_acquire)){
                    combine(pair,slice);
                    return true;
                }
            }
        }
        return false;
    }
    void work(int worker) {
        while(!ready.load(std::memory_order_acquire))asm volatile("yield":::"memory");
        for(int item;(item=next.fetch_add(1,std::memory_order_relaxed))<112;){
            product(item,scratch[worker].b);
            if(pair_left[item/7].fetch_sub(1,std::memory_order_acq_rel)==1)
                combine_bits[item/7].store((1u<<slices)-1,std::memory_order_release);
            remaining.fetch_sub(1,std::memory_order_release);
        }
        // GEMMs have priority; idle workers drain disjoint P128 output slices.
        while(combine_remaining.load(std::memory_order_acquire))
            if(!help_combine(worker))asm volatile("yield":::"memory");
    }
    void run() {
        ready.store(1,std::memory_order_release);
        work(0);
        while(remaining.load(std::memory_order_acquire))asm volatile("yield":::"memory");
    }
};

// Aligned (t,r) reconstruction -> (r,t) TB input transform, directly from ZA.
// Products M2..M7 are complete; each output pointer addresses one full P16/U32.
struct SmeAcTerminalPanel {
    SmeSumPanel base;
    const double* other[6];
    double* output[4][7];
};
static_assert(offsetof(SmeAcTerminalPanel,other)==96 &&
              offsetof(SmeAcTerminalPanel,output)==144 && sizeof(SmeAcTerminalPanel)==368,"terminal descriptor");
__attribute__((noinline, aligned(64))) static void sme_ac_terminal_plain(const SmeAcTerminalPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z6.d}, p0/z, [x9]\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z12.d}, p0/z, [x9]\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z18.d}, p0/z, [x9]\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 1b\n"
        "8:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "ldr x17, [x28]\n"
        "sub x17, x17, x27\n"
        "lsl x6, x17, #12\n"
        "ldr x9, [x28, #96]\n"
        "add x9, x9, x6\n"
        "ldr x10, [x28, #104]\n"
        "add x10, x10, x6\n"
        "ldr x11, [x28, #112]\n"
        "add x11, x11, x6\n"
        "ldr x13, [x28, #120]\n"
        "add x13, x13, x6\n"
        "ldr x14, [x28, #128]\n"
        "add x14, x14, x6\n"
        "ldr x15, [x28, #136]\n"
        "add x15, x15, x6\n"
        "mov x5, #56\n"
        "madd x16, x17, x5, x28\n"
        "add x16, x16, #144\n"
        "ldr x4, [x16, #0]\n"
        "ldr x5, [x16, #8]\n"
        "ldr x6, [x16, #16]\n"
        "ldr x17, [x16, #32]\n"
        "ldr x19, [x16, #40]\n"
        "ldr x20, [x16, #48]\n"
        "ldr x16, [x16, #24]\n"
        "mov w12, #0\n"
        "300:\n"
        "mova z0.d, p0/m, za0v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #0, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #0, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #0, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #0, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #0, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #0, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #0, mul vl]\n"
        "mova z0.d, p0/m, za4v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #1, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #1, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #1, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #1, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #1, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #1, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #1, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add x4, x4, #128\n"
        "add x5, x5, #128\n"
        "add x6, x6, #128\n"
        "add x16, x16, #128\n"
        "add x17, x17, #128\n"
        "add x19, x19, #128\n"
        "add x20, x20, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 300b\n"
        "mov w12, #0\n"
        "301:\n"
        "mova z0.d, p0/m, za1v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #0, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #0, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #0, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #0, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #0, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #0, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #0, mul vl]\n"
        "mova z0.d, p0/m, za5v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #1, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #1, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #1, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #1, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #1, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #1, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #1, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add x4, x4, #128\n"
        "add x5, x5, #128\n"
        "add x6, x6, #128\n"
        "add x16, x16, #128\n"
        "add x17, x17, #128\n"
        "add x19, x19, #128\n"
        "add x20, x20, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 301b\n"
        "mov w12, #0\n"
        "302:\n"
        "mova z0.d, p0/m, za2v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #0, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #0, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #0, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #0, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #0, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #0, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #0, mul vl]\n"
        "mova z0.d, p0/m, za6v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #1, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #1, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #1, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #1, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #1, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #1, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #1, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add x4, x4, #128\n"
        "add x5, x5, #128\n"
        "add x6, x6, #128\n"
        "add x16, x16, #128\n"
        "add x17, x17, #128\n"
        "add x19, x19, #128\n"
        "add x20, x20, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 302b\n"
        "mov w12, #0\n"
        "303:\n"
        "mova z0.d, p0/m, za3v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #0, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #0, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #0, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #0, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #0, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #0, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #0, mul vl]\n"
        "mova z0.d, p0/m, za7v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #1, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #1, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #1, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #1, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #1, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #1, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #1, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add x4, x4, #128\n"
        "add x5, x5, #128\n"
        "add x6, x6, #128\n"
        "add x16, x16, #128\n"
        "add x17, x17, #128\n"
        "add x19, x19, #128\n"
        "add x20, x20, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 303b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","x7","x6","x5","x4","cc","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}
__attribute__((noinline, aligned(64))) static void sme_ac_terminal_prefetch(const SmeAcTerminalPanel &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "mov x19, x26\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z6.d}, p0/z, [x9]\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z12.d}, p0/z, [x9]\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "ld1d {z18.d}, p0/z, [x9]\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n" "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n" "prfm pldl2keep, [x19]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n" "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n" "ld1d {z0.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n" "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n" "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n" "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n" "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z2.d}, p0/z, [x10]\n" "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n" "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n" "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n" "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n" "ld1d {z6.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n" "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n" "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n" "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n" "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z8.d}, p0/z, [x10]\n" "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n" "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n" "prfm pldl2keep, [x19]\n" "add x19, x19, #64\n" "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n" "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n" "ld1d {z12.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n" "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n" "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n" "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n" "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z14.d}, p0/z, [x10]\n" "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n" "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n" "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n" "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n" "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n" "ld1d {z18.d}, p0/z, [x9]\n" "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n" "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n" "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n" "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n" "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n" "ld1d {z20.d}, p0/z, [x10]\n" "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n" "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n" "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n" "add x9, x9, #128\n" "add x10, x10, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 1b\n"
        "8:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "prfm pldl2keep, [x19]\n"
        "add x19, x19, #64\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "ldr x17, [x28]\n"
        "sub x17, x17, x27\n"
        "lsl x6, x17, #12\n"
        "ldr x9, [x28, #96]\n"
        "add x9, x9, x6\n"
        "ldr x10, [x28, #104]\n"
        "add x10, x10, x6\n"
        "ldr x11, [x28, #112]\n"
        "add x11, x11, x6\n"
        "ldr x13, [x28, #120]\n"
        "add x13, x13, x6\n"
        "ldr x14, [x28, #128]\n"
        "add x14, x14, x6\n"
        "ldr x15, [x28, #136]\n"
        "add x15, x15, x6\n"
        "mov x5, #56\n"
        "madd x16, x17, x5, x28\n"
        "add x16, x16, #144\n"
        "ldr x4, [x16, #0]\n"
        "ldr x5, [x16, #8]\n"
        "ldr x6, [x16, #16]\n"
        "ldr x17, [x16, #32]\n"
        "ldr x19, [x16, #40]\n"
        "ldr x20, [x16, #48]\n"
        "ldr x16, [x16, #24]\n"
        "mov w12, #0\n"
        "300:\n"
        "mova z0.d, p0/m, za0v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #0, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #0, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #0, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #0, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #0, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #0, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #0, mul vl]\n"
        "mova z0.d, p0/m, za4v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #1, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #1, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #1, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #1, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #1, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #1, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #1, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add x4, x4, #128\n"
        "add x5, x5, #128\n"
        "add x6, x6, #128\n"
        "add x16, x16, #128\n"
        "add x17, x17, #128\n"
        "add x19, x19, #128\n"
        "add x20, x20, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 300b\n"
        "mov w12, #0\n"
        "301:\n"
        "mova z0.d, p0/m, za1v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #0, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #0, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #0, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #0, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #0, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #0, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #0, mul vl]\n"
        "mova z0.d, p0/m, za5v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #1, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #1, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #1, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #1, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #1, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #1, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #1, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add x4, x4, #128\n"
        "add x5, x5, #128\n"
        "add x6, x6, #128\n"
        "add x16, x16, #128\n"
        "add x17, x17, #128\n"
        "add x19, x19, #128\n"
        "add x20, x20, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 301b\n"
        "mov w12, #0\n"
        "302:\n"
        "mova z0.d, p0/m, za2v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #0, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #0, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #0, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #0, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #0, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #0, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #0, mul vl]\n"
        "mova z0.d, p0/m, za6v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #1, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #1, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #1, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #1, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #1, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #1, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #1, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add x4, x4, #128\n"
        "add x5, x5, #128\n"
        "add x6, x6, #128\n"
        "add x16, x16, #128\n"
        "add x17, x17, #128\n"
        "add x19, x19, #128\n"
        "add x20, x20, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 302b\n"
        "mov w12, #0\n"
        "303:\n"
        "mova z0.d, p0/m, za3v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #0, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #0, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #0, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #0, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #0, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #0, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #0, mul vl]\n"
        "mova z0.d, p0/m, za7v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #1, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #1, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #1, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #1, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #1, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #1, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #1, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add x4, x4, #128\n"
        "add x5, x5, #128\n"
        "add x6, x6, #128\n"
        "add x16, x16, #128\n"
        "add x17, x17, #128\n"
        "add x19, x19, #128\n"
        "add x20, x20, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 303b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d) : "memory","x7","x6","x5","x4","cc","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}
template<bool subtract>
__attribute__((noinline,aligned(64))) static void sme_ac_terminal_sum(const SmeAcTerminalPanel& d){
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28, %[descriptor]\n"
        "ldr x27, [x28]\n"
        "ldr x22, [x28, #16]\n"
        "ldr x23, [x28, #24]\n"
        "ldr x24, [x28, #32]\n"
        "ldr x25, [x28, #40]\n"
        "ldr x26, [x28, #48]\n"
        "ldr x8, [x28, #80]\n"
        "ldr x7, [x28, #88]\n"
        "smstart\n"
        "ptrue p0.d\n"
        "90:\n"
        ".arch_extension sme-f64f64\n"
        "mov x9, x22\n"
        "mov x10, x23\n"
        "mov x6, x7\n"
        "ldr x11, [x28, #8]\n"
        "mov x17, #128\n"
        "mov x19, x25\n"
        "mov x20, #256\n"
        "zero {za}\n"
        "cbz w8, 91f\n"
        "mov x13, x24\n"
        "add x14, x13, #1024\n"
        "add x15, x14, #1024\n"
        "add x16, x15, #1024\n"
        "mov x21, #8\n"
        "mov w12, #0\n"
        "92:\n"
        "ld1d {za0v.d[w12, 0]}, p0/z, [x13]\n"
        "ld1d {za4v.d[w12, 0]}, p0/z, [x13, x21, lsl #3]\n"
        "add x13, x13, #128\n"
        "ld1d {za1v.d[w12, 0]}, p0/z, [x14]\n"
        "ld1d {za5v.d[w12, 0]}, p0/z, [x14, x21, lsl #3]\n"
        "add x14, x14, #128\n"
        "ld1d {za2v.d[w12, 0]}, p0/z, [x15]\n"
        "ld1d {za6v.d[w12, 0]}, p0/z, [x15, x21, lsl #3]\n"
        "add x15, x15, #128\n"
        "ld1d {za3v.d[w12, 0]}, p0/z, [x16]\n"
        "ld1d {za7v.d[w12, 0]}, p0/z, [x16, x21, lsl #3]\n"
        "add x16, x16, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 92b\n"
        "91:\n"
        "cmp x11, #4\n"
        "b.lt 7f\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z2.d, z2.d, z24.d\n"
        ".else\n"
        "fadd z2.d, z2.d, z24.d\n"
        ".endif\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z3.d, z3.d, z25.d\n"
        ".else\n"
        "fadd z3.d, z3.d, z25.d\n"
        ".endif\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z4.d, z4.d, z26.d\n"
        ".else\n"
        "fadd z4.d, z4.d, z26.d\n"
        ".endif\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z5.d, z5.d, z27.d\n"
        ".else\n"
        "fadd z5.d, z5.d, z27.d\n"
        ".endif\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z8.d, z8.d, z24.d\n"
        ".else\n"
        "fadd z8.d, z8.d, z24.d\n"
        ".endif\n"
        "st1d {z8.d}, p0, [x19]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z9.d, z9.d, z25.d\n"
        ".else\n"
        "fadd z9.d, z9.d, z25.d\n"
        ".endif\n"
        "st1d {z9.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z10.d, z10.d, z26.d\n"
        ".else\n"
        "fadd z10.d, z10.d, z26.d\n"
        ".endif\n"
        "st1d {z10.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z11.d, z11.d, z27.d\n"
        ".else\n"
        "fadd z11.d, z11.d, z27.d\n"
        ".endif\n"
        "st1d {z11.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z14.d, z14.d, z24.d\n"
        ".else\n"
        "fadd z14.d, z14.d, z24.d\n"
        ".endif\n"
        "st1d {z14.d}, p0, [x19]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z15.d, z15.d, z25.d\n"
        ".else\n"
        "fadd z15.d, z15.d, z25.d\n"
        ".endif\n"
        "st1d {z15.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z16.d, z16.d, z26.d\n"
        ".else\n"
        "fadd z16.d, z16.d, z26.d\n"
        ".endif\n"
        "st1d {z16.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z17.d, z17.d, z27.d\n"
        ".else\n"
        "fadd z17.d, z17.d, z27.d\n"
        ".endif\n"
        "st1d {z17.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z20.d, z20.d, z24.d\n"
        ".else\n"
        "fadd z20.d, z20.d, z24.d\n"
        ".endif\n"
        "st1d {z20.d}, p0, [x19]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z21.d, z21.d, z25.d\n"
        ".else\n"
        "fadd z21.d, z21.d, z25.d\n"
        ".endif\n"
        "st1d {z21.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z22.d, z22.d, z26.d\n"
        ".else\n"
        "fadd z22.d, z22.d, z26.d\n"
        ".endif\n"
        "st1d {z22.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z23.d, z23.d, z27.d\n"
        ".else\n"
        "fadd z23.d, z23.d, z27.d\n"
        ".endif\n"
        "st1d {z23.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.lt 8f\n"
        "1:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z2.d, z2.d, z24.d\n"
        ".else\n"
        "fadd z2.d, z2.d, z24.d\n"
        ".endif\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z3.d, z3.d, z25.d\n"
        ".else\n"
        "fadd z3.d, z3.d, z25.d\n"
        ".endif\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z4.d, z4.d, z26.d\n"
        ".else\n"
        "fadd z4.d, z4.d, z26.d\n"
        ".endif\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z5.d, z5.d, z27.d\n"
        ".else\n"
        "fadd z5.d, z5.d, z27.d\n"
        ".endif\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "ld1d {z6.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "ld1d {z7.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z8.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z8.d, z8.d, z24.d\n"
        ".else\n"
        "fadd z8.d, z8.d, z24.d\n"
        ".endif\n"
        "st1d {z8.d}, p0, [x19]\n"
        "ld1d {z9.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z9.d, z9.d, z25.d\n"
        ".else\n"
        "fadd z9.d, z9.d, z25.d\n"
        ".endif\n"
        "st1d {z9.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z10.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z10.d, z10.d, z26.d\n"
        ".else\n"
        "fadd z10.d, z10.d, z26.d\n"
        ".endif\n"
        "st1d {z10.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z11.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z11.d, z11.d, z27.d\n"
        ".else\n"
        "fadd z11.d, z11.d, z27.d\n"
        ".endif\n"
        "st1d {z11.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "ld1d {z12.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "ld1d {z13.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z14.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z14.d, z14.d, z24.d\n"
        ".else\n"
        "fadd z14.d, z14.d, z24.d\n"
        ".endif\n"
        "st1d {z14.d}, p0, [x19]\n"
        "ld1d {z15.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z15.d, z15.d, z25.d\n"
        ".else\n"
        "fadd z15.d, z15.d, z25.d\n"
        ".endif\n"
        "st1d {z15.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z16.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z16.d, z16.d, z26.d\n"
        ".else\n"
        "fadd z16.d, z16.d, z26.d\n"
        ".endif\n"
        "st1d {z16.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z17.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z17.d, z17.d, z27.d\n"
        ".else\n"
        "fadd z17.d, z17.d, z27.d\n"
        ".endif\n"
        "st1d {z17.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "ld1d {z18.d}, p0/z, [x9]\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "ld1d {z19.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z20.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z20.d, z20.d, z24.d\n"
        ".else\n"
        "fadd z20.d, z20.d, z24.d\n"
        ".endif\n"
        "st1d {z20.d}, p0, [x19]\n"
        "ld1d {z21.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z21.d, z21.d, z25.d\n"
        ".else\n"
        "fadd z21.d, z21.d, z25.d\n"
        ".endif\n"
        "st1d {z21.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z22.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z22.d, z22.d, z26.d\n"
        ".else\n"
        "fadd z22.d, z22.d, z26.d\n"
        ".endif\n"
        "st1d {z22.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z23.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z23.d, z23.d, z27.d\n"
        ".else\n"
        "fadd z23.d, z23.d, z27.d\n"
        ".endif\n"
        "st1d {z23.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "sub x11, x11, #4\n"
        "cmp x11, #4\n"
        "b.ge 1b\n"
        "8:\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "fmopa za0.d, p0/m, p0/m, z6.d, z8.d\n"
        "fmopa za1.d, p0/m, p0/m, z6.d, z9.d\n"
        "fmopa za2.d, p0/m, p0/m, z6.d, z10.d\n"
        "fmopa za3.d, p0/m, p0/m, z6.d, z11.d\n"
        "fmopa za4.d, p0/m, p0/m, z7.d, z8.d\n"
        "fmopa za5.d, p0/m, p0/m, z7.d, z9.d\n"
        "fmopa za6.d, p0/m, p0/m, z7.d, z10.d\n"
        "fmopa za7.d, p0/m, p0/m, z7.d, z11.d\n"
        "fmopa za0.d, p0/m, p0/m, z12.d, z14.d\n"
        "fmopa za1.d, p0/m, p0/m, z12.d, z15.d\n"
        "fmopa za2.d, p0/m, p0/m, z12.d, z16.d\n"
        "fmopa za3.d, p0/m, p0/m, z12.d, z17.d\n"
        "fmopa za4.d, p0/m, p0/m, z13.d, z14.d\n"
        "fmopa za5.d, p0/m, p0/m, z13.d, z15.d\n"
        "fmopa za6.d, p0/m, p0/m, z13.d, z16.d\n"
        "fmopa za7.d, p0/m, p0/m, z13.d, z17.d\n"
        "fmopa za0.d, p0/m, p0/m, z18.d, z20.d\n"
        "fmopa za1.d, p0/m, p0/m, z18.d, z21.d\n"
        "fmopa za2.d, p0/m, p0/m, z18.d, z22.d\n"
        "fmopa za3.d, p0/m, p0/m, z18.d, z23.d\n"
        "fmopa za4.d, p0/m, p0/m, z19.d, z20.d\n"
        "fmopa za5.d, p0/m, p0/m, z19.d, z21.d\n"
        "fmopa za6.d, p0/m, p0/m, z19.d, z22.d\n"
        "fmopa za7.d, p0/m, p0/m, z19.d, z23.d\n"
        "cbz x11, 9f\n"
        "7:\n"
        "ld1d {z0.d}, p0/z, [x9]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10]\n"
        "ld1d {z24.d}, p0/z, [x6]\n"
        ".if %c[subtract]\n"
        "fsub z2.d, z2.d, z24.d\n"
        ".else\n"
        "fadd z2.d, z2.d, z24.d\n"
        ".endif\n"
        "st1d {z2.d}, p0, [x19]\n"
        "ld1d {z3.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z25.d}, p0/z, [x6, #1, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z3.d, z3.d, z25.d\n"
        ".else\n"
        "fadd z3.d, z3.d, z25.d\n"
        ".endif\n"
        "st1d {z3.d}, p0, [x19, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x10, #2, mul vl]\n"
        "ld1d {z26.d}, p0/z, [x6, #2, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z4.d, z4.d, z26.d\n"
        ".else\n"
        "fadd z4.d, z4.d, z26.d\n"
        ".endif\n"
        "st1d {z4.d}, p0, [x19, #2, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x10, #3, mul vl]\n"
        "ld1d {z27.d}, p0/z, [x6, #3, mul vl]\n"
        ".if %c[subtract]\n"
        "fsub z5.d, z5.d, z27.d\n"
        ".else\n"
        "fadd z5.d, z5.d, z27.d\n"
        ".endif\n"
        "st1d {z5.d}, p0, [x19, #3, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, x20\n"
        "add x6, x6, x20\n"
        "add x19, x19, #256\n"
        "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n"
        "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n"
        "fmopa za2.d, p0/m, p0/m, z0.d, z4.d\n"
        "fmopa za3.d, p0/m, p0/m, z0.d, z5.d\n"
        "fmopa za4.d, p0/m, p0/m, z1.d, z2.d\n"
        "fmopa za5.d, p0/m, p0/m, z1.d, z3.d\n"
        "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n"
        "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n"
        "subs x11, x11, #1\n"
        "b.ne 7b\n"
        "9:\n"
        "ldr x17, [x28]\n"
        "sub x17, x17, x27\n"
        "lsl x6, x17, #12\n"
        "ldr x9, [x28, #96]\n"
        "add x9, x9, x6\n"
        "ldr x10, [x28, #104]\n"
        "add x10, x10, x6\n"
        "ldr x11, [x28, #112]\n"
        "add x11, x11, x6\n"
        "ldr x13, [x28, #120]\n"
        "add x13, x13, x6\n"
        "ldr x14, [x28, #128]\n"
        "add x14, x14, x6\n"
        "ldr x15, [x28, #136]\n"
        "add x15, x15, x6\n"
        "mov x5, #56\n"
        "madd x16, x17, x5, x28\n"
        "add x16, x16, #144\n"
        "ldr x4, [x16, #0]\n"
        "ldr x5, [x16, #8]\n"
        "ldr x6, [x16, #16]\n"
        "ldr x17, [x16, #32]\n"
        "ldr x19, [x16, #40]\n"
        "ldr x20, [x16, #48]\n"
        "ldr x16, [x16, #24]\n"
        "mov w12, #0\n"
        "300:\n"
        "mova z0.d, p0/m, za0v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #0, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #0, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #0, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #0, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #0, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #0, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #0, mul vl]\n"
        "mova z0.d, p0/m, za4v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #1, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #1, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #1, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #1, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #1, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #1, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #1, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add x4, x4, #128\n"
        "add x5, x5, #128\n"
        "add x6, x6, #128\n"
        "add x16, x16, #128\n"
        "add x17, x17, #128\n"
        "add x19, x19, #128\n"
        "add x20, x20, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 300b\n"
        "mov w12, #0\n"
        "301:\n"
        "mova z0.d, p0/m, za1v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #0, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #0, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #0, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #0, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #0, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #0, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #0, mul vl]\n"
        "mova z0.d, p0/m, za5v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #1, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #1, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #1, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #1, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #1, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #1, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #1, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add x4, x4, #128\n"
        "add x5, x5, #128\n"
        "add x6, x6, #128\n"
        "add x16, x16, #128\n"
        "add x17, x17, #128\n"
        "add x19, x19, #128\n"
        "add x20, x20, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 301b\n"
        "mov w12, #0\n"
        "302:\n"
        "mova z0.d, p0/m, za2v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #0, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #0, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #0, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #0, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #0, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #0, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #0, mul vl]\n"
        "mova z0.d, p0/m, za6v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #1, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #1, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #1, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #1, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #1, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #1, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #1, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add x4, x4, #128\n"
        "add x5, x5, #128\n"
        "add x6, x6, #128\n"
        "add x16, x16, #128\n"
        "add x17, x17, #128\n"
        "add x19, x19, #128\n"
        "add x20, x20, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 302b\n"
        "mov w12, #0\n"
        "303:\n"
        "mova z0.d, p0/m, za3v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #0, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #0, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #0, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #0, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #0, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #0, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #0, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #0, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #0, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #0, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #0, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #0, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #0, mul vl]\n"
        "mova z0.d, p0/m, za7v.d[w12, 0]\n"
        "ld1d {z1.d}, p0/z, [x9, #1, mul vl]\n"
        "ld1d {z2.d}, p0/z, [x10, #1, mul vl]\n"
        "ld1d {z3.d}, p0/z, [x11, #1, mul vl]\n"
        "ld1d {z4.d}, p0/z, [x13, #1, mul vl]\n"
        "ld1d {z5.d}, p0/z, [x14, #1, mul vl]\n"
        "ld1d {z6.d}, p0/z, [x15, #1, mul vl]\n"
        "fadd z16.d, z0.d, z3.d\n"
        "fsub z16.d, z16.d, z4.d\n"
        "fadd z16.d, z16.d, z6.d\n"
        "fadd z17.d, z2.d, z4.d\n"
        "fadd z18.d, z1.d, z3.d\n"
        "fsub z19.d, z0.d, z1.d\n"
        "fadd z19.d, z19.d, z2.d\n"
        "fadd z19.d, z19.d, z5.d\n"
        "fadd z20.d, z16.d, z19.d\n"
        "st1d {z20.d}, p0, [x4, #1, mul vl]\n"
        "fadd z20.d, z17.d, z19.d\n"
        "st1d {z20.d}, p0, [x5, #1, mul vl]\n"
        "st1d {z16.d}, p0, [x6, #1, mul vl]\n"
        "st1d {z19.d}, p0, [x16, #1, mul vl]\n"
        "fadd z20.d, z16.d, z18.d\n"
        "st1d {z20.d}, p0, [x17, #1, mul vl]\n"
        "fsub z20.d, z17.d, z16.d\n"
        "st1d {z20.d}, p0, [x19, #1, mul vl]\n"
        "fsub z20.d, z18.d, z19.d\n"
        "st1d {z20.d}, p0, [x20, #1, mul vl]\n"
        "add x9, x9, #128\n"
        "add x10, x10, #128\n"
        "add x11, x11, #128\n"
        "add x13, x13, #128\n"
        "add x14, x14, #128\n"
        "add x15, x15, #128\n"
        "add x4, x4, #128\n"
        "add x5, x5, #128\n"
        "add x6, x6, #128\n"
        "add x16, x16, #128\n"
        "add x17, x17, #128\n"
        "add x19, x19, #128\n"
        "add x20, x20, #128\n"
        "add w12, w12, #1\n"
        "cmp w12, #8\n"
        "b.ne 303b\n"
        "ldr x9, [x28, #56]\n"
        "add x23, x23, x9\n"
        "add x7, x7, x9\n"
        "add x24, x24, #4096\n"
        "ldr x9, [x28, #64]\n"
        "add x25, x25, x9\n"
        "ldr x9, [x28, #72]\n"
        "add x26, x26, x9\n"
        "subs x27, x27, #1\n"
        "b.ne 90b\n"
        "smstop\n"
        : : [descriptor]"r"(&d), [subtract]"i"(subtract), [pf0]"i"(0+0), [pf1]"i"(0+64), [pf2]"i"(0+128), [pf3]"i"(0+192) : "memory","x4","cc","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x19","x20","x21","x22","x23","x24","x25","x26","x27","x28","z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15","z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}

// Raw C is [U][R=2][S=8000]. Transpose the four R/S quadrants and
// form four independent RHS basis operands while the values are still in ZA/Z.
// One call covers U32 x K (320 or 160); only one streaming-mode entry.
// Four ZA 8x8 tiles hold a U8/K8 slice. Input+output Kr8/U32 is 16 KiB.
struct RhsPackDescriptor {
    const double *source;
    double *output[7];
    long k;
};
static_assert(offsetof(RhsPackDescriptor,k)==64,"RHS pack descriptor");
__attribute__((noinline,aligned(64))) static void sme_rhs_pack(const RhsPackDescriptor &d) {
    asm volatile(
        ".arch_extension sme-f64f64\n"
        "mov x28,%[d]\nldr x19,[x28]\nldr x20,[x28,#64]\nlsr x20,x20,#3\n"
        "mov x21,#16000\nlsl x21,x21,#3\nmov x22,#64000\nmov x23,#32000\n"
        "mov x24,#0\n"
        "smstart\nptrue p0.d\n"
        // x19 points to the current K8 slice; x24 is its output byte offset.
        "mov x25,#0\n"
        "1:\nldr x19,[x28]\nlsl x26,x25,#3\nmadd x19,x26,x21,x19\n"
        "ldr x20,[x28,#64]\nlsr x20,x20,#3\nmov x24,#0\n"
        "2:\n"
        // Each U8 increment is eight raw C rows (8 * 128000 bytes).
        "mov x9,x19\n"
        "add x10,x9,x22\nadd x11,x9,x23\nadd x13,x10,x23\n"
        "mov w12,#0\n"
        "3:\n"
        "ld1d {za0h.d[w12,0]},p0/z,[x9]\n"
        "ld1d {za1h.d[w12,0]},p0/z,[x10]\n"
        "ld1d {za2h.d[w12,0]},p0/z,[x11]\n"
        "ld1d {za3h.d[w12,0]},p0/z,[x13]\n"
        "add x9,x9,x21\nadd x10,x10,x21\nadd x11,x11,x21\nadd x13,x13,x21\n"
        "add w12,w12,#1\ncmp w12,#8\nb.ne 3b\n"
        "add x26,x24,x25,lsl #6\n"
        "ldr x9,[x28,#8]\nldr x10,[x28,#16]\nldr x11,[x28,#24]\n"
        "ldr x13,[x28,#32]\n"
        "add x9,x9,x26\nadd x10,x10,x26\nadd x11,x11,x26\nadd x13,x13,x26\n"
        "mov w12,#0\n"
        "4:\n"
        "mova z0.d,p0/m,za0v.d[w12,0]\n"
        "mova z1.d,p0/m,za1v.d[w12,0]\n"
        "mova z2.d,p0/m,za2v.d[w12,0]\n"
        "mova z3.d,p0/m,za3v.d[w12,0]\n"
        "fadd z4.d,z0.d,z3.d\nst1d {z4.d},p0,[x9]\n"
        "fsub z4.d,z1.d,z3.d\nst1d {z4.d},p0,[x10]\n"
        "fsub z4.d,z2.d,z0.d\nst1d {z4.d},p0,[x11]\n"
        "st1d {z3.d},p0,[x13]\n"
        "add x9,x9,#256\nadd x10,x10,#256\nadd x11,x11,#256\nadd x13,x13,#256\n"
        "add w12,w12,#1\ncmp w12,#8\nb.ne 4b\n"
        "add x19,x19,#64\nadd x24,x24,#2048\nsubs x20,x20,#1\nb.ne 2b\n"
        "add x25,x25,#1\ncmp x25,#4\nb.ne 1b\nsmstop\n"
        ::[d]"r"(&d)
        :"memory","cc","x9","x10","x11","x12","x13","x14","x15","x16",
         "x19","x20","x21","x22","x23","x24","x25","x26","x28",
         "z0","z1","z2","z3","z4","z5","z6","z7","z8","z9","z10","z11","z12","z13","z14","z15",
         "z16","z17","z18","z19","z20","z21","z22","z23","z24","z25","z26","z27","z28","z29","z30","z31",
         "p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15");
}

// TC2-only C edge4->5 becomes6->5. All transfer/epoch logic matches the original helper.
static void tc2_c_relay_transfer(double *dst, size_t bytes, size_t metadata, Sdma &dma, const std::vector<u32> &pids,
                                  uintptr_t root, uintptr_t relay, uintptr_t relay1, int rank, MPI_Comm comm,
                                  void (*progress)(void*)=nullptr, void* context=nullptr) {
    const size_t chunks = (bytes + c_ready_chunk - 1) / c_ready_chunk;
    if (rank == 0)
        std::abort();
    const int owner = rank == 5 ? 6 : rank == 2 || rank == 3 ? 1 : rank == 7 ? 6 : rank >= 5 ? 4 : 0;
    if ((owner == 6 || owner == 2) && !tree_pools_known)
        std::abort();
    const uintptr_t source = owner == 1                 ? relay1
                             : owner == 4               ? relay
                             : owner == 6 || owner == 2 ? tree_pools[owner]
                                                        : root;
    auto *snapshot = (u64 *)((char *)dst + metadata + chunks * 64);
    for (size_t c = 0; c < chunks; ++c) {
        Sdma &active = dma;

        const double deadline = tc_seconds() + 30;
        for (;;) {
            if(progress)progress(context);
            hisi_sdma_sqe_task flag{};
            flag.src_addr = source + metadata + c * 64;
            flag.dst_addr = (uintptr_t)snapshot;
            flag.src_process_id = pids[owner];
            flag.dst_process_id = active.process;
            flag.length = 64;
            active.submit_task(flag);
            active.wait();

            const u64 ready = *(volatile u64 *)snapshot;
            if (ready == c_epoch) {
                break;
            }
            if (ready > c_epoch)
                MPI_Abort(comm, 5);
            if (tc_seconds() > deadline)
                MPI_Abort(comm, 5);
        }

        const size_t offset = c * c_ready_chunk;
        hisi_sdma_sqe_task task{};
        task.src_addr = source + offset;
        task.dst_addr = (uintptr_t)dst + offset;
        task.src_process_id = pids[owner];
        task.dst_process_id = active.process;
        task.length = std::min(c_ready_chunk, bytes - offset);
        active.submit_task(task);

        active.wait();
        {
            asm volatile("dmb oshst" ::: "memory");
            __atomic_store_n((u64 *)((char *)dst + metadata + c * 64), u64(c_epoch), 3);
        }
    }
}

// Layout in doubles: [U group][K step][4 basis operands][N32][K][32].
// Each consumer's N128/K320 panel is contiguous and uses the unchanged kernel.
static constexpr tbstrassen::Recipe rhs_recipe[7]={{0,-1,0},{0,3,-1},{1,-1,0},{2,-1,0},{3,-1,0},{0,1,1},{0,2,1}};
static size_t rhs_offset(int group,int product,int step) {
    const int width=std::min(128,4000-group*128),k=std::min(320,4000-step*320);
    return size_t(group)*128*4000*4+size_t(step)*320*width*4+size_t(product)*width*k;
}
struct Tc2RhsBroadcast {
    static constexpr size_t bytes=512000000,metadata=bytes;
    static constexpr int chunks=(bytes+c_ready_chunk-1)/c_ready_chunk,tasks=32*13;
    PrepAcBroadcast control;
    Sdma &dma;
    const std::vector<u32> &pids;
    std::array<uintptr_t,8> pools{};
    std::array<CReadyCounter,chunks> counters;
    static_assert(bytes+(chunks+2)*64<=CommPool::large,"RHS metadata capacity");
    Tc2RhsBroadcast(MPI_Comm comm,int rank):control(comm,rank,true),dma(control.dma),pids(control.pids) {
        pools=tree_pools;
        if(rank==0){
            std::array<int,chunks> pending{};
            for(int id=0;id<tasks;++id){
                const size_t begin=rhs_offset(id/13,0,id%13)*8;
                const int width=std::min(128,4000-id/13*128),k=std::min(320,4000-id%13*320);
                const size_t end=begin+size_t(4)*width*k*8;
                for(size_t c=begin/c_ready_chunk;c<=(end-1)/c_ready_chunk;++c)++pending[c];
            }
            for(int c=0;c<chunks;++c){if(!pending[c])std::abort();counters[c].left.store(pending[c],std::memory_order_relaxed);}
        }
    }
    void pack(unsigned id,const double* input){
        const int group=id/13,step=id%13,u0=group*128,width=std::min(128,4000-u0),k=std::min(320,4000-step*320);
        for(int n=0;n<width;n+=32){
            RhsPackDescriptor d{input+size_t(u0+n)*16000+step*320,{},k};
            for(int z=0;z<4;++z)d.output[z]=comm_pool->c+rhs_offset(group,z,step)+size_t(n)*k;
            sme_rhs_pack(d);
        }
        const size_t begin=rhs_offset(group,0,step)*8,end=begin+size_t(4)*width*k*8;
        for(size_t c=begin/c_ready_chunk;c<=(end-1)/c_ready_chunk;++c)
            if(counters[c].left.fetch_sub(1,std::memory_order_acq_rel)==1){

                asm volatile("dmb oshst":::"memory");
                __atomic_store_n((u64*)((char*)comm_pool->c+metadata+c*64),u64(c_epoch),__ATOMIC_RELEASE);
            }
    }
    void receive(int rank,MPI_Comm comm){
        tc2_c_relay_transfer(comm_pool->c,bytes,metadata,dma,pids,pools[0],pools[4],pools[1],rank,comm,[](void*){},nullptr);
    }
};

// T/R/S Strassen. Each product is dense P x U;
// R-separated C quadrants use the existing fused sum-pack first M16 kernel.
// All input transforms and product combinations run inside the timed entry.
struct Tc2AcStrassenTeam {
    static constexpr int panel=128,groups=(4000+panel-1)/panel,tasks=groups*7;
    const int logical,m;
    const double *wire;
    double *target;
    static constexpr int jobs=groups*4;
    static constexpr int leaders[4]={0,2,3,4},follower[7]={-1,-1,5,6,1,-1,-1};
    KFirstQueue<2> queue{jobs,5};
    unsigned order[jobs];
    std::atomic<int> left[groups],remaining{tasks};
    // A bit covers all seven TB operands and all P rows of one U128 group.
    // The last group covers U3968..3999. RMW publication joins prior writers.
    alignas(64) std::atomic<uint32_t> completed_panels{0};
    void publish_panel(int group){

        const uint32_t bit=uint32_t(1)<<group;
        if(completed_panels.fetch_or(bit,std::memory_order_acq_rel)&bit)std::abort();
    }
    Tc2AcStrassenTeam(int pp,const double* c,double* t):logical(pp),m((pp+15)/16*16),wire(c),target(t){
        for(int i=0;i<jobs;++i)order[i]=i/4*7+leaders[i%4];
        if(size_t(tasks)*m*panel*8>tc2_ac_product_bytes)std::abort();
        for(auto& x:left)x.store(7,std::memory_order_relaxed);
    }
    // Rows are P16. A's two row quadrants are t0/t1, its K quadrants are S0/S1.
    void prepare(int block,const double* lo,long ls,const double* hi,long hs){
        const int row=block*16;

        alignas(64) const double zero[16]={};
        if(!lo){lo=zero;ls=0;}if(!hi){hi=zero;hs=0;}
        const svbool_t pg=svptrue_b64();
        for(int k0=0;k0<4000;k0+=320){
            const int k=std::min(320,4000-k0);
            double* d[7];
            for(int z=0;z<7;++z)d[z]=tc2_ac_operand(z,m)+size_t(k0)*m+size_t(row)*k;
            for(int kk=0;kk<k;++kk)for(int half=0;half<2;++half){
                const int si=k0+kk,j=kk*16+half*8;
                const double* src=half?hi:lo;const long stride=half?hs:ls;
                const svfloat64x2_t s0=svld2_f64(pg,src+si*stride),s1=svld2_f64(pg,src+(si+4000)*stride);
                const svfloat64_t a00=svget2_f64(s0,0),a01=svget2_f64(s1,0),a10=svget2_f64(s0,1),a11=svget2_f64(s1,1);
                svst1_f64(pg,d[0]+j,svadd_f64_x(pg,a00,a11));
                svst1_f64(pg,d[1]+j,svadd_f64_x(pg,a10,a11));
                svst1_f64(pg,d[2]+j,a00);svst1_f64(pg,d[3]+j,a11);
                svst1_f64(pg,d[4]+j,svadd_f64_x(pg,a00,a01));
                svst1_f64(pg,d[5]+j,svsub_f64_x(pg,a10,a00));
                svst1_f64(pg,d[6]+j,svsub_f64_x(pg,a01,a11));
            }
        }
    }
    void step(int worker,int item,int step,bool inplace=false){
        const int group=item/7,which=item%7,u0=group*panel,width=std::min(panel,4000-group*panel);
        const int k0=step*320,k=std::min(320,4000-k0),nr=m/16;

        if(which==0&&step==12){terminal(worker,group);return;}
        const bool boundary=which==0&&step==12;
        {

        double* packed=scratch[worker].b;
        double* c=tc2_ac_products+size_t(item)*m*panel;
        std::array<const double*,64> rows;
        for(int i=0;i<nr;++i)rows[i]=tc2_ac_operand(which,m)+size_t(k0)*m+size_t(i)*16*k;
        SmeNPanel first{width/32,k,rows[0],packed,c,packed,rows[1],long(k)*32*8,long(k)*32*8,k*16*8/(width/32),step!=0};
        {

            const auto recipe=rhs_recipe[which];
            auto ptr=[&](int q){return wire+rhs_offset(group,q,step);};
            first.b=ptr(recipe.x);first.b_stride=long(k)*32*8;
            if(inplace){
                SmeSumPanel sum{first,packed};
                if(recipe.sign==1)tc1_rhs_sum_lookahead<false,true>(sum);
                else tc1_rhs_sum_lookahead<true,true>(sum);
            }else if(recipe.y<0)tc1_rhs_pack_lookahead(first);
            else {
                SmeSumPanel sum{first,ptr(recipe.y)};
                if(recipe.sign==1)tc1_rhs_sum_lookahead<false>(sum);
                else tc1_rhs_sum_lookahead<true>(sum);
            }
        }
        {

        SmeNPanel mid{width/32,k,rows[1],packed,c+16*width,packed,rows[2],long(k)*32*8,long(k)*32*8,k*16*8/(width/32),step!=0};
        SmeMRun run{{mid,c+32*width},rows.data()+1,nr-2};
        if(width==32)sme_mrun_ac_n32(run);else
        sme_mrun_accumulate(run);
        SmeNPanel last{width/32,k,rows[nr-1],packed,c+(m-16)*width,packed,nullptr,long(k)*32*8,long(k)*32*8,k*16*8/(width/32),step!=0};
        sme_npanel_pipeline(last);
        }
        }
        if(step==12){
            if(left[group].fetch_sub(1,std::memory_order_acq_rel)==1)combine(group);
            remaining.fetch_sub(1,std::memory_order_release);
        }
    }
    void terminal(int worker,int group){
        const int width=std::min(panel,4000-group*panel),u0=group*panel,k=160,k0=3840;

        if(left[group].load(std::memory_order_acquire)!=1)std::abort();
        const size_t stride=size_t(m)*panel;
        const double* base=tc2_ac_products+size_t(group)*7*stride;
        double* packed=scratch[worker].b;
        {
          std::memcpy(packed,wire+rhs_offset(group,0,12),size_t(width)*k*8); }
        for(int row=0;row<m;row+=16){
            const double* a=tc2_ac_operand(0,m)+size_t(k0)*m+size_t(row)*k;
            const double* future=row+16<m?a+16*k:nullptr;
            SmeNPanel n{width/32,k,a,packed,const_cast<double*>(base)+size_t(row)*width,
                        packed,future,k*32*8,k*32*8,k*16*8/(width/32),1};
            SmeAcTerminalPanel d{{n,nullptr},{},{}};
            for(int z=0;z<6;++z)d.other[z]=base+size_t(z+1)*stride+size_t(row)*width;
            for(int nb=0;nb<width/32;++nb)for(int z=0;z<7;++z)
                d.output[nb][z]=target+tc2_strassen_index(z,row,u0+nb*32);
            if(future)sme_ac_terminal_prefetch(d);
            else sme_ac_terminal_plain(d);
        }
        if(left[group].fetch_sub(1,std::memory_order_acq_rel)!=1)std::abort();
        publish_panel(group);
        remaining.fetch_sub(1,std::memory_order_release);
    }
    // AC quadrants are (t,r), TB quadrants are (r,t). Each output is now
    // written at the same P/U location, avoiding four P/U output quadrants.
    void combine(int group){
        const int width=std::min(panel,4000-group*panel),u0=group*panel;

        const size_t stride=size_t(m)*panel;
        const double* base=tc2_ac_products+size_t(group)*7*stride;
        const svbool_t pg=svptrue_b64();
        for(int row=0;row<m;row+=16)for(int nb=0;nb<width/32;++nb){
            double* out[7];
            for(int z=0;z<7;++z)out[z]=target+tc2_strassen_index(z,row,u0+nb*32);
            for(int u=0;u<32;++u)for(int lane=0;lane<16;lane+=8){
                const size_t at=size_t(row)*width+nb*512+u*16+lane;
                const svfloat64_t m1=svld1_f64(pg,base+at),m2=svld1_f64(pg,base+stride+at),
                    m3=svld1_f64(pg,base+2*stride+at),m4=svld1_f64(pg,base+3*stride+at),
                    m5=svld1_f64(pg,base+4*stride+at),m6=svld1_f64(pg,base+5*stride+at),m7=svld1_f64(pg,base+6*stride+at);
                const svfloat64_t x00=svadd_f64_x(pg,svsub_f64_x(pg,svadd_f64_x(pg,m1,m4),m5),m7),
                    x01=svadd_f64_x(pg,m3,m5),x10=svadd_f64_x(pg,m2,m4),
                    x11=svadd_f64_x(pg,svadd_f64_x(pg,svsub_f64_x(pg,m1,m2),m3),m6);
                const int j=u*16+lane;
                svst1_f64(pg,out[0]+j,svadd_f64_x(pg,x00,x11));
                svst1_f64(pg,out[1]+j,svadd_f64_x(pg,x01,x11));
                svst1_f64(pg,out[2]+j,x00);svst1_f64(pg,out[3]+j,x11);
                svst1_f64(pg,out[4]+j,svadd_f64_x(pg,x00,x10));
                svst1_f64(pg,out[5]+j,svsub_f64_x(pg,x01,x00));
                svst1_f64(pg,out[6]+j,svsub_f64_x(pg,x10,x11));
            }
        }
        publish_panel(group);
    }
    bool ready(unsigned item,int batch)const{
        const int group=item/7,which=item%7,width=std::min(128,4000-group*128);
        const auto r=rhs_recipe[which];
        for(int step=batch*3;step<std::min(13,(batch+1)*3);++step)for(int basis:{r.x,r.y}){
            if(basis<0)continue;
            const size_t lo=rhs_offset(group,basis,step)*8;
            const size_t hi=lo+size_t(width)*std::min(320,4000-step*320)*8;
            for(size_t chunk=lo/c_ready_chunk;chunk<=(hi-1)/c_ready_chunk;++chunk)
                if(__atomic_load_n((u64*)((char*)wire+Tc2RhsBroadcast::metadata+chunk*64),__ATOMIC_ACQUIRE)!=c_epoch)return false;
        }
        return true;
    }
    bool pair_ready(unsigned item,int batch)const{
        const int child=follower[item%7];
        return ready(item,batch) && (child<0 || ready(item-item%7+child,batch));
    }
    void pair_step(int worker,unsigned item,int k){
        step(worker,item,k);
        const int child=follower[item%7];
        if(child>=0)step(worker,item-item%7+child,k,true);
    }
    // One batch of the unchanged K-first queue, with a nonblocking no-work
    // return so that temporarily idle workers can consume ready TB panels.
    bool try_work(int worker){
        if(queue.final_claimed.load(std::memory_order_acquire)>=jobs)return false;
        for(int batch=0;batch<5;++batch)for(int h=0;h<2;++h){
            auto& bank=queue.banks[batch].bits[h];
            uint64_t word=bank.load(std::memory_order_relaxed);
            while(word){
                uint64_t eligible=word;
                while(eligible){
                    const unsigned item=order[h*64+__builtin_ctzll(eligible)];
                    if(pair_ready(item,batch) && (batch!=4 || item%7!=0 ||
                        left[item/7].load(std::memory_order_acquire)==1))break;
                    eligible&=eligible-1;
                }
                if(!eligible)break;
                const unsigned bit=__builtin_ctzll(eligible);
                const uint64_t mask=uint64_t(1)<<bit;
                if(!bank.compare_exchange_weak(word,word&~mask,std::memory_order_acquire,
                                               std::memory_order_relaxed))continue;
                if(batch==4)queue.final_claimed.fetch_add(1,std::memory_order_release);
                for(int k=batch*3;k<std::min(13,(batch+1)*3);++k)pair_step(worker,order[h*64+bit],k);
                if(batch<4)queue.banks[batch+1].bits[h].fetch_or(mask,std::memory_order_release);
                return true;
            }
        }
        return false;
    }

    void work(int worker){
        // Four K960 batches, then K160. M1 waits for the other six products.
        queue.drain_terminal(order,[&](unsigned item,int batch){
            for(int k=batch*3;k<std::min(13,(batch+1)*3);++k)pair_step(worker,item,k);
        },[&](unsigned item,int batch){
            return pair_ready(item,batch)&&(batch!=4||item%7!=0||left[item/7].load(std::memory_order_acquire)==1);
        });
    }
};

static void tc2_prep_ac_fused(std::vector<double> &a, std::vector<double> &b, std::vector<double> &c,
                              std::vector<double> &out, MPI_Comm comm, int rank) {
    if (omp_get_max_threads() != 38 || svcntd() != 8 || a.size() != 16000000 || b.size() != 4000000 ||
        c.size() != 64000000)
        std::abort();
    AcWorkload workload{};

    // own p first, then donated groups; 16-row AC blocks hold 8 p x 2 t, donors keep the 2000-row A slabs
    // sharing needs the row-64 A packing and the streamed fused pipeline; every rank sees the same settings
    const bool sharing = true;
    const ShareRole share = sharing ? share_plan(tc2_share_groups, tc2_share_cap, rank) : ShareRole{};
    const int own = 1000 - 8 * share.give, P = own + 8 * share.take;
    const int blocks = P / 8, tiles = 63 * blocks;
    tc2_rows_stride = std::max(2000, 2 * P);
    tc2_zero_row = P % 16 == 8 ? 2 * P - 16 : -1;
    tc2_strassen_padded_p=(P+15)/16*16;

    PrepAcQueue cq, aq, bq, acq;
    for (int w = 0; w < 38; ++w) {
        cq.set(w, 0, rank == 0 ? (Tc2RhsBroadcast::tasks + 37 - w) / 38 : 0);
        aq.set(w, (tc2_strassen_padded_p/16) * w / 38, (tc2_strassen_padded_p/16) * (w + 1) / 38);
        bq.set(w, tc2_tb_rhs_tasks * w / 38, tc2_tb_rhs_tasks * (w + 1) / 38);
        unsigned start = workload.task_count;
        for (int tile = tiles * w / 38, end = tiles * (w + 1) / 38; tile < end;) {
            int panel = tile / blocks, stop = std::min(end, (panel + 1) * blocks);
            if (workload.task_count == 100)
                std::abort();
            workload.tasks[workload.task_count++] = {w, panel, (tile % blocks) * 16, (stop - tile) * 16};
            tile = stop;
        }
        acq.set(w, start, workload.task_count);
    }
    if (workload.task_count != 100 || acq.total != 100 || aq.total != unsigned(tc2_strassen_padded_p/16) || bq.total != tc2_tb_rhs_tasks)
        std::abort();
    acq.prepare_longest(workload.tasks);
    Tc2KFirstQueue kqueue;
    std::array<size_t, 101> c_offsets{};
    for (int id = 0; id < 100; ++id)
        c_offsets[id + 1] =
            c_offsets[id] + size_t(workload.tasks[id].rows) * (workload.tasks[id].panel == 62 ? 64 : 128);
    if (c_offsets[100] != size_t(2 * P) * 8000)
        std::abort();
    static_assert(CommPool::large - (384ul << 20) >= 128000000, "reuse pre-touched native C region");

    Tc2RhsBroadcast exchange(comm, rank);
    // donated A columns as [s][8 p][2 t] (donor -> helper); finished output rows [8 p][q*r] (helper -> donor)
    constexpr size_t a_group = 2 * ShareArea::group;
    std::array<uintptr_t, 8> pools{};
    unsigned spill_count = 0;
    std::atomic<int> share_pulled{0}, gathered_workers{0}, gather_next{0}, shared_ready{share.take ? 0 : 1};
    Output tb_out{out.data(), 8000, 2, 0, P};
    if (sharing) {
        pools = share_pools(comm);
        if (share.take) {
            share_authorize(exchange.pids);
            tb_out.spill = (double *)ShareArea::at(ShareArea::o_rows);
            tb_out.spill_row = 1000;
            tb_out.spill_count = &spill_count;
            tb_out.spill_flag = (unsigned long *)ShareArea::at(ShareArea::o_flag);
            tb_out.spill_tiles = 16;
            tb_out.spill_value = 2;
        }
    }
    auto share_prepare = [&](int worker) {
        if (share.give) {
            // 38 slices claimed dynamically, so a late worker (starved edge core) does not hold up the helpers
            double *gathered = (double *)ShareArea::at(ShareArea::a_out);
            for (int slice; (slice = gather_next.fetch_add(1, std::memory_order_relaxed)) < 38;) {
                for (int si = 8000 * slice / 38; si < 8000 * (slice + 1) / 38; ++si)
                    for (int g = 0; g < share.give; ++g)
                        memcpy(gathered + g * a_group / 8 + size_t(si) * 16,
                               a.data() + size_t(si) * 2000 + 2 * (own + 8 * g), 16 * sizeof(double));
                if (gathered_workers.fetch_add(1, std::memory_order_acq_rel) == 37) {
                    asm volatile("dmb oshst" ::: "memory");
                    __atomic_store_n((unsigned long *)ShareArea::at(ShareArea::a_flag), 2ul, __ATOMIC_RELEASE);
                }
            }
        }
        if (share.take && worker == 0) {
            for (int k = 0; k < share.take; ++k)
                ShareArea::pull(exchange.dma, exchange.pids[share.donor[k]], pools[share.donor[k]], ShareArea::a_flag,
                                2, ShareArea::a_out + share.group[k] * a_group, a_group,
                                ShareArea::at(ShareArea::a_in + k * a_group), comm);
            shared_ready.store(1, std::memory_order_release);
        }
    };
    // After local products and output slices finish, one donor worker pulls the
    // helpers' rows; the whole team copies them into out after joining.
    std::atomic<int> puller{0};
    auto pull_shared_out = [&]() {
        for (int g = 0; g < share.give; ++g) {
            const int h = share.helper[g];
            ShareArea::wait_flag(exchange.dma, exchange.pids[h], pools[h], ShareArea::o_flag, 2, comm);
        }
        for (int g = 0; g < share.give; ++g) {
            const int h = share.helper[g];
            ShareArea::submit_copy(exchange.dma, exchange.pids[h], pools[h],
                                   ShareArea::o_rows + share.slot[g] * ShareArea::group, ShareArea::group,
                                   ShareArea::at(ShareArea::o_in + g * ShareArea::group));
        }
        exchange.dma.wait();
        share_pulled.store(1, std::memory_order_release);
    };
    WorkBuffer all_b(8000ul * 512 * 8, true), temp(7ul * tc2_strassen_padded_p * 4000, false);
    std::unique_ptr<PersistentGather> pending_b;

    // B SDMA writes only [0,250 MiB); products own [256,480 MiB).
    // These upper bytes held ordinary-AC scratch in the previous implementation.
    static_assert(8000ul*512*8*8 <= (256ul<<20),"B gather must stay below AC products");
    static_assert((256ul<<20)+tc2_ac_product_bytes<=CommPool::large,"AC products fit B pool");
    tc2_ac_products=comm_pool->b_full+(256ul<<20)/8;
    Tc2StrassenTeam tb_team(temp.data(),all_b.data(),tb_out,tc2_strassen_padded_p);
    Tc2AcStrassenTeam ac_team(P,comm_pool->c,temp.data());
#pragma omp parallel num_threads(38)
    {
        const int worker = omp_get_thread_num();
        if (omp_get_num_threads() != 38)
            std::abort();
        auto &wr = workload.workers[worker];

        share_prepare(worker);
        if (rank == 0)
            cq.drain(worker, [&](int owner, unsigned id) {
                unsigned task = owner + 38 * id;

                exchange.pack(task, c.data());
            });
        else if (worker == 0) {

            exchange.receive(rank, comm);
        }
        bq.drain_with_publish(
            worker, [&](int, unsigned id) {  pack_tc2_tb_rhs_task(b.data(), id); }, [&]() {});
        aq.drain_with_publish(
            worker,
            [&](int, unsigned id) {
                {
                    if(share.take)while(!shared_ready.load(std::memory_order_acquire))asm volatile("yield":::"memory");
                    auto span=[&](int p)->std::pair<const double*,long>{
                        if(p>=P)return {nullptr,0};
                        if(p<1000)return {a.data()+p*2,2000};
                        const int j=(p-1000)/8;
                        return {(const double*)ShareArea::at(ShareArea::a_in+j*a_group),16};
                    };
                    const auto lo=span(id*16),hi=span(id*16+8);
                    ac_team.prepare(id,lo.first,lo.second,hi.first,hi.second);
                }
            },
            [&]() {});

        if (worker == 0) {
            while (!cq.done() || !aq.done() || !bq.done())
                asm volatile("yield" ::: "memory");

            {

            pending_b.reset(new PersistentGather(b, all_b, 8000 * 512, comm));
            }

        } else {
            while (!aq.done())
                asm volatile("yield" ::: "memory");
        }

        while(true){
            if(worker==0 && !tb_team.ready.load(std::memory_order_relaxed) && pending_b->progress()){

                tb_team.ready.store(1,std::memory_order_release);
            }
            if(ac_team.try_work(worker))continue;
            const bool ac_done=ac_team.remaining.load(std::memory_order_acquire)==0;
            const uint32_t panels=ac_team.completed_panels.load(std::memory_order_acquire);
            if(tb_team.try_product(worker,panels,ac_done))continue;
            if(tb_team.help_combine(worker))continue;
            if(ac_done && tb_team.remaining.load(std::memory_order_acquire)==0 &&
               tb_team.combine_remaining.load(std::memory_order_acquire)==0)break;
            asm volatile("yield":::"memory");
        }
        if(worker==0){
            pending_b->finish();
            if(share.give)while(!share_pulled.load(std::memory_order_acquire))
                asm volatile("yield":::"memory");
        }else if(share.give && !puller.exchange(1))pull_shared_out();
    }
    if (share.give)
        parallel_copy(out.data() + size_t(own) * 8000, (const double *)ShareArea::at(ShareArea::o_in),
                      share.give * ShareArea::group);
    if(ac_team.remaining.load()!=0 || ac_team.queue.final_claimed.load()!=Tc2AcStrassenTeam::jobs || !pending_b)std::abort();

    // Reuse removes 444 MiB of allocations; retain the 192 MiB temp tail.
}

// This header follows the shared TC2 terminal kernel definitions.

// S/R/T AC Strassen, matching the R/S TB transform.
// Included inside tcopt after shared kernel definitions; reuses existing buffers.
namespace ac1aligned {
// TC_T_SLOTS: TB operand ring slot of a K320 batch; five derived sums of 25 MiB per slot, four aliased operands of
// 10 MiB per slot (slots 0..7 in the temp pool, further slots after the derived ones)
inline int t_slot(int batch){return batch%tc_t_slots;}
inline double* t_derived(int batch){return tb_strassen_t_workspace+size_t(t_slot(batch))*10*1024*320;}
inline double* t_alias(int batch){
 const int slot=t_slot(batch);
 return slot<8?comm_pool->temp+size_t(slot)*4*1024*320:
  tb_strassen_t_workspace+size_t(tc_t_slots)*10*1024*320+size_t(slot-8)*4*1024*320;
}
constexpr int M=1024,groups=63;
constexpr size_t product=1024ul*128;
inline int width(int group){return group==62?64:128;}
template<bool last400> constexpr int steps(){return last400?6:7;}
template<bool last400> inline int depth(int step){return step==steps<last400>()-1?(last400?400:80):320;}
template<bool last400> inline size_t ci(int group,int basis,int step){
 const int n=width(group),k=depth<last400>(step);
 return size_t(group)*128*2000*4+size_t(step)*320*n*4+size_t(basis)*n*k;
}
static_assert(4ul*M*2000*8<=packed_a_capacity,"four A products fit packed A");
static_assert((80ul<<20)+3ul*M*2000*8<=(128ul<<20),"three A products after aliases");
static_assert((128ul<<20)+3ul*35*(1ul<<20)<=CommPool::temp_bytes,"three AC product slots");
inline double* operand(int which){return which<4?packed_a_workspace+size_t(which)*M*2000:
 comm_pool->temp+(80ul<<20)/8+size_t(which-4)*M*2000;}
inline double* ac_product(int group,int which){return comm_pool->temp+(128ul<<20)/8+
 size_t(group/5%3)*35*(1ul<<20)/8+size_t(group%5*7+which)*product;}
template<bool last400> static void prepare_a(const double* input,int task){
 const int hs=task/32,p=task%32*16;
 for(int step=0;step<steps<last400>();++step){
  const int k0=step*320,k=depth<last400>(step);
  double* d[7];for(int z=0;z<7;++z)d[z]=operand(z)+size_t(k0)*M+size_t(hs*512+p)*k;
  for(int kk=0;kk<k;++kk)for(int lane=0;lane<16;lane+=8){
   const svbool_t valid=svwhilelt_b64(long(p+lane),500L),all=svptrue_b64();
   const size_t lo=(size_t(hs)*4000+k0+kk)*500+p+lane,hi=lo+2ul*4000*500;
   const auto a00=svld1_f64(valid,input+lo),a01=svld1_f64(valid,input+lo+2000ul*500),
              a10=svld1_f64(valid,input+hi),a11=svld1_f64(valid,input+hi+2000ul*500);
   const int j=kk*16+lane;
   svst1_f64(all,d[0]+j,svadd_f64_x(all,a00,a11));
   svst1_f64(all,d[1]+j,svadd_f64_x(all,a10,a11));
   svst1_f64(all,d[2]+j,a00);svst1_f64(all,d[3]+j,a11);
   svst1_f64(all,d[4]+j,svadd_f64_x(all,a00,a01));
   svst1_f64(all,d[5]+j,svsub_f64_x(all,a10,a00));
   svst1_f64(all,d[6]+j,svsub_f64_x(all,a01,a11));
  }
 }
}
// ld4 separates the four R channels before forming four RHS bases. No new
// full-size array: the wire remains 64M doubles, including the final U32.
template<bool last400> static void prepare_c(const double* input,int group,int step){
 const int k0=step*320,k=depth<last400>(step),u0=group*64,nu=width(group)/2;
 const auto pg=svptrue_b64();
 for(int kk=0;kk<k;++kk)for(int nb=0;nb<nu;nb+=32)for(int j=0;j<32;j+=8){
  const double* lo=input+size_t(k0+kk)*16000+(u0+nb+j)*4;
  const auto l=svld4_f64(pg,lo),h=svld4_f64(pg,lo+2000ul*16000);
  const svfloat64_t a0=svget4_f64(l,0),a1=svget4_f64(l,1),a2=svget4_f64(l,2),a3=svget4_f64(l,3),
                    b0=svget4_f64(h,0),b1=svget4_f64(h,1),b2=svget4_f64(h,2),b3=svget4_f64(h,3);
  const size_t off=size_t(nb)*k+kk*32+j,off1=size_t(nu)*k+off;
  double* d[4];for(int z=0;z<4;++z)d[z]=comm_pool->c+ci<last400>(group,z,step);
  svst1_f64(pg,d[0]+off,svadd_f64_x(pg,a0,b2));svst1_f64(pg,d[0]+off1,svadd_f64_x(pg,a1,b3));
  svst1_f64(pg,d[1]+off,svsub_f64_x(pg,a2,b2));svst1_f64(pg,d[1]+off1,svsub_f64_x(pg,a3,b3));
  svst1_f64(pg,d[2]+off,svsub_f64_x(pg,b0,a0));svst1_f64(pg,d[2]+off1,svsub_f64_x(pg,b1,a1));
  svst1_f64(pg,d[3]+off,b2);svst1_f64(pg,d[3]+off1,b3);
 }
}
struct Outputs {
 double *aliases,*derived;
 double* at(int batch,int which,int hs,int row,int u){
  const int k=batch==12?160:320;
  const size_t offset=size_t(row)*k+size_t(u)*16;
  const int slot=tbstrassen::derived_slot[which];
  if(slot>=0)return t_derived(batch)+size_t(slot*2+hs)*1024*320+offset;
  return t_alias(batch)+size_t(hs*2+(which==3))*1024*320+offset;
 }
};
// follower >= 0 (TC_PAIR_PACK&8): after each step of which, the same step of the sum product follower, whose
// panel is X0 +- this product's packed single basis, formed in place in L2
template<bool last400> static void multiply(int worker,int group,int which,int follower,int first_step,int end_step){
 const int n=width(group),nr=M/16;
 auto one=[&](int z,int step,bool inplace){
  double* c=ac_product(group,z);
  const int k0=step*320,k=depth<last400>(step);
  double* packed=k>320?scratch[worker].c:scratch[worker].b;
  std::array<const double*,64> rows;
  for(int i=0;i<nr;++i)rows[i]=operand(z)+size_t(k0)*M+size_t(i)*16*k;
  const auto recipe=rhs_recipe[z];
  SmeNPanel first{n/32,k,rows[0],comm_pool->c+ci<last400>(group,recipe.x,step),c,packed,rows[1],
                  long(k)*32*8,long(k)*32*8,k*16*8/(n/32),step!=0};
  {
   if(inplace){SmeSumPanel d{first,packed};
    if(recipe.sign<0)tc1_rhs_sum_lookahead<true,true>(d);else tc1_rhs_sum_lookahead<false,true>(d);}
   else if(recipe.y<0)tc1_rhs_pack_lookahead(first);
   else{SmeSumPanel d{first,comm_pool->c+ci<last400>(group,recipe.y,step)};
    if(recipe.sign<0)tc1_rhs_sum_lookahead<true>(d);else tc1_rhs_sum_lookahead<false>(d);}
  }
  {
   SmeNPanel mid{n/32,k,rows[1],packed,c+16*n,packed,rows[2],long(k)*32*8,long(k)*32*8,k*16*8/(n/32),step!=0};
   SmeMRun run{{mid,c+32*n},rows.data()+1,nr-2};sme_mrun_accumulate(run);
   SmeNPanel last{n/32,k,rows[nr-1],packed,c+size_t(M-16)*n,packed,nullptr,long(k)*32*8,long(k)*32*8,k*16*8/(n/32),step!=0};
   sme_npanel_pipeline(last);
  }
 };
 for(int step=first_step;step<end_step;++step){
  one(which,step,false);
  if(follower>=0)one(follower,step,true);
 }
}
template<bool last400> static void terminal(int worker,int group,Outputs out){
 const int step=steps<last400>()-1,k=depth<last400>(step),k0=step*320,n=width(group),nu=n/2;
 const int u0=group*64,batch=u0/320,local=u0%320;
 const double* base=ac_product(group,0);
 double* packed=k>320?scratch[worker].c:scratch[worker].b;
 std::memcpy(packed,comm_pool->c+ci<last400>(group,0,step),size_t(n)*k*8);
 for(int row=0;row<M;row+=16){
  const int hs=row/512,p=row%512;
  const double* a=operand(0)+size_t(k0)*M+size_t(row)*k;
  SmeNPanel panel{n/32,k,a,packed,const_cast<double*>(base)+size_t(row)*n,packed,row+16<M?a+16*k:nullptr,
                   long(k)*32*8,long(k)*32*8,k*16*8/(n/32),1};
  SmeAcTerminalPanel d{{panel,nullptr},{},{}};
  for(int z=0;z<6;++z)d.other[z]=base+size_t(z+1)*product+size_t(row)*n;
  for(int nb=0;nb<n/32;++nb)for(int z=0;z<7;++z)
   d.output[nb][z]=out.at(batch,z,hs,nb/(nu/32)*512+p,local+(nb%(nu/32))*32);
  if(row+16<M)sme_ac_terminal_prefetch(d);else sme_ac_terminal_plain(d);
 }
}
}

// Ranks0/1/4 use M960; five helpers fill their twelve padded rows. Pool strides stay M1024.
namespace ac1owner {
using ac1aligned::operand;using ac1aligned::ac_product;using ac1aligned::product;
using ac1aligned::width;using ac1aligned::steps;using ac1aligned::depth;using ac1aligned::ci;
using ac1aligned::Outputs;using tbstrassen::br;using tbstrassen::b_quad;using tbstrassen::c_product;
template<bool helper> inline svfloat64_t load_a(const double* input,const double* donated,int st,int p,int own,int first_rows){
 const auto all=svptrue_b64();
 if constexpr(helper){
  if(p==496){
   const auto first4=svwhilelt_b64(0L,4L);
   const auto local=svld1_f64(first4,input+size_t(st)*500+496);
   const auto remote=svld1_f64(all,donated+size_t(st)*first_rows);
   return svsel_f64(first4,local,svext_f64(svdup_f64(0),remote,4));
  }
  if(p==504){
   if(first_rows==12)return svld1_f64(all,donated+size_t(st)*12+4);
   const auto first4=svwhilelt_b64(0L,4L);
   const auto lo=svld1_f64(first4,donated+size_t(st)*8+4);
   const auto hi=svld1_f64(first4,donated+16000ul*8+size_t(st)*4);
   return svsel_f64(first4,lo,svext_f64(svdup_f64(0),hi,4));
  }
 }
 return svld1_f64(svwhilelt_b64(long(p),long(own)),input+size_t(st)*500+p);
}
template<int P,bool helper> static void prepare_a(const double* input,const double* donated,int task,int first_rows=12){
 constexpr int M=2*P;
 const int hs=task/(P/16),p=task%(P/16)*16;
 for(int step=0;step<steps<false>();++step){
  const int k0=step*320,k=depth<false>(step);
  double* d[7];for(int z=0;z<7;++z)d[z]=operand(z)+size_t(k0)*M+size_t(hs*P+p)*k;
  for(int kk=0;kk<k;++kk)for(int lane=0;lane<16;lane+=8){
   const auto all=svptrue_b64();
   const int st=hs*4000+k0+kk;
   const auto a00=load_a<helper>(input,donated,st,p+lane,P==480?480:500,first_rows),
              a01=load_a<helper>(input,donated,st+2000,p+lane,P==480?480:500,first_rows),
              a10=load_a<helper>(input,donated,st+8000,p+lane,P==480?480:500,first_rows),
              a11=load_a<helper>(input,donated,st+10000,p+lane,P==480?480:500,first_rows);
   const int j=kk*16+lane;
   svst1_f64(all,d[0]+j,svadd_f64_x(all,a00,a11));
   svst1_f64(all,d[1]+j,svadd_f64_x(all,a10,a11));
   svst1_f64(all,d[2]+j,a00);svst1_f64(all,d[3]+j,a11);
   svst1_f64(all,d[4]+j,svadd_f64_x(all,a00,a01));
   svst1_f64(all,d[5]+j,svsub_f64_x(all,a10,a00));
   svst1_f64(all,d[6]+j,svsub_f64_x(all,a01,a11));
  }
 }
}
template<int P> static void multiply(int worker,int group,int which,int follower,int first_step,int end_step){
 constexpr int M=2*P;
 const int n=width(group),nr=M/16;
 auto one=[&](int z,int step,bool inplace){
  double* c=ac_product(group,z);
  const int k0=step*320,k=depth<false>(step);
  double* packed=k>320?scratch[worker].c:scratch[worker].b;
  std::array<const double*,64> rows;
  for(int i=0;i<nr;++i)rows[i]=operand(z)+size_t(k0)*M+size_t(i)*16*k;
  const auto recipe=rhs_recipe[z];
  SmeNPanel first{n/32,k,rows[0],comm_pool->c+ci<false>(group,recipe.x,step),c,packed,rows[1],
                  long(k)*32*8,long(k)*32*8,k*16*8/(n/32),step!=0};
  {
   if(inplace){SmeSumPanel d{first,packed};
    if(recipe.sign<0)tc1_rhs_sum_lookahead<true,true>(d);else tc1_rhs_sum_lookahead<false,true>(d);}
   else if(recipe.y<0)tc1_rhs_pack_lookahead(first);
   else{SmeSumPanel d{first,comm_pool->c+ci<false>(group,recipe.y,step)};
    if(recipe.sign<0)tc1_rhs_sum_lookahead<true>(d);else tc1_rhs_sum_lookahead<false>(d);}
  }
  {
   SmeNPanel mid{n/32,k,rows[1],packed,c+16*n,packed,rows[2],long(k)*32*8,long(k)*32*8,k*16*8/(n/32),step!=0};
   SmeMRun run{{mid,c+32*n},rows.data()+1,nr-2};sme_mrun_accumulate(run);
   SmeNPanel last{n/32,k,rows[nr-1],packed,c+size_t(M-16)*n,packed,nullptr,long(k)*32*8,long(k)*32*8,k*16*8/(n/32),step!=0};
   sme_npanel_pipeline(last);
  }
 };
 for(int step=first_step;step<end_step;++step){
  one(which,step,false);
  if(follower>=0)one(follower,step,true);
 }
}
template<int P> static void terminal(int worker,int group,Outputs out){
 constexpr int M=2*P;
 const int step=steps<false>()-1,k=depth<false>(step),k0=step*320,n=width(group),nu=n/2;
 const int u0=group*64,batch=u0/320,local=u0%320;
 const double* base=ac_product(group,0);
 double* packed=k>320?scratch[worker].c:scratch[worker].b;
 std::memcpy(packed,comm_pool->c+ci<false>(group,0,step),size_t(n)*k*8);
 for(int row=0;row<M;row+=16){
  const int hs=row/P,p=row%P;
  const double* a=operand(0)+size_t(k0)*M+size_t(row)*k;
  SmeNPanel panel{n/32,k,a,packed,const_cast<double*>(base)+size_t(row)*n,packed,row+16<M?a+16*k:nullptr,
                   long(k)*32*8,long(k)*32*8,k*16*8/(n/32),1};
  SmeAcTerminalPanel d{{panel,nullptr},{},{}};
  for(int z=0;z<6;++z)d.other[z]=base+size_t(z+1)*product+size_t(row)*n;
  for(int nb=0;nb<n/32;++nb)for(int z=0;z<7;++z)
   d.output[nb][z]=out.at(batch,z,hs,nb/(nu/32)*P+p,local+(nb%(nu/32))*32);
  if(row+16<M)sme_ac_terminal_prefetch(d);else sme_ac_terminal_plain(d);
 }
}
template<int M> static void update_panel(const double* a,const double* b,double* packed,double* partial,
                         int which,int hs,int u0,int q0,bool add=true,bool inplace=false){
 const int k=std::min(320,4000-u0);
 const auto r=br[which];
 std::array<const double*,64> rows;
 for(int i=0;i<M/16;++i)rows[i]=a+size_t(i)*16*k;
 SmeNPanel first{4,k,rows[0],b_quad(b,r.x,hs,u0,q0),partial,packed,rows[1],
                 16000L*32*8,long(k)*32*8,k*16*8/4,add};
 {
 if(inplace){ // TC_PAIR_PACK&1: packed holds the partner's y basis; X0 +- packed, in place
  SmeSumPanel sum{first,packed};
  if(r.sign==1)tc1_rhs_sum_lookahead<false,true>(sum);else tc1_rhs_sum_lookahead<true,true>(sum);
 }else if(r.y<0)tc1_rhs_pack_lookahead(first);
 else{
  SmeSumPanel sum{first,b_quad(b,r.y,hs,u0,q0)};
  if(r.sign==1)tc1_rhs_sum_lookahead<false>(sum);
  else tc1_rhs_sum_lookahead<true>(sum);
 }
 }
 {
 SmeNPanel middle{4,k,rows[1],packed,partial+16*128,packed,rows[2],
                  long(k)*32*8,long(k)*32*8,k*16*8/4,add};
 SmeMRun run{{middle,partial+32*128},rows.data()+1,M/16-2};
 sme_mrun_accumulate(run);
 SmeNPanel last{4,k,rows[M/16-1],packed,partial+(M-16)*128,packed,nullptr,
                long(k)*32*8,long(k)*32*8,k*16*8/4,add};
 sme_npanel_pipeline(last);
 }
}
template<int P,bool helper> static void combine_store(const double* products,double* out,double* donated,int pair,int begin=0,int end=P){
 const double* base=products+size_t(pair)*7*c_product;
 alignas(64) double tile[4][2*512];
 for(int p0=begin;p0<end;p0+=16)for(int nb=0;nb<4;++nb){
  for(int r=0;r<2;++r){
   const size_t at=size_t((r*P+p0)/16)*2048+nb*512;
   const double *a=base+at,*b=a+c_product,*c=b+c_product,*d=c+c_product,
                *e=d+c_product,*f=e+c_product,*g=f+c_product;
#pragma omp simd
   for(int i=0;i<512;++i){
    const double m1=a[i],m2=b[i],m3=c[i],m4=d[i],m5=e[i],m6=f[i],m7=g[i];
    tile[0][r*512+i]=((m1+m4)-m5)+m7;
    tile[1][r*512+i]=m3+m5;
    tile[2][r*512+i]=m2+m4;
    tile[3][r*512+i]=((m1-m2)+m3)+m6;
   }
  }
  for(int pi=0;pi<16;++pi)for(int j=0;j<32;++j){
   const int qlocal=pair%2*256+nb*32+j;
   const int q=pair/2*500+qlocal;
   const int p=p0+pi;
   if(p>=(P==480?480:helper?512:500))continue;
   double* destination=out+size_t(std::min(p,499))*16000;
   if constexpr(helper)if(p>=500)destination=donated+size_t(p-500)*16000;
   for(int r=0;r<2;++r){
    const size_t i=r*512+j*16+pi;
    if(qlocal<500)destination[q*4+r]=tile[0][i];
    if(qlocal+128<500)destination[(q+128)*4+r]=tile[1][i];
    if(qlocal<500)destination[q*4+r+2]=tile[2][i];
    if(qlocal+128<500)destination[(q+128)*4+r+2]=tile[3][i];
   }
  }
 }
}

}

// Reuse the pinned share area and separate TC1 epoch flag cache lines.
class TC1RowShare {
 struct Edge {int donor,helper,p,count,offset;};
 static constexpr Edge edges[]={{0,6,480,12,0},{0,7,492,8,0},
   {1,2,480,12,0},{1,3,492,8,0},{4,5,480,12,0},{4,3,492,4,8},{4,7,496,4,8}};
 static constexpr int tasks=16,st_rows=16000;
 static constexpr size_t row_bytes=st_rows*8;
 static constexpr size_t aflag=ShareArea::a_flag+64,oflag=ShareArea::o_flag+64;
 const int rank;const double* input;double* output;MPI_Comm comm;
 Sdma& dma;const std::vector<u32>& pids;std::array<uintptr_t,8> pools;
 const unsigned long epoch;
 std::atomic<int> next{0},left{tasks};
 std::atomic<bool> received{false};
public:
 TC1RowShare(int r,const double* a,double* out,MPI_Comm c,PrepAcBroadcast& x)
  :rank(r),input(a),output(out),comm(c),dma(x.dma),pids(x.pids),pools(share_pools(c)),epoch(c_epoch){
  static_assert(20*row_bytes<=ShareArea::a_flag-ShareArea::a_out,"donor A fits");
  static_assert(12*row_bytes<=ShareArea::o_rows-ShareArea::a_in,"helper A fits");
  static_assert(12*row_bytes<=ShareArea::o_flag-ShareArea::o_rows,"helper output fits");
  static_assert(20*row_bytes<=ShareArea::snapshot-ShareArea::o_in,"returned output fits");
  if(!epoch)std::abort();
 }
 bool donor()const{return rank==0||rank==1||rank==4;}
 bool helper()const{return !donor();}
 int first_rows()const{return rank==3||rank==7?8:12;}
 bool prepare_one(){
  if(!donor()||next.load(std::memory_order_relaxed)>=tasks)return false;
  const int task=next.fetch_add(1,std::memory_order_relaxed);if(task>=tasks)return false;
  auto* dst=(double*)ShareArea::at(ShareArea::a_out);
  for(const auto& e:edges)if(e.donor==rank)
   for(int row=task*1000;row<(task+1)*1000;++row)
    std::memcpy(dst+size_t(e.p-480)*st_rows+size_t(row)*e.count,
                input+size_t(row)*500+e.p,e.count*8);
  if(left.fetch_sub(1,std::memory_order_acq_rel)==1){
   asm volatile("dmb oshst":::"memory");
   __atomic_store_n((unsigned long*)ShareArea::at(aflag),epoch,__ATOMIC_RELEASE);
  }
  return true;
 }
 void receive(){
  if(!helper())return;
  for(const auto& e:edges)if(e.helper==rank){
   {
    ShareArea::wait_flag(dma,pids[e.donor],pools[e.donor],aflag,epoch,comm);
   }
   {
    ShareArea::submit_copy(dma,pids[e.donor],pools[e.donor],ShareArea::a_out+size_t(e.p-480)*row_bytes,
                           e.count*row_bytes,ShareArea::at(ShareArea::a_in+e.offset*row_bytes));
    dma.wait();
   }
  }
  asm volatile("dmb osh":::"memory");
  received.store(true,std::memory_order_release);
 }
 // TC_SHARE_ASYNC: one non-blocking step of receive(), called by worker 0 between C chunk polls; true once all rows
 // arrived. A donor flag that comes late then no longer holds back this rank's C stream.
 int pulled=0;
 bool progress(){
  if(!helper() || received.load(std::memory_order_relaxed))return true;
  int index=0;
  for(const auto& e:edges)if(e.helper==rank){
   if(index++<pulled)continue;
   if(!ShareArea::flag_raised(dma,pids[e.donor],pools[e.donor],aflag,epoch))return false;
   ShareArea::submit_copy(dma,pids[e.donor],pools[e.donor],ShareArea::a_out+size_t(e.p-480)*row_bytes,
                          e.count*row_bytes,ShareArea::at(ShareArea::a_in+e.offset*row_bytes));
   dma.wait();
   ++pulled;
  }
  asm volatile("dmb osh":::"memory");
  received.store(true,std::memory_order_release);
  return true;
 }
 const double* a_rows(){
  while(!received.load(std::memory_order_acquire))asm volatile("yield":::"memory");
  return (const double*)ShareArea::at(ShareArea::a_in);
 }
 double* out_rows(){return (double*)ShareArea::at(ShareArea::o_rows);}
 // TC_SHARE_PULL: one idle worker waits for the helpers and pulls their rows into o_in while others still combine
 std::atomic<int> pull_state{0};
 bool pull_output(){
  if(!donor() || pull_state.load(std::memory_order_relaxed))return false;
  int expected=0;
  if(!pull_state.compare_exchange_strong(expected,1,std::memory_order_acq_rel))return false;
  pull();
  pull_state.store(2,std::memory_order_release);
  return true;
 }
 void pull(){
  for(const auto& e:edges)if(e.donor==rank){
   ShareArea::wait_flag(dma,pids[e.helper],pools[e.helper],oflag,epoch,comm);
  }
  {
   for(const auto& e:edges)if(e.donor==rank)
    ShareArea::submit_copy(dma,pids[e.helper],pools[e.helper],ShareArea::o_rows+e.offset*row_bytes,
                           e.count*row_bytes,ShareArea::at(ShareArea::o_in+size_t(e.p-480)*row_bytes));
   dma.wait();asm volatile("dmb osh":::"memory");
  }
 }
 void finish(){
  // The OpenMP join completed every output tile before publication.
  if(helper()){
   asm volatile("dmb oshst":::"memory");
   __atomic_store_n((unsigned long*)ShareArea::at(oflag),epoch,__ATOMIC_RELEASE);
   return;
  }
  if(left.load(std::memory_order_acquire))std::abort();
  if(1){
   if(pull_state.load(std::memory_order_acquire)!=2)pull();
   parallel_copy(output+480ul*st_rows,(const double*)ShareArea::at(ShareArea::o_in),20*row_bytes);
   return;
  }
  for(const auto& e:edges)if(e.donor==rank){
   ShareArea::wait_flag(dma,pids[e.helper],pools[e.helper],oflag,epoch,comm);
  }
  {
   for(const auto& e:edges)if(e.donor==rank)
    ShareArea::submit_copy(dma,pids[e.helper],pools[e.helper],ShareArea::o_rows+e.offset*row_bytes,
                           e.count*row_bytes,ShareArea::at(ShareArea::o_in+size_t(e.p-480)*row_bytes));
   dma.wait();asm volatile("dmb osh":::"memory");
  }
  {
   std::memcpy(output+480ul*st_rows,ShareArea::at(ShareArea::o_in),20*row_bytes);
  }
 }
};

struct Tc1AlignedBroadcast {
 static constexpr size_t bytes=512000000,metadata=bytes;
 static constexpr int chunks=(bytes+c_ready_chunk-1)/c_ready_chunk;
 std::array<unsigned,63*7> tasks{};
 PrepAcBroadcast control;
 std::array<CReadyCounter,chunks> counters;
 Tc1AlignedBroadcast(MPI_Comm comm,int rank):control(comm,rank){
  if(rank==0){
   std::array<int,chunks> pending{};
   for(int g=0;g<63;++g)for(int step=0;step<7;++step){
    const size_t begin=ac1aligned::ci<false>(g,0,step)*8;
    const size_t end=begin+size_t(4)*ac1aligned::width(g)*ac1aligned::depth<false>(step)*8;
    for(size_t c=begin/c_ready_chunk;c<=(end-1)/c_ready_chunk;++c)++pending[c];
   }
   for(int c=0;c<chunks;++c){if(!pending[c])std::abort();counters[c].left.store(pending[c],std::memory_order_relaxed);}
  }
 }
 void pack(unsigned id,const double* input){
  const int g=id/7,step=id%7;
  ac1aligned::prepare_c<false>(input,g,step);
  const size_t begin=ac1aligned::ci<false>(g,0,step)*8;
  const size_t end=begin+size_t(4)*ac1aligned::width(g)*ac1aligned::depth<false>(step)*8;
  for(size_t c=begin/c_ready_chunk;c<=(end-1)/c_ready_chunk;++c)
   if(counters[c].left.fetch_sub(1,std::memory_order_acq_rel)==1){
    asm volatile("dmb oshst":::"memory");
    __atomic_store_n((u64*)((char*)comm_pool->c+metadata+c*64),u64(c_epoch),__ATOMIC_RELEASE);
   }
 }
 void receive(int rank,MPI_Comm comm,void(*progress)(void*)=nullptr,void* context=nullptr){
  c_tree_ready_transfer(comm_pool->c,bytes,metadata,control.dma,control.pids,control.root,control.relay,control.relay1,rank,comm,progress,context);
 }
};
struct TC1AlignedCoop {
 static constexpr int workers=38,batches=13,q_groups=32,tb_updates_per_batch=224;
 // TC_PAIR_PACK: X0-X3, X0+X1, X0+X2 (which 1, 5, 6) run right after X3, X1, X2 (which 4, 2, 3) on one worker
 static constexpr int pair_follower[7]={-1,-1,5,6,1,-1,-1};
 static constexpr bool is_follower[7]={false,true,false,false,false,true,true};
 std::array<std::atomic<int>,112> product_next{};
 std::array<std::atomic<int>,16> product_pair_left{};
 std::array<std::atomic<unsigned>,16> combine_bits{};
 alignas(64) std::atomic<int> combine_left{64};
 struct alignas(64) Cursor {int next=0;};std::array<Cursor,38> cursors;
 alignas(64) std::atomic<bool> b_ready{false},a_ready{false};
 std::array<std::atomic<bool>,32> b_panels{};
 std::array<std::atomic<int>,13> panel_tb_left{},ac_left{};
 std::array<std::atomic<bool>,13> tb_finished{},ac_finished{};
 std::array<std::array<std::atomic<int>,35>,13> ac_next{};
 std::array<std::atomic<int>,63> group_left{};
 alignas(64) std::atomic<int> consume_batch{0},ac_retired{0};
 alignas(64) std::atomic<int> ac_jobs_left{441},tb_k160_left{224*25};
 const double* b;double* out;const int rank;TC1RowShare& sharing;
 TC1AlignedCoop(const double* bb,double* oo,int r,TC1RowShare& sh):b(bb),out(oo),rank(r),sharing(sh){
  for(auto& x:product_next)x.store(0,std::memory_order_relaxed);
  for(auto& x:product_pair_left)x.store(7,std::memory_order_relaxed);
  for(auto& x:combine_bits)x.store(0,std::memory_order_relaxed);
  for(auto& x:b_panels)x.store(false,std::memory_order_relaxed);
  for(auto& x:panel_tb_left)x.store(224,std::memory_order_relaxed);
  for(auto& x:group_left)x.store(7,std::memory_order_relaxed);
  for(int batch=0;batch<13;++batch){
   ac_left[batch].store(batch==12?21:35,std::memory_order_relaxed);
   tb_finished[batch].store(false,std::memory_order_relaxed);ac_finished[batch].store(false,std::memory_order_relaxed);
   for(auto& x:ac_next[batch])x.store(0,std::memory_order_relaxed);
  }
  for(int w=0;w<38;++w)cursors[w].next=w;
  if(slices()>32)std::abort();
  combine_left.store(16*slices(),std::memory_order_relaxed);
 }
 int rows()const{return sharing.donor()?480:512;}
 int slices()const{return (rows()+32-1)/32;}
 bool done()const{return consume_batch.load(std::memory_order_acquire)==13 && combine_left.load(std::memory_order_acquire)==0;}
 int producer_quota()const{
  const int jobs=ac_jobs_left.load(std::memory_order_relaxed);if(!jobs)return 0;
  const int ac=12*jobs,tb=tb_k160_left.load(std::memory_order_relaxed);
  return std::min(37,std::max(std::min(7,jobs),(38*ac+(ac+tb)/2)/(ac+tb)));
 }
 double* sum_at(int batch,int slot,int hs){
  return ac1aligned::t_derived(batch)+size_t(slot*2+hs)*1024*320;
 }
 double* alias_at(int batch,int which,int hs){
  return ac1aligned::t_alias(batch)+size_t(hs*2+(which==3))*1024*320;
 }
 void retire(int phase){
  const int batch=phase/2;
  tb_k160_left.fetch_sub(batch==12?1:2,std::memory_order_relaxed);
  if(panel_tb_left[batch].fetch_sub(1,std::memory_order_acq_rel)!=1)return;
  tb_finished[batch].store(true,std::memory_order_release);
  int prefix=consume_batch.load(std::memory_order_acquire);
  while(prefix<13&&tb_finished[prefix].load(std::memory_order_acquire))
   if(consume_batch.compare_exchange_weak(prefix,prefix+1,std::memory_order_acq_rel))++prefix;
 }
 bool try_produce(int worker,int batch){
  if(!a_ready.load(std::memory_order_acquire))return false;
  const int count=batch==12?21:35;
  const bool paired=9&8;
  for(int own=1;own>=0;--own)for(int task=0;task<count;++task){
   if((task%38==worker)!=bool(own))continue;
   const int group=batch*5+task/7,which=task%7;
   // TC_PAIR_PACK&8: a sum product runs inside its single-basis partner's job (see try_consume)
   if(paired && is_follower[which])continue;
   const int follower=paired?pair_follower[which]:-1,products=follower>=0?2:1;
   const int steps=7-(which==0);
   int step=ac_next[batch][task].load(std::memory_order_acquire);
   if(step<0 || step>=steps)continue;
   const int end_step=std::min(step+3,steps);
   const size_t begin=size_t(group)*128*2000*4*8,end=begin+size_t(ac1aligned::width(group))*2000*4*8;
   bool ready=true;
   for(size_t c=begin/c_ready_chunk;c<=(end-1)/c_ready_chunk;++c)
    if(__atomic_load_n((u64*)((char*)comm_pool->c+512000000+c*64),__ATOMIC_ACQUIRE)!=c_epoch){ready=false;break;}
   if(!ready)continue;
   // A lease owns both products, including packed RHS reuse, for whole K steps.
   if(!ac_next[batch][task].compare_exchange_strong(step,-step-1,std::memory_order_acq_rel))continue;
   {
    if(sharing.donor())ac1owner::multiply<480>(worker,group,which,follower,step,end_step);
    else ac1aligned::multiply<false>(worker,group,which,follower,step,end_step);
    if(end_step==steps){
    if(group_left[group].fetch_sub(products,std::memory_order_acq_rel)==products){
     if(sharing.donor())ac1owner::terminal<480>(worker,group,{comm_pool->temp,tb_strassen_t_workspace});
     else ac1aligned::terminal<false>(worker,group,{comm_pool->temp,tb_strassen_t_workspace});
    }
    }
   }
   // Publish after the full paired lease; subsequent owners acquire its partials.
   ac_next[batch][task].store(end_step,std::memory_order_release);
   if(end_step<steps)return true;
   ac_jobs_left.fetch_sub(products,std::memory_order_relaxed);
   if(ac_left[batch].fetch_sub(products,std::memory_order_acq_rel)==products){
    ac_finished[batch].store(true,std::memory_order_release);
    int prefix=ac_retired.load(std::memory_order_acquire);
    while(prefix<13&&ac_finished[prefix].load(std::memory_order_acquire))
     if(ac_retired.compare_exchange_weak(prefix,prefix+1,std::memory_order_acq_rel))++prefix;
   }
   return true;
  }
  return false;
 }
 // Consume completed AC panels in K order for each TB product.
     // TC_PAIR_PACK&1: the sum products (which 1, 5, 6) are not claimed themselves; each part runs right after the
     // same part of its single-basis partner (which 4, 2, 3) and adds X0 to the packed panel
     void update(int worker,int destination,int part,bool inplace){
            const int task_batch=part/2,hs=part%2,which=destination%7,pair=destination/7;
            const int slot=tbstrassen::derived_slot[which],depth=task_batch==12?160:320;
            {
                const double* source=slot>=0?sum_at(task_batch,slot,hs):alias_at(task_batch,which,hs);
                double* target=tb_strassen_products_workspace+size_t(destination)*tbstrassen::c_product;
                // TC_P0ADD: parts run in order, so the first one can store ZA without loading zeroed accumulators
                const bool add=!(1 && part==0);
                if(part==0 && add)memset(target,0,tbstrassen::c_product*sizeof(double));
                if(sharing.donor())ac1owner::update_panel<960>(source,b,scratch[worker].b,target,which,hs,task_batch*320,pair*256,add,inplace);
                else tbstrassen::update_panel(source,b,scratch[worker].b,target,which,hs,task_batch*320,pair*256,add,inplace);
                if(part==25 && product_pair_left[pair].fetch_sub(1,std::memory_order_acq_rel)==1)
                    combine_bits[pair].store(slices()==32?~0u:(1u<<slices())-1,std::memory_order_release);
            }
     }
     bool try_consume(int worker,int batch){
        const bool paired=9&1;
        const bool tail=ac_jobs_left.load(std::memory_order_acquire)==0;
        int first=cursors[worker].next;
        if(tail){
            // Keep every paired update and full-M kernel intact. Once AC has
            // finished, prioritize the largest remaining serial TB chain.
            int best=-1,most=-1;
            for(int offset=0;offset<112;++offset){
                const int item=(first+offset)%112,which=item%7;
                if(is_follower[which])continue;
                const int part=product_next[item].load(std::memory_order_acquire);
                if(part<0 || part>=26)continue;
                // Parts 0..23 have K320; parts 24/25 have K160.
                const int units=part<24?50-2*part:26-part;
                const int weight=units*(pair_follower[which]>=0?2:1);
                if(weight>most){best=item;most=weight;}
            }
            if(best<0)return false;
            first=best;
        }
        // A lost best-chain claim returns to the cooperative loop instead of
        // silently substituting a short chain. Existing data-ready checks stay.
        for(int offset=0;offset<(tail?1:112);++offset){
            const int destination=(first+offset)%112;
            const int which=destination%7,pair=destination/7;
            if(paired && is_follower[which])continue;
            const int follower=paired?pair_follower[which]:-1;
            int part=product_next[destination].load(std::memory_order_acquire);
            if(part<0 || part>=26)continue;
            const int task_batch=part/2;
            if(!ac_finished[task_batch].load(std::memory_order_acquire))continue;
            const auto recipe=tbstrassen::br[which];
            if(!b_panels[pair*2+recipe.x%2].load(std::memory_order_acquire) ||
               (recipe.y>=0 && !b_panels[pair*2+recipe.y%2].load(std::memory_order_acquire)))continue;
            if(follower>=0 && !b_panels[pair*2].load(std::memory_order_acquire))continue; // X0 for the follower
            if(!product_next[destination].compare_exchange_strong(part,-part-1,std::memory_order_acq_rel))continue;
            update(worker,destination,part,false);
            if(follower>=0){
                update(worker,pair*7+follower,part,sharing.donor()||(1&1));
                product_next[pair*7+follower].store(part+1,std::memory_order_release);
            }
            product_next[destination].store(part+1,std::memory_order_release);
            cursors[worker].next=(destination+1)%112;
            retire(task_batch*2+1);
            if(follower>=0)retire(task_batch*2+1);
            return true;
        }
        return false;
    }
 // GEMMs retain priority. Only idle workers claim disjoint final P128 slices.
 bool try_combine(int worker){
  for(int i=0;i<16;++i){
   const int pair=(worker+i)%16;
   auto& bank=combine_bits[pair];unsigned bits=bank.load(std::memory_order_acquire);
   while(bits){
    const int slice=__builtin_ctz(bits);
    if(bank.compare_exchange_weak(bits,bits&(bits-1),std::memory_order_acq_rel,std::memory_order_acquire)){
     const int begin=slice*32,end=std::min(begin+32,rows());
     {
      if(sharing.donor())ac1owner::combine_store<480,false>(tb_strassen_products_workspace,out,nullptr,pair,begin,end);
      else ac1owner::combine_store<512,true>(tb_strassen_products_workspace,out,sharing.out_rows(),pair,begin,end);
     }
     combine_left.fetch_sub(1,std::memory_order_release);
     return true;
    }
   }
  }
  return false;
 }

 bool try_compute(int worker){
  const int consumer=consume_batch.load(std::memory_order_acquire);
  // TC_W0_TB: worker 0 polls the B gather between short TB updates; no AC jobs, which would delay the B panels
  if(worker==0&&!b_ready.load(std::memory_order_acquire))return 1&&consumer<13&&try_consume(worker,consumer);
  if(consumer==13)return try_combine(worker);
  auto produce=[&](){
   const int prefix=ac_retired.load(std::memory_order_acquire);
   for(int batch=prefix;batch<std::min({13,prefix+3,consumer+tc_t_slots});++batch)
    if(!ac_finished[batch].load(std::memory_order_acquire)&&try_produce(worker,batch))return true;
   return false;
  };
  // TC_W37: worker 37 shares its core with the node's monitoring daemons; it takes only short TB updates and slices
  if(1&&worker==workers-1)return try_consume(worker,consumer)||try_combine(worker);
  // TC_AC_ENDGAME: once few AC jobs remain, every worker starts one before taking TB updates
  const bool prefer=worker>0&&(worker<=producer_quota()||ac_jobs_left.load(std::memory_order_relaxed)<=40);
  if(prefer&&produce())return true;
  if(0&&try_combine(worker))return true;
  if(try_consume(worker,consumer))return true;
  if(prefer)return try_combine(worker);
  return produce()||try_combine(worker);
 }
};

// Each A panel and each B pack task is claimed once. B completion never gates
// AC; only the main worker polls the asynchronous B queue.
struct TC1AlignedReadyPipeline {
    TC1AlignedCoop* cooperative;
    const double *a_input,*b_input;
    const int pack_workers,a_tasks;
    alignas(64) std::atomic<int> next_a{0},next_b{0},b_left{tc1_tb_rhs_tasks},a_left{64};
    void* b_context=nullptr;
    void (*b_start)(void*)=nullptr;
    static constexpr size_t b_flag=CommPool::small-128;
    TC1AlignedReadyPipeline(const double* a,const double* b,TC1AlignedCoop* coop)
        :cooperative(coop),a_input(a),b_input(b),pack_workers(tc1_coop_c_pack_workers),a_tasks(coop->sharing.donor()?60:64){
        a_left.store(a_tasks,std::memory_order_relaxed);
        if(pack_workers<1 || pack_workers>37)std::abort();
    }
    bool packing_worker(int worker)const{
        if(1)return worker>=38-1-pack_workers && worker<38-1;
        return worker>=38-pack_workers;
    }
    void start_b_if_ready(){
        if(__atomic_load_n((u64*)((char*)comm_pool->b_source+b_flag),__ATOMIC_ACQUIRE)==1)
            b_start(b_context);
    }
    bool prepare_one(){
        if(cooperative->sharing.prepare_one())return true;
        int index=next_a.load(std::memory_order_relaxed);
        if(index<a_tasks)index=next_a.fetch_add(1,std::memory_order_relaxed);
        if(index<a_tasks){
            if(cooperative->sharing.donor())ac1owner::prepare_a<480,false>(a_input,nullptr,index);
            else if(cooperative->sharing.helper() && index%32==31)
                ac1owner::prepare_a<512,true>(a_input,cooperative->sharing.a_rows(),index,cooperative->sharing.first_rows());
            else ac1aligned::prepare_a<false>(a_input,index);
            if(a_left.fetch_sub(1,std::memory_order_acq_rel)==1)cooperative->a_ready.store(true,std::memory_order_release);
            return true;
        }
        index=next_b.load(std::memory_order_relaxed);
        if(index<tc1_tb_rhs_tasks)index=next_b.fetch_add(1,std::memory_order_relaxed);
        if(index>=tc1_tb_rhs_tasks)return false;
        pack_tc1_tb_rhs_task(b_input,index);
        if(b_left.fetch_sub(1,std::memory_order_acq_rel)==1){
            asm volatile("dmb oshst":::"memory");
            __atomic_store_n((u64*)((char*)comm_pool->b_source+b_flag),u64(1),__ATOMIC_RELEASE);
        }
        return true;
    }
    bool try_compute(int worker){return cooperative->try_compute(worker);}
    bool done()const{return cooperative->done();}
};
static TC1AlignedReadyPipeline* tc1_aligned_pipeline=nullptr;

// Only the MPI/main worker progresses this separate B DMA queue.
class TC1AlignedGather {
    Sdma& dma;MPI_Comm comm;bool finished=false,started=false;
    std::vector<u32> pids;std::vector<unsigned long> addresses;
    double* destination=nullptr;size_t width=0,total=0;
    size_t checked_peer=0;
    size_t published_panels=0;
    double flag_deadline=0,bulk_deadline=0;
    uint64_t trace_begin=0;
    static Sdma& select_dma(MPI_Comm communicator){
        int rank;MPI_Comm_rank(communicator,&rank);
        sdma_check(comm_pool->dma->process==comm_pool->other_binding->process,"B routing requires equal process IDs");
        return *((224>>rank)&1?comm_pool->dma:comm_pool->other_binding);
    }
public:
    template<class Buffer>
    TC1AlignedGather(const std::vector<double>& local,Buffer& full,int count,MPI_Comm communicator):
        dma(select_dma(communicator)),comm(communicator),destination(full.data()),width(size_t(count)*8),total(full.size()*8){
        if(!tc1_aligned_pipeline || local.size()*8>CommPool::small || total>CommPool::large || full.data()!=comm_pool->b_full)MPI_Abort(comm,4);
        if(width>TC1AlignedReadyPipeline::b_flag)MPI_Abort(comm,4);
        memset((char*)comm_pool->b_source+TC1AlignedReadyPipeline::b_flag,0,128);
        int size;MPI_Comm_size(comm,&size);pids.resize(size);addresses.resize(size);
        unsigned long address=(uintptr_t)comm_pool->b_source;
        MPI_Allgather(&dma.process,1,MPI_UNSIGNED,pids.data(),1,MPI_UNSIGNED,comm);
        MPI_Allgather(&address,1,MPI_UNSIGNED_LONG,addresses.data(),1,MPI_UNSIGNED_LONG,comm);
        // A receiver may use either device, so every source authorizes both.
        comm_pool->dma->authorize(pids);comm_pool->other_binding->authorize(pids);
        MPI_Barrier(comm);
        tc1_aligned_pipeline->b_context=this;
        tc1_aligned_pipeline->b_start=[](void* context){
            auto* gather=static_cast<TC1AlignedGather*>(context);
            {
                auto& ready=tc1_aligned_pipeline->cooperative->b_ready;
                if(ready.load(std::memory_order_acquire))return;
                if(gather->progress())ready.store(true,std::memory_order_release);
                return;
            }
        };
    }
    bool progress(){
        if(finished)return true;
        if(!flag_deadline){
            flag_deadline=tc_seconds()+30;
        }
        if(!started){
            auto* snapshot=(u64*)((char*)comm_pool->b_source+TC1AlignedReadyPipeline::b_flag+64);
            while(checked_peer<pids.size()){
                hisi_sdma_sqe_task flag{};
                flag.src_addr=addresses[checked_peer]+TC1AlignedReadyPipeline::b_flag;flag.dst_addr=(uintptr_t)snapshot;
                flag.src_process_id=pids[checked_peer];flag.dst_process_id=dma.process;flag.length=64;
                dma.submit_task(flag);dma.wait(); // Only this 64-byte probe is outstanding.
                const u64 value=*(volatile u64*)snapshot;
                if(value!=1){
                    if(value!=0 || tc_seconds()>flag_deadline)MPI_Abort(comm,5);
                    return false; // Continue C progress instead of spinning on a peer.
                }
                ++checked_peer;
            }
            for(size_t peer=0;peer<pids.size();++peer){
                constexpr size_t panels_per_peer=512/128;
                sdma_check(pids.size()==8 && width==16000ul*512*8 && total==width*8,
                           "TC1 B panel shape");
                const size_t panel_bytes=width/panels_per_peer;
                for(size_t panel=0;panel<panels_per_peer;++panel){
                    const size_t offset=panel*panel_bytes;
                    hisi_sdma_sqe_task task{};
                    task.src_addr=addresses[peer]+offset;
                    task.dst_addr=(uintptr_t)destination+peer*width+offset;
                    task.src_process_id=pids[peer];task.dst_process_id=dma.process;task.length=panel_bytes;
                    task.dst_stride_len=total-panel_bytes;task.stride_num=1;dma.submit_task(task);
                }
            }
            started=true;bulk_deadline=tc_seconds()+10;
        }
        const bool complete=dma.poll();
        // poll may drain a prefix and keep the queue leased. Completion IDs
        // are checked there; outstanding remains valid after its final reset.
        constexpr size_t panel_count=TC1AlignedCoop::q_groups;
        const size_t completed=panel_count-dma.outstanding;
        sdma_check(completed>=published_panels && completed<=panel_count,"B panel completion prefix");
        if(completed>published_panels)asm volatile("dmb osh":::"memory");
        while(published_panels<completed){
            tc1_aligned_pipeline->cooperative->b_panels[published_panels].store(true,std::memory_order_release);
            ++published_panels;
        }
        if(!complete){
            if(tc_seconds()>bulk_deadline)throw std::runtime_error("asynchronous B completion timeout");
            return false;
        }
        finished=true;
        return true;
    }
    void finish(){
        while(!progress())asm volatile("yield":::"memory");
    }
    ~TC1AlignedGather(){if(!finished)std::_Exit(3);MPI_Barrier(comm);}
};

static void tc1_aligned_ready(std::vector<double>& a,std::vector<double>& b,
                              std::vector<double>& c,std::vector<double>& out,MPI_Comm comm,int rank){
    if(omp_get_max_threads()!=38 || svcntd()!=8 || a.size()!=8000000 ||
       b.size()!=8000000 || c.size()!=64000000 || out.size()!=8000000)std::abort();
    // This also advances C epochs and authorizes the current master DMA tree.
    Tc1AlignedBroadcast exchange(comm,rank);
    WorkBuffer all_b(16000ul*512*8,true);
    TC1RowShare sharing(rank,a.data(),out.data(),comm,exchange.control);
    TC1AlignedCoop fused(all_b.data(),out.data(),rank,sharing);
    TC1AlignedReadyPipeline pipeline(a.data(),b.data(),&fused);
    tc1_aligned_pipeline=&pipeline;
    TC1AlignedGather pending_b(b,all_b,16000*512,comm);
    std::atomic<int> next_pack{0};
#pragma omp parallel num_threads(38)
    {
        const int worker=omp_get_thread_num();
        if(omp_get_num_threads()!=38)std::abort();
        if(worker==0 && !1)sharing.receive();
        if(rank!=0 && worker==0){
            if(1)sharing.progress();
            exchange.receive(rank,comm,[](void* context){
                auto* p=static_cast<TC1AlignedReadyPipeline*>(context);
                if(1)p->cooperative->sharing.progress();
                p->start_b_if_ready();
            },&pipeline);
            if(1){
                const double deadline=tc_seconds()+30;
                while(!sharing.progress())if(tc_seconds()>deadline)MPI_Abort(comm,5);
            }
        }
        for(;;){
            if(worker==0)pipeline.start_b_if_ready();
            if(rank!=0){
                if(worker && pipeline.prepare_one())continue;
                if(pipeline.try_compute(worker))continue;
            }else{
                if(!pipeline.packing_worker(worker) && pipeline.prepare_one())continue;
                const bool packing=next_pack.load(std::memory_order_relaxed)<int(exchange.tasks.size());
                if((!pipeline.packing_worker(worker) || !packing) && pipeline.try_compute(worker))continue;
                if(packing){
                    const int index=next_pack.fetch_add(1,std::memory_order_relaxed);
                    if(index<int(exchange.tasks.size())){
                        exchange.pack(index,c.data());
                        continue;
                    }
                }
                if(pipeline.prepare_one())continue;
            }
            if(pipeline.done())break;
            // TC_SHARE_PULL: no TB update is left and every combine slice is taken
            if(1 && fused.consume_batch.load(std::memory_order_acquire)==13 && sharing.pull_output())continue;
            asm volatile("yield":::"memory");
        }
    }
    pending_b.finish();
    sharing.finish();
    tc1_aligned_pipeline=nullptr;
}

static void tc1(std::vector<double> &a, std::vector<double> &b, std::vector<double> &c, std::vector<double> &out,
                MPI_Comm comm, int s, int t, int u, int p, int q, int r, int pl, int ql, int rank, int size) {
    if (s != 4 || t != 4000 || u != 4000 || p != 4000 || q != 4000 || r != 4 || pl != 500 || ql != 500 || size != 8)
        std::abort();
    tc1_aligned_ready(a, b, c, out, comm, rank);
}

static void tc2(std::vector<double> &a, std::vector<double> &b, std::vector<double> &c, std::vector<double> &out,
                MPI_Comm comm, int s, int p, int t, int q, int u, int r, int pl, int ql, int rank, int size) {
    if (s != 8000 || p != 8000 || t != 2 || q != 4000 || u != 4000 || r != 2 || pl != 1000 || ql != 500 || size != 8)
        std::abort();
    tc2_prep_ac_fused(a, b, c, out, comm, rank);
}
} // namespace tcopt
