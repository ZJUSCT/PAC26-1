/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under a modified version of the MIT license. See LICENSE in the project root for license information.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <arm_bf16.h>
#include <arm_sme.h>
#include <arm_sve.h>
#include <cmath>
#include <cstdint>
#include <climits>
#include <linux/futex.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "Tensor.h"
#include "utils.h"

class MqaOpenMPTeam {
public:
    MqaOpenMPTeam()
    {
        const int place_count = omp_get_num_places();
        for (int place = 0; place < place_count; ++place) {
            const int count = omp_get_place_num_procs(place);
            std::vector<int> ids(count);
            omp_get_place_proc_ids(place, ids.data());
            for (int cpu : ids)
                if (std::find(cpus_.begin(), cpus_.end(), cpu) == cpus_.end())
                    cpus_.push_back(cpu);
        }
        if (cpus_.empty()) {
            cpu_set_t allowed;
            CPU_ZERO(&allowed);
            sched_getaffinity(0, sizeof(allowed), &allowed);
            for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
                if (CPU_ISSET(cpu, &allowed))
                    cpus_.push_back(cpu);
        }
        const int caller_cpu = sched_getcpu();
        const auto caller = std::find(cpus_.begin(), cpus_.end(), caller_cpu);
        if (caller != cpus_.end())
            std::iter_swap(cpus_.begin(), caller);
        thread_count_ = std::min<int>(
            {omp_get_max_threads(), static_cast<int>(cpus_.size()), 64});
        omp_pause_resource_all(omp_pause_hard);
        std::thread([this] {
#pragma omp parallel num_threads(thread_count_ - 1)
            worker_loop(omp_get_thread_num() + 1);
        }).detach();
        while (ready_.load(std::memory_order_acquire) != thread_count_ - 1)
            __asm__ volatile("yield");
    }

    void run(void *context, void (*task)(void *, int, int))
    {
        while (busy_.test_and_set(std::memory_order_acquire))
            __asm__ volatile("yield");
        context_ = context;
        task_ = task;
        const int generation =
            generation_.fetch_add(1, std::memory_order_release) + 1;
        if (sleepers_.load(std::memory_order_relaxed) != 0)
            syscall(SYS_futex, reinterpret_cast<int *>(&generation_),
                    FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
        task(context, 0, thread_count_);
        for (int id = 1; id < thread_count_; ++id)
            while (finished_[id].value.load(std::memory_order_acquire)
                   != generation)
                __asm__ volatile("yield");
        busy_.clear(std::memory_order_release);
    }

private:
    void worker_loop(int id)
    {
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        CPU_SET(cpus_[id], &affinity);
        pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
        int seen = generation_.load(std::memory_order_relaxed);
        ready_.fetch_add(1, std::memory_order_release);
        for (;;) {
            int current = seen;
            for (int spin = 0; spin < 8000000; ++spin) {
                current = generation_.load(std::memory_order_acquire);
                if (current != seen)
                    break;
                __asm__ volatile("yield");
            }
            if (current == seen) {
                sleepers_.fetch_add(1, std::memory_order_relaxed);
                current = generation_.load(std::memory_order_acquire);
                if (current == seen)
                    syscall(SYS_futex,
                            reinterpret_cast<int *>(&generation_),
                            FUTEX_WAIT_PRIVATE, seen, nullptr, nullptr, 0);
                sleepers_.fetch_sub(1, std::memory_order_relaxed);
                current = generation_.load(std::memory_order_acquire);
            }
            if (current == seen)
                continue;
            seen = current;
            task_(context_, id, thread_count_);
            finished_[id].value.store(seen, std::memory_order_release);
        }
    }

    struct alignas(64) Completion {
        std::atomic<int> value{0};
    };

    int thread_count_ = 1;
    std::vector<int> cpus_;
    std::atomic<int> generation_{0};
    Completion finished_[64];
    std::atomic<int> ready_{0};
    std::atomic<int> sleepers_{0};
    std::atomic_flag busy_ = ATOMIC_FLAG_INIT;
    void *context_ = nullptr;
    void (*task_)(void *, int, int) = nullptr;
};

static MqaOpenMPTeam *mqa_openmp_team = nullptr;

alignas(64) static const uint32_t mqa_pack_blocks_01[16] = {
    0, 1, 2, 3, 16, 17, 18, 19,
    4, 5, 6, 7, 20, 21, 22, 23,
};
alignas(64) static const uint32_t mqa_pack_blocks_23[16] = {
    8, 9, 10, 11, 24, 25, 26, 27,
    12, 13, 14, 15, 28, 29, 30, 31,
};
alignas(64) static const uint32_t mqa_combine_low[16] = {
    0, 1, 2, 3, 4, 5, 6, 7,
    16, 17, 18, 19, 20, 21, 22, 23,
};
alignas(64) static const uint32_t mqa_combine_high[16] = {
    8, 9, 10, 11, 12, 13, 14, 15,
    24, 25, 26, 27, 28, 29, 30, 31,
};
__arm_new("za")
__attribute__((noinline, aligned(16), preserve_none))
static void mqa_sme_pack_q1(const uint32_t *source,
                            uint32_t *destination,
                            const uint32_t *first_k)
    __arm_streaming
{
    const svbool_t all = svptrue_b32();
    for (int pair_group = 0; pair_group < 4; ++pair_group) {
        for (int head_group = 0; head_group < 4; ++head_group) {
            const int tile = pair_group * 4 + head_group;
            const uint32_t *tile_source = source
                + head_group * 16 * 64 + pair_group * 16;
            uint32_t *tile_destination = destination
                + (pair_group * 16 * 4 + head_group) * 16;
#pragma clang loop unroll_count(2)
            for (int row = 0; row < 16; ++row) {
                svld1_hor_za32(0, row, all, tile_source + row * 64);
                __asm__ volatile("prfm pldl2keep, [%0]"
                    : : "r"(first_k + (tile * 16 + row) * 16));
            }
#pragma clang loop unroll_count(2)
            for (int column = 0; column < 16; ++column)
                svst1_ver_za32(0, column, all,
                    tile_destination + column * 4 * 16);
        }
    }
}

__arm_new("za")
__attribute__((noinline, aligned(16), preserve_none))
static void mqa_sme_pack_q2(const uint32_t *source,
                            uint32_t *destination,
                            const uint32_t *first_k)
    __arm_streaming
{
    const svbool_t all = svptrue_b32();
    for (int next = 0; next < 2; ++next) {
        for (int pair_group = 0; pair_group < 4; ++pair_group) {
            for (int head_group = 0; head_group < 4; ++head_group) {
                const int tile = pair_group * 4 + head_group;
                const uint32_t *tile_source = source + next * 64 * 64
                    + head_group * 16 * 64 + pair_group * 16;
                uint32_t *tile_destination = destination + next * 64 * 64
                    + (pair_group * 16 * 4 + head_group) * 16;
#pragma clang loop unroll_count(2)
                for (int row = 0; row < 16; ++row) {
                    svld1_hor_za32(0, row, all,
                                   tile_source + row * 64);
                    if (next == 0)
                        __asm__ volatile("prfm pldl2keep, [%0]"
                            : : "r"(first_k + (tile * 16 + row) * 16));
                }
#pragma clang loop unroll_count(2)
                for (int column = 0; column < 16; ++column)
                    svst1_ver_za32(0, column, all,
                        tile_destination + column * 4 * 16);
            }
        }
    }
}

__attribute__((always_inline)) static inline void mqa_sme_paged_block_n1_sve(
    const uint32_t* source_k,
    const uint32_t* next_k,
    const uint32_t* packed_q,
    const float* weights,
    float* output,
    int valid_tokens,
    bool prefetch_next,
    bool vnum_q) __arm_streaming __arm_inout("za") {
  const svbool_t all_u32 = svptrue_b32();
  const svbool_t all_bf16 = svptrue_b16();
  const svbool_t all_f32 = svptrue_b32();
  const svfloat32_t inactive = svundef_f32();
  const svfloat32_t zero = svdup_n_f32(0.0f);
  const int token_groups = vnum_q ? (valid_tokens + 15) / 16 : 4;
#pragma clang loop unroll(disable)
  for (int token_group = 0; token_group < token_groups; ++token_group) {
    if (token_group != 0)
      svzero_za();
#pragma clang loop unroll(disable)
    for (int pair_group = 0; pair_group < 4; ++pair_group) {
      const uint32_t* s = source_k + token_group * 16 * 64 + pair_group * 16;
      svuint32_t v0, v1, v2, v3, v4, v5, v6, v7;
      svuint32_t v8, v9, v10, v11, v12, v13, v14, v15;
      {
        const svuint32_t in_a = svld1_u32(all_u32, s + (0) * 64);
        const svuint32_t in_b = svld1_u32(all_u32, s + (1) * 64);
        svuint32_t out_a;
        svuint32_t out_b;
        __asm__ volatile(
            "trn1 %0.s, %2.s, %3.s\n\t"
            "trn2 %1.s, %2.s, %3.s"
            : "=&w"(out_a), "=&w"(out_b)
            : "w"(in_a), "w"(in_b));
        v0 = out_a;
        v1 = out_b;
      }
      {
        const svuint32_t in_a = svld1_u32(all_u32, s + (2) * 64);
        const svuint32_t in_b = svld1_u32(all_u32, s + (3) * 64);
        svuint32_t out_a;
        svuint32_t out_b;
        __asm__ volatile(
            "trn1 %0.s, %2.s, %3.s\n\t"
            "trn2 %1.s, %2.s, %3.s"
            : "=&w"(out_a), "=&w"(out_b)
            : "w"(in_a), "w"(in_b));
        v2 = out_a;
        v3 = out_b;
      }
      {
        const svuint32_t in_a = svld1_u32(all_u32, s + (4) * 64);
        const svuint32_t in_b = svld1_u32(all_u32, s + (5) * 64);
        svuint32_t out_a;
        svuint32_t out_b;
        __asm__ volatile(
            "trn1 %0.s, %2.s, %3.s\n\t"
            "trn2 %1.s, %2.s, %3.s"
            : "=&w"(out_a), "=&w"(out_b)
            : "w"(in_a), "w"(in_b));
        v4 = out_a;
        v5 = out_b;
      }
      {
        const svuint32_t in_a = svld1_u32(all_u32, s + (6) * 64);
        const svuint32_t in_b = svld1_u32(all_u32, s + (7) * 64);
        svuint32_t out_a;
        svuint32_t out_b;
        __asm__ volatile(
            "trn1 %0.s, %2.s, %3.s\n\t"
            "trn2 %1.s, %2.s, %3.s"
            : "=&w"(out_a), "=&w"(out_b)
            : "w"(in_a), "w"(in_b));
        v6 = out_a;
        v7 = out_b;
      }
      {
        const svuint32_t in_a = svld1_u32(all_u32, s + (8) * 64);
        const svuint32_t in_b = svld1_u32(all_u32, s + (9) * 64);
        svuint32_t out_a;
        svuint32_t out_b;
        __asm__ volatile(
            "trn1 %0.s, %2.s, %3.s\n\t"
            "trn2 %1.s, %2.s, %3.s"
            : "=&w"(out_a), "=&w"(out_b)
            : "w"(in_a), "w"(in_b));
        v8 = out_a;
        v9 = out_b;
      }
      {
        const svuint32_t in_a = svld1_u32(all_u32, s + (10) * 64);
        const svuint32_t in_b = svld1_u32(all_u32, s + (11) * 64);
        svuint32_t out_a;
        svuint32_t out_b;
        __asm__ volatile(
            "trn1 %0.s, %2.s, %3.s\n\t"
            "trn2 %1.s, %2.s, %3.s"
            : "=&w"(out_a), "=&w"(out_b)
            : "w"(in_a), "w"(in_b));
        v10 = out_a;
        v11 = out_b;
      }
      {
        const svuint32_t in_a = svld1_u32(all_u32, s + (12) * 64);
        const svuint32_t in_b = svld1_u32(all_u32, s + (13) * 64);
        svuint32_t out_a;
        svuint32_t out_b;
        __asm__ volatile(
            "trn1 %0.s, %2.s, %3.s\n\t"
            "trn2 %1.s, %2.s, %3.s"
            : "=&w"(out_a), "=&w"(out_b)
            : "w"(in_a), "w"(in_b));
        v12 = out_a;
        v13 = out_b;
      }
      {
        const svuint32_t in_a = svld1_u32(all_u32, s + (14) * 64);
        const svuint32_t in_b = svld1_u32(all_u32, s + (15) * 64);
        svuint32_t out_a;
        svuint32_t out_b;
        __asm__ volatile(
            "trn1 %0.s, %2.s, %3.s\n\t"
            "trn2 %1.s, %2.s, %3.s"
            : "=&w"(out_a), "=&w"(out_b)
            : "w"(in_a), "w"(in_b));
        v14 = out_a;
        v15 = out_b;
      }
      {
        svuint32_t out_a;
        svuint32_t out_b;
        __asm__ volatile(
            "trn1 %0.d, %2.d, %3.d\n\t"
            "trn2 %1.d, %2.d, %3.d"
            : "=&w"(out_a), "=&w"(out_b)
            : "w"(v0), "w"(v2));
        v0 = out_a;
        v2 = out_b;
      }
      {
        svuint32_t out_a;
        svuint32_t out_b;
        __asm__ volatile(
            "trn1 %0.d, %2.d, %3.d\n\t"
            "trn2 %1.d, %2.d, %3.d"
            : "=&w"(out_a), "=&w"(out_b)
            : "w"(v1), "w"(v3));
        v1 = out_a;
        v3 = out_b;
      }
      {
        svuint32_t out_a;
        svuint32_t out_b;
        __asm__ volatile(
            "trn1 %0.d, %2.d, %3.d\n\t"
            "trn2 %1.d, %2.d, %3.d"
            : "=&w"(out_a), "=&w"(out_b)
            : "w"(v4), "w"(v6));
        v4 = out_a;
        v6 = out_b;
      }
      {
        svuint32_t out_a;
        svuint32_t out_b;
        __asm__ volatile(
            "trn1 %0.d, %2.d, %3.d\n\t"
            "trn2 %1.d, %2.d, %3.d"
            : "=&w"(out_a), "=&w"(out_b)
            : "w"(v5), "w"(v7));
        v5 = out_a;
        v7 = out_b;
      }
      {
        svuint32_t out_a;
        svuint32_t out_b;
        __asm__ volatile(
            "trn1 %0.d, %2.d, %3.d\n\t"
            "trn2 %1.d, %2.d, %3.d"
            : "=&w"(out_a), "=&w"(out_b)
            : "w"(v8), "w"(v10));
        v8 = out_a;
        v10 = out_b;
      }
      {
        svuint32_t out_a;
        svuint32_t out_b;
        __asm__ volatile(
            "trn1 %0.d, %2.d, %3.d\n\t"
            "trn2 %1.d, %2.d, %3.d"
            : "=&w"(out_a), "=&w"(out_b)
            : "w"(v9), "w"(v11));
        v9 = out_a;
        v11 = out_b;
      }
      {
        svuint32_t out_a;
        svuint32_t out_b;
        __asm__ volatile(
            "trn1 %0.d, %2.d, %3.d\n\t"
            "trn2 %1.d, %2.d, %3.d"
            : "=&w"(out_a), "=&w"(out_b)
            : "w"(v12), "w"(v14));
        v12 = out_a;
        v14 = out_b;
      }
      {
        svuint32_t out_a;
        svuint32_t out_b;
        __asm__ volatile(
            "trn1 %0.d, %2.d, %3.d\n\t"
            "trn2 %1.d, %2.d, %3.d"
            : "=&w"(out_a), "=&w"(out_b)
            : "w"(v13), "w"(v15));
        v13 = out_a;
        v15 = out_b;
      }
      const uint32_t* pf = nullptr;
      if (prefetch_next) {
        pf = next_k + (token_group * 4 + pair_group) * 16 * 16;
        __asm__ volatile(
            "prfm pldl2keep, [%0]\n\t"
            "prfm pldl2keep, [%0, #64]\n\t"
            "prfm pldl2keep, [%0, #128]\n\t"
            "prfm pldl2keep, [%0, #192]"
            :
            : "r"(pf + 0 * 64));
        __asm__ volatile(
            "prfm pldl2keep, [%0]\n\t"
            "prfm pldl2keep, [%0, #64]\n\t"
            "prfm pldl2keep, [%0, #128]\n\t"
            "prfm pldl2keep, [%0, #192]"
            :
            : "r"(pf + 1 * 64));
      }
      {
        const svuint32x2_t upper = svcreate2_u32(v0, v4);
        const svuint32x2_t lower = svcreate2_u32(v8, v12);
        const svuint32_t u01 = svtbl2_u32(upper, svld1_u32(all_u32, mqa_pack_blocks_01));
        const svuint32_t l01 = svtbl2_u32(lower, svld1_u32(all_u32, mqa_pack_blocks_01));
        const svuint32x2_t b01 = svcreate2_u32(u01, l01);
        const svuint32_t k01_low = svtbl2_u32(b01, svld1_u32(all_u32, mqa_combine_low));
        const svuint32_t k01_high = svtbl2_u32(b01, svld1_u32(all_u32, mqa_combine_high));
        __asm__ volatile("" : : "w"(k01_low), "w"(k01_high));
        {
          const svbfloat16_t k = svreinterpret_bf16_u32(k01_low);
          const bfloat16_t* q_base =
              reinterpret_cast<const bfloat16_t*>(packed_q + (pair_group * 16 + 0) * 4 * 16);
          if (vnum_q) {
            svbfloat16_t qv = svld1_vnum_bf16(all_bf16, q_base, 0);
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 1);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 2);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 3);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, qv, k);
          } else {
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 0 * 32), k);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 1 * 32), k);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 2 * 32), k);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 3 * 32), k);
          }
        }
        {
          const svbfloat16_t k = svreinterpret_bf16_u32(k01_high);
          const bfloat16_t* q_base =
              reinterpret_cast<const bfloat16_t*>(packed_q + (pair_group * 16 + 0 + 4) * 4 * 16);
          if (vnum_q) {
            svbfloat16_t qv = svld1_vnum_bf16(all_bf16, q_base, 0);
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 1);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 2);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 3);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, qv, k);
          } else {
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 0 * 32), k);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 1 * 32), k);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 2 * 32), k);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 3 * 32), k);
          }
        }
        const svuint32_t u23 = svtbl2_u32(upper, svld1_u32(all_u32, mqa_pack_blocks_23));
        const svuint32_t l23 = svtbl2_u32(lower, svld1_u32(all_u32, mqa_pack_blocks_23));
        const svuint32x2_t b23 = svcreate2_u32(u23, l23);
        const svuint32_t k23_low = svtbl2_u32(b23, svld1_u32(all_u32, mqa_combine_low));
        const svuint32_t k23_high = svtbl2_u32(b23, svld1_u32(all_u32, mqa_combine_high));
        __asm__ volatile("" : : "w"(k23_low), "w"(k23_high));
        {
          const svbfloat16_t k = svreinterpret_bf16_u32(k23_low);
          const bfloat16_t* q_base =
              reinterpret_cast<const bfloat16_t*>(packed_q + (pair_group * 16 + 0 + 8) * 4 * 16);
          if (vnum_q) {
            svbfloat16_t qv = svld1_vnum_bf16(all_bf16, q_base, 0);
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 1);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 2);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 3);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, qv, k);
          } else {
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 0 * 32), k);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 1 * 32), k);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 2 * 32), k);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 3 * 32), k);
          }
        }
        {
          const svbfloat16_t k = svreinterpret_bf16_u32(k23_high);
          const bfloat16_t* q_base =
              reinterpret_cast<const bfloat16_t*>(packed_q + (pair_group * 16 + 0 + 12) * 4 * 16);
          if (vnum_q) {
            svbfloat16_t qv = svld1_vnum_bf16(all_bf16, q_base, 0);
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 1);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 2);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 3);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, qv, k);
          } else {
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 0 * 32), k);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 1 * 32), k);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 2 * 32), k);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 3 * 32), k);
          }
        }
      }
      {
        const svuint32x2_t upper = svcreate2_u32(v1, v5);
        const svuint32x2_t lower = svcreate2_u32(v9, v13);
        const svuint32_t u01 = svtbl2_u32(upper, svld1_u32(all_u32, mqa_pack_blocks_01));
        const svuint32_t l01 = svtbl2_u32(lower, svld1_u32(all_u32, mqa_pack_blocks_01));
        const svuint32x2_t b01 = svcreate2_u32(u01, l01);
        const svuint32_t k01_low = svtbl2_u32(b01, svld1_u32(all_u32, mqa_combine_low));
        const svuint32_t k01_high = svtbl2_u32(b01, svld1_u32(all_u32, mqa_combine_high));
        __asm__ volatile("" : : "w"(k01_low), "w"(k01_high));
        {
          const svbfloat16_t k = svreinterpret_bf16_u32(k01_low);
          const bfloat16_t* q_base =
              reinterpret_cast<const bfloat16_t*>(packed_q + (pair_group * 16 + 1) * 4 * 16);
          if (vnum_q) {
            svbfloat16_t qv = svld1_vnum_bf16(all_bf16, q_base, 0);
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 1);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 2);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 3);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, qv, k);
          } else {
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 0 * 32), k);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 1 * 32), k);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 2 * 32), k);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 3 * 32), k);
          }
        }
        {
          const svbfloat16_t k = svreinterpret_bf16_u32(k01_high);
          const bfloat16_t* q_base =
              reinterpret_cast<const bfloat16_t*>(packed_q + (pair_group * 16 + 1 + 4) * 4 * 16);
          if (vnum_q) {
            svbfloat16_t qv = svld1_vnum_bf16(all_bf16, q_base, 0);
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 1);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 2);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 3);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, qv, k);
          } else {
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 0 * 32), k);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 1 * 32), k);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 2 * 32), k);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 3 * 32), k);
          }
        }
        const svuint32_t u23 = svtbl2_u32(upper, svld1_u32(all_u32, mqa_pack_blocks_23));
        const svuint32_t l23 = svtbl2_u32(lower, svld1_u32(all_u32, mqa_pack_blocks_23));
        const svuint32x2_t b23 = svcreate2_u32(u23, l23);
        const svuint32_t k23_low = svtbl2_u32(b23, svld1_u32(all_u32, mqa_combine_low));
        const svuint32_t k23_high = svtbl2_u32(b23, svld1_u32(all_u32, mqa_combine_high));
        __asm__ volatile("" : : "w"(k23_low), "w"(k23_high));
        {
          const svbfloat16_t k = svreinterpret_bf16_u32(k23_low);
          const bfloat16_t* q_base =
              reinterpret_cast<const bfloat16_t*>(packed_q + (pair_group * 16 + 1 + 8) * 4 * 16);
          if (vnum_q) {
            svbfloat16_t qv = svld1_vnum_bf16(all_bf16, q_base, 0);
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 1);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 2);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 3);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, qv, k);
          } else {
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 0 * 32), k);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 1 * 32), k);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 2 * 32), k);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 3 * 32), k);
          }
        }
        {
          const svbfloat16_t k = svreinterpret_bf16_u32(k23_high);
          const bfloat16_t* q_base =
              reinterpret_cast<const bfloat16_t*>(packed_q + (pair_group * 16 + 1 + 12) * 4 * 16);
          if (vnum_q) {
            svbfloat16_t qv = svld1_vnum_bf16(all_bf16, q_base, 0);
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 1);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 2);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 3);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, qv, k);
          } else {
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 0 * 32), k);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 1 * 32), k);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 2 * 32), k);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 3 * 32), k);
          }
        }
      }
      if (prefetch_next) {
        __asm__ volatile(
            "prfm pldl2keep, [%0]\n\t"
            "prfm pldl2keep, [%0, #64]\n\t"
            "prfm pldl2keep, [%0, #128]\n\t"
            "prfm pldl2keep, [%0, #192]"
            :
            : "r"(pf + 2 * 64));
        __asm__ volatile(
            "prfm pldl2keep, [%0]\n\t"
            "prfm pldl2keep, [%0, #64]\n\t"
            "prfm pldl2keep, [%0, #128]\n\t"
            "prfm pldl2keep, [%0, #192]"
            :
            : "r"(pf + 3 * 64));
      }
      {
        const svuint32x2_t upper = svcreate2_u32(v2, v6);
        const svuint32x2_t lower = svcreate2_u32(v10, v14);
        const svuint32_t u01 = svtbl2_u32(upper, svld1_u32(all_u32, mqa_pack_blocks_01));
        const svuint32_t l01 = svtbl2_u32(lower, svld1_u32(all_u32, mqa_pack_blocks_01));
        const svuint32x2_t b01 = svcreate2_u32(u01, l01);
        const svuint32_t k01_low = svtbl2_u32(b01, svld1_u32(all_u32, mqa_combine_low));
        const svuint32_t k01_high = svtbl2_u32(b01, svld1_u32(all_u32, mqa_combine_high));
        __asm__ volatile("" : : "w"(k01_low), "w"(k01_high));
        {
          const svbfloat16_t k = svreinterpret_bf16_u32(k01_low);
          const bfloat16_t* q_base =
              reinterpret_cast<const bfloat16_t*>(packed_q + (pair_group * 16 + 2) * 4 * 16);
          if (vnum_q) {
            svbfloat16_t qv = svld1_vnum_bf16(all_bf16, q_base, 0);
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 1);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 2);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 3);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, qv, k);
          } else {
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 0 * 32), k);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 1 * 32), k);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 2 * 32), k);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 3 * 32), k);
          }
        }
        {
          const svbfloat16_t k = svreinterpret_bf16_u32(k01_high);
          const bfloat16_t* q_base =
              reinterpret_cast<const bfloat16_t*>(packed_q + (pair_group * 16 + 2 + 4) * 4 * 16);
          if (vnum_q) {
            svbfloat16_t qv = svld1_vnum_bf16(all_bf16, q_base, 0);
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 1);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 2);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 3);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, qv, k);
          } else {
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 0 * 32), k);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 1 * 32), k);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 2 * 32), k);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 3 * 32), k);
          }
        }
        const svuint32_t u23 = svtbl2_u32(upper, svld1_u32(all_u32, mqa_pack_blocks_23));
        const svuint32_t l23 = svtbl2_u32(lower, svld1_u32(all_u32, mqa_pack_blocks_23));
        const svuint32x2_t b23 = svcreate2_u32(u23, l23);
        const svuint32_t k23_low = svtbl2_u32(b23, svld1_u32(all_u32, mqa_combine_low));
        const svuint32_t k23_high = svtbl2_u32(b23, svld1_u32(all_u32, mqa_combine_high));
        __asm__ volatile("" : : "w"(k23_low), "w"(k23_high));
        {
          const svbfloat16_t k = svreinterpret_bf16_u32(k23_low);
          const bfloat16_t* q_base =
              reinterpret_cast<const bfloat16_t*>(packed_q + (pair_group * 16 + 2 + 8) * 4 * 16);
          if (vnum_q) {
            svbfloat16_t qv = svld1_vnum_bf16(all_bf16, q_base, 0);
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 1);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 2);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 3);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, qv, k);
          } else {
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 0 * 32), k);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 1 * 32), k);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 2 * 32), k);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 3 * 32), k);
          }
        }
        {
          const svbfloat16_t k = svreinterpret_bf16_u32(k23_high);
          const bfloat16_t* q_base =
              reinterpret_cast<const bfloat16_t*>(packed_q + (pair_group * 16 + 2 + 12) * 4 * 16);
          if (vnum_q) {
            svbfloat16_t qv = svld1_vnum_bf16(all_bf16, q_base, 0);
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 1);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 2);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 3);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, qv, k);
          } else {
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 0 * 32), k);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 1 * 32), k);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 2 * 32), k);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 3 * 32), k);
          }
        }
      }
      {
        const svuint32x2_t upper = svcreate2_u32(v3, v7);
        const svuint32x2_t lower = svcreate2_u32(v11, v15);
        const svuint32_t u01 = svtbl2_u32(upper, svld1_u32(all_u32, mqa_pack_blocks_01));
        const svuint32_t l01 = svtbl2_u32(lower, svld1_u32(all_u32, mqa_pack_blocks_01));
        const svuint32x2_t b01 = svcreate2_u32(u01, l01);
        const svuint32_t k01_low = svtbl2_u32(b01, svld1_u32(all_u32, mqa_combine_low));
        const svuint32_t k01_high = svtbl2_u32(b01, svld1_u32(all_u32, mqa_combine_high));
        __asm__ volatile("" : : "w"(k01_low), "w"(k01_high));
        {
          const svbfloat16_t k = svreinterpret_bf16_u32(k01_low);
          const bfloat16_t* q_base =
              reinterpret_cast<const bfloat16_t*>(packed_q + (pair_group * 16 + 3) * 4 * 16);
          if (vnum_q) {
            svbfloat16_t qv = svld1_vnum_bf16(all_bf16, q_base, 0);
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 1);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 2);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 3);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, qv, k);
          } else {
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 0 * 32), k);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 1 * 32), k);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 2 * 32), k);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 3 * 32), k);
          }
        }
        {
          const svbfloat16_t k = svreinterpret_bf16_u32(k01_high);
          const bfloat16_t* q_base =
              reinterpret_cast<const bfloat16_t*>(packed_q + (pair_group * 16 + 3 + 4) * 4 * 16);
          if (vnum_q) {
            svbfloat16_t qv = svld1_vnum_bf16(all_bf16, q_base, 0);
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 1);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 2);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 3);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, qv, k);
          } else {
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 0 * 32), k);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 1 * 32), k);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 2 * 32), k);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 3 * 32), k);
          }
        }
        const svuint32_t u23 = svtbl2_u32(upper, svld1_u32(all_u32, mqa_pack_blocks_23));
        const svuint32_t l23 = svtbl2_u32(lower, svld1_u32(all_u32, mqa_pack_blocks_23));
        const svuint32x2_t b23 = svcreate2_u32(u23, l23);
        const svuint32_t k23_low = svtbl2_u32(b23, svld1_u32(all_u32, mqa_combine_low));
        const svuint32_t k23_high = svtbl2_u32(b23, svld1_u32(all_u32, mqa_combine_high));
        __asm__ volatile("" : : "w"(k23_low), "w"(k23_high));
        {
          const svbfloat16_t k = svreinterpret_bf16_u32(k23_low);
          const bfloat16_t* q_base =
              reinterpret_cast<const bfloat16_t*>(packed_q + (pair_group * 16 + 3 + 8) * 4 * 16);
          if (vnum_q) {
            svbfloat16_t qv = svld1_vnum_bf16(all_bf16, q_base, 0);
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 1);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 2);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 3);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, qv, k);
          } else {
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 0 * 32), k);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 1 * 32), k);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 2 * 32), k);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 3 * 32), k);
          }
        }
        {
          const svbfloat16_t k = svreinterpret_bf16_u32(k23_high);
          const bfloat16_t* q_base =
              reinterpret_cast<const bfloat16_t*>(packed_q + (pair_group * 16 + 3 + 12) * 4 * 16);
          if (vnum_q) {
            svbfloat16_t qv = svld1_vnum_bf16(all_bf16, q_base, 0);
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 1);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 2);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, qv, k);
            qv = svld1_vnum_bf16(all_bf16, q_base, 3);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, qv, k);
          } else {
            svmopa_za32_bf16_m(0, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 0 * 32), k);
            svmopa_za32_bf16_m(1, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 1 * 32), k);
            svmopa_za32_bf16_m(2, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 2 * 32), k);
            svmopa_za32_bf16_m(3, all_bf16, all_bf16, svld1_bf16(all_bf16, q_base + 3 * 32), k);
          }
        }
      }
    }
    svfloat32_t r0 = zero;
    svfloat32_t r1 = zero;
    svfloat32_t r2 = zero;
    svfloat32_t r3 = zero;
#pragma clang loop unroll_count(2)
    for (int row = 0; row < 16; ++row) {
      svfloat32_t v0 = svread_hor_za32_f32_m(inactive, all_f32, 0, row);
      __asm__ volatile("fmax %0.s, %1/m, %0.s, #0.0" : "+w"(v0) : "Upl"(all_f32));
      r0 = svmla_f32_x(all_f32, r0, v0, svdup_n_f32(weights[row]));
      svfloat32_t v1 = svread_hor_za32_f32_m(inactive, all_f32, 1, row);
      __asm__ volatile("fmax %0.s, %1/m, %0.s, #0.0" : "+w"(v1) : "Upl"(all_f32));
      r1 = svmla_f32_x(all_f32, r1, v1, svdup_n_f32(weights[16 + row]));
      svfloat32_t v2 = svread_hor_za32_f32_m(inactive, all_f32, 2, row);
      __asm__ volatile("fmax %0.s, %1/m, %0.s, #0.0" : "+w"(v2) : "Upl"(all_f32));
      r2 = svmla_f32_x(all_f32, r2, v2, svdup_n_f32(weights[32 + row]));
      svfloat32_t v3 = svread_hor_za32_f32_m(inactive, all_f32, 3, row);
      __asm__ volatile("fmax %0.s, %1/m, %0.s, #0.0" : "+w"(v3) : "Upl"(all_f32));
      r3 = svmla_f32_x(all_f32, r3, v3, svdup_n_f32(weights[48 + row]));
    }
    const svfloat32_t reduced =
        svadd_f32_x(all_f32, svadd_f32_x(all_f32, r0, r1), svadd_f32_x(all_f32, r2, r3));
    const int64_t offset = token_group * 16;
    svst1_f32(svwhilelt_b32(offset, int64_t(valid_tokens)), output + offset, reduced);
  }
}
__arm_new("za") __attribute__((noinline, aligned(16), preserve_none))
static void mqa_sme_paged_range_n1(const bfloat16_t *kv,
                                   const int64_t *table,
                                   int64_t logical_begin,
                                   int64_t logical_end,
                                   int64_t num_blocks,
                                   int64_t context_len,
                                   const uint32_t *packed_q,
                                   const float *weights,
                                   float *output)
    __arm_streaming
{
    for (int64_t logical = logical_begin; logical < logical_end; ++logical) {
        if (logical != logical_begin)
            svzero_za();
        const uint32_t *source = reinterpret_cast<const uint32_t *>(
            kv + table[logical] * 64 * 128);
        const uint32_t *next_source = reinterpret_cast<const uint32_t *>(
            kv + table[std::min<int64_t>(logical + 1, num_blocks - 1)]
                * 64 * 128);
        const int64_t token = logical * 64;
        const int valid = static_cast<int>(std::min<int64_t>(64,
            std::max<int64_t>(0, context_len - token)));
        mqa_sme_paged_block_n1_sve(source, next_source, packed_q, weights,
                                   output + token, valid, true, false);
    }
}

__arm_new("za") __attribute__((noinline, aligned(16), preserve_none))
static void mqa_sme_paged_range_n2(const bfloat16_t *kv,
                                   const int64_t *table,
                                   int64_t logical_begin,
                                   int64_t logical_end,
                                   int64_t num_blocks,
                                   int64_t context_len,
                                   const uint32_t *packed_q0,
                                   const float *weights0,
                                   float *output0,
                                   const uint32_t *packed_q1,
                                   const float *weights1,
                                   float *output1)
    __arm_streaming
{
    for (int64_t logical = logical_begin; logical < logical_end; ++logical) {
        if (logical != logical_begin)
            svzero_za();
        const uint32_t *source = reinterpret_cast<const uint32_t *>(
            kv + table[logical] * 64 * 128);
        const uint32_t *next_source = reinterpret_cast<const uint32_t *>(
            kv + table[std::min<int64_t>(logical + 1, num_blocks - 1)]
                * 64 * 128);
        const int64_t token = logical * 64;
        const int valid0 = static_cast<int>(std::min<int64_t>(64,
            std::max<int64_t>(0, context_len - 1 - token)));
        const int valid1 = static_cast<int>(std::min<int64_t>(64,
            std::max<int64_t>(0, context_len - token)));
        mqa_sme_paged_block_n1_sve(source, next_source, packed_q0, weights0,
                                   output0 + token, valid0, true, true);
        svzero_za();
        mqa_sme_paged_block_n1_sve(source, nullptr, packed_q1, weights1,
                                   output1 + token, valid1, false, true);
    }
}

template<int NextN>
static inline void indexer_bf16_paged_mqa_logits_impl(
    const Tensor<bfloat16_t, 4> &q,                                     // [batch_size, next_n, num_heads, dim]
    const Tensor<bfloat16_t, 4> &kv_cache,                              // [num_blocks, block_size, 1, dim]
    const Tensor<int64_t, 2> &block_tables,                             // [batch_size, max_num_blocks]
    const Tensor<int64_t, 1> &context_lens,                             // [batch_size]
    const Tensor<float, 2> &weights,                                    // [batch_size * next_n, num_heads]
    const Tensor<float, 2> &output,                                     // [batch_size * next_n, max_model_len] (output)
    int64_t batch_size,
    int64_t num_heads,
    int64_t dim,
    int64_t block_size,
    int64_t max_model_len
) {
    constexpr int64_t next_n = NextN;
    const bfloat16_t *q_ptr = q.data_ptr();
    const bfloat16_t *kv_ptr = kv_cache.data_ptr();
    const int64_t *table_ptr = block_tables.data_ptr();
    const int64_t *lens_ptr = context_lens.data_ptr();
    const float *weight_ptr = weights.data_ptr();
    float *out_ptr = output.data_ptr();
    const int64_t max_num_blocks = block_tables.size(1);
    // Hip12 exposes a 512-bit SVE vector: 16 FP32 results or 32 BF16
    // operands.  One BFDOT therefore computes two dimensions for 16 tokens.
    assert(svcntw() == 16 && block_size == 64 && dim == 128);
    constexpr int64_t block_chunk = 1;
    int work_offsets[129];
    work_offsets[0] = 0;
    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int64_t num_blocks = ceil_div(lens_ptr[batch_idx], block_size);
        work_offsets[batch_idx + 1] = work_offsets[batch_idx]
            + static_cast<int>(ceil_div(num_blocks, block_chunk));
    }
    const int work_count = work_offsets[batch_size];
    auto run_thread = [&](int thread_id, int team_size) {
        const int active_threads = team_size;
        const int64_t q_items = batch_size * next_n;
        const int64_t q_base = q_items / active_threads;
        const int64_t q_extra = q_items % active_threads;
        const int q_thread = active_threads - 1 - thread_id;
        const int64_t q_begin =
            q_thread * q_base + std::min<int64_t>(q_thread, q_extra);
        const int64_t q_end = q_begin + q_base + (q_thread < q_extra);
        const int work_base = work_count / active_threads;
        const int work_extra = work_count % active_threads;
        const int work_begin =
            thread_id * work_base + std::min(thread_id, work_extra);
        const int work_end =
            work_begin + work_base + (thread_id < work_extra);
        if (thread_id < active_threads) {
            alignas(NextN == 2 ? 256 : 128)
                uint32_t local_q[NextN * 64 * 64];
            int64_t packed_batch = -1;
            int64_t batch_idx = 0;
            while (batch_idx < batch_size
                   && work_offsets[batch_idx + 1] <= work_begin)
                ++batch_idx;
            int work_idx = work_begin;
            while (work_idx < work_end) {
            while (work_offsets[batch_idx + 1] <= work_idx)
                ++batch_idx;
            const int64_t block_begin =
                (work_idx - work_offsets[batch_idx]) * block_chunk;
            const int batch_work_end =
                std::min(work_end, work_offsets[batch_idx + 1]);
            if (batch_idx != packed_batch) {
                packed_batch = batch_idx;
                const uint32_t *first_source =
                    reinterpret_cast<const uint32_t *>(
                        kv_ptr + table_ptr[batch_idx * max_num_blocks
                            + block_begin] * block_size * dim);
                const int64_t first_task = batch_idx * next_n;
                if constexpr (NextN == 2)
                    mqa_sme_pack_q2(
                        reinterpret_cast<const uint32_t *>(
                            q_ptr + first_task * num_heads * dim),
                        local_q, first_source);
                else
                    mqa_sme_pack_q1(
                        reinterpret_cast<const uint32_t *>(
                            q_ptr + first_task * num_heads * dim),
                        local_q, first_source);
            }
            const int64_t context_len = lens_ptr[batch_idx];
            const int64_t num_blocks = ceil_div(context_len, block_size);
            const int64_t block_end = std::min<int64_t>(
                num_blocks,
                (batch_work_end - work_offsets[batch_idx]) * block_chunk);

            const int64_t task0 = batch_idx * next_n;
            const int64_t *batch_table =
                table_ptr + batch_idx * max_num_blocks;
            if (next_n == 2)
                mqa_sme_paged_range_n2(
                    kv_ptr, batch_table, block_begin, block_end,
                    num_blocks, context_len,
                    local_q, weight_ptr + task0 * num_heads,
                    out_ptr + task0 * max_model_len,
                    local_q + 64 * 64,
                    weight_ptr + (task0 + 1) * num_heads,
                    out_ptr + (task0 + 1) * max_model_len);
            else
                mqa_sme_paged_range_n1(
                    kv_ptr, batch_table, block_begin, block_end,
                    num_blocks, context_len,
                    local_q, weight_ptr + task0 * num_heads,
                    out_ptr + task0 * max_model_len);
            work_idx = batch_work_end;
        }
        }
        if (thread_id < active_threads)
        for (int64_t task = q_begin; task < q_end; ++task) {
            const int64_t batch_idx = task / next_n;
            const int64_t next_idx = task - batch_idx * next_n;
            const int64_t valid_tokens =
                lens_ptr[batch_idx] - next_n + next_idx + 1;
            float *const output_end = out_ptr + (task + 1) * max_model_len;
            if constexpr (NextN == 2) {
                float *tail =
                    out_ptr + task * max_model_len + valid_tokens;
                int64_t remaining = output_end - tail;
                const int64_t vector_words = svcntw();
                const svbool_t all = svptrue_b32();
                const svfloat32_t negative_infinity =
                    svdup_n_f32(-INFINITY);
                if (remaining > 0
                    && (reinterpret_cast<uintptr_t>(tail) & 63) != 0) {
                    const int64_t prefix = std::min<int64_t>(
                        (64 - (reinterpret_cast<uintptr_t>(tail) & 63)) / 4,
                        remaining);
                    svst1_f32(svwhilelt_b32(int64_t(0), prefix), tail,
                              negative_infinity);
                    tail += prefix;
                    remaining -= prefix;
                }
                while (remaining >= 8 * vector_words) {
                    __asm__ volatile(
                        "stnt1w %0.s, %1, [%2]\n\t"
                        "stnt1w %0.s, %1, [%2, #1, mul vl]\n\t"
                        "stnt1w %0.s, %1, [%2, #2, mul vl]\n\t"
                        "stnt1w %0.s, %1, [%2, #3, mul vl]\n\t"
                        "stnt1w %0.s, %1, [%2, #4, mul vl]\n\t"
                        "stnt1w %0.s, %1, [%2, #5, mul vl]\n\t"
                        "stnt1w %0.s, %1, [%2, #6, mul vl]\n\t"
                        "stnt1w %0.s, %1, [%2, #7, mul vl]"
                        :
                        : "w"(negative_infinity), "Upl"(all), "r"(tail)
                        : "memory");
                    tail += 8 * vector_words;
                    remaining -= 8 * vector_words;
                }
                if (remaining >= 4 * vector_words) {
                    __asm__ volatile(
                        "stnt1w %0.s, %1, [%2]\n\t"
                        "stnt1w %0.s, %1, [%2, #1, mul vl]\n\t"
                        "stnt1w %0.s, %1, [%2, #2, mul vl]\n\t"
                        "stnt1w %0.s, %1, [%2, #3, mul vl]"
                        :
                        : "w"(negative_infinity), "Upl"(all), "r"(tail)
                        : "memory");
                    tail += 4 * vector_words;
                    remaining -= 4 * vector_words;
                }
                while (remaining > 0) {
                    const svbool_t active =
                        svwhilelt_b32(int64_t(0), remaining);
                    svst1_f32(active, tail, negative_infinity);
                    tail += vector_words;
                    remaining -= vector_words;
                }
            } else {
                std::fill(out_ptr + task * max_model_len + valid_tokens,
                          output_end, -INFINITY);
            }
        }
    };
    if (mqa_openmp_team == nullptr)
        mqa_openmp_team = new MqaOpenMPTeam;
    mqa_openmp_team->run(&run_thread,
              [](void *context, int thread_id, int team_size) {
        (*static_cast<decltype(run_thread) *>(context))(
            thread_id, team_size);
    });
}

inline void indexer_bf16_paged_mqa_logits(
    const Tensor<bfloat16_t, 4> &q,
    const Tensor<bfloat16_t, 4> &kv_cache,
    const Tensor<int64_t, 2> &block_tables,
    const Tensor<int64_t, 1> &context_lens,
    const Tensor<float, 2> &weights,
    const Tensor<float, 2> &output,
    int64_t batch_size,
    int64_t next_n,
    int64_t num_heads,
    int64_t dim,
    int64_t block_size,
    int64_t max_model_len)
{
    assert(next_n == 1 || next_n == 2);
    if (next_n == 2)
        indexer_bf16_paged_mqa_logits_impl<2>(
            q, kv_cache, block_tables, context_lens, weights, output,
            batch_size, num_heads, dim, block_size, max_model_len);
    else
        indexer_bf16_paged_mqa_logits_impl<1>(
            q, kv_cache, block_tables, context_lens, weights, output,
            batch_size, num_heads, dim, block_size, max_model_len);
}
