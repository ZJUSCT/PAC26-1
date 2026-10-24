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

#include <cstdlib>
#include <cassert>
#include <cstddef>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <atomic>
#include <limits>
#include <new>
#include <utility>
#include <type_traits>
#include <vector>
#include <arm_neon.h>
#include <arm_bf16.h>
#include <arm_sve.h>
#include <arm_sme.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <thread>

#include "ref_mqa_logits.h"
#include "allocator.h"
#include "Tensor.h"

namespace mqa_lat {

// Atomic read-modify-writes that many workers issue at once (claim and
// readiness counters) as single LSE instructions (ARMv8.1, part of ARMv9-A).
// With -march=armv9-a GCC emits these, while BiSheng clang emits
// load/store-exclusive retry loops, which serialize badly under contention.
static inline bool mqa_exchange_acquire(std::atomic<bool> &target, bool value)
{
    static_assert(sizeof(std::atomic<bool>) == 1);
    uint32_t old;
    asm volatile(".arch_extension lse\n" "swpab %w[v], %w[o], [%[p]]\n"
                 : [o] "=r"(old) : [v] "r"(uint32_t(value)), [p] "r"(&target) : "memory");
    return old != 0;
}

// fetch_add on the claim / readiness counters (BiSheng 5 emits ldxr/stxr loops).
static inline int64_t mqa_fetch_add_relaxed(std::atomic<int64_t> &target, int64_t value)
{
    static_assert(sizeof(std::atomic<int64_t>) == 8);
    int64_t old;
    asm volatile(".arch_extension lse\n" "ldadd %x[v], %x[o], [%[p]]\n"
                 : [o] "=r"(old) : [v] "r"(value), [p] "r"(&target) : "memory");
    return old;
}

static inline int64_t mqa_fetch_add_acq_rel(std::atomic<int64_t> &target, int64_t value)
{
    static_assert(sizeof(std::atomic<int64_t>) == 8);
    int64_t old;
    asm volatile(".arch_extension lse\n" "ldaddal %x[v], %x[o], [%[p]]\n"
                 : [o] "=r"(old) : [v] "r"(value), [p] "r"(&target) : "memory");
    return old;
}

// Persistent flat worker team. Workers spin on one generation word between
// calls and fall back to a futex sleep after a short idle period, so they do
// not compete with OpenMP regions outside the operator. It is only used when
// OpenMP workers sleep between regions (OMP_WAIT_POLICY=PASSIVE); otherwise
// both teams would spin on the same cores.
struct alignas(128) MqaPoolFlag {
    std::atomic<uint32_t> value{0};
};

// Pool worker found noisy at startup (-1: none).
inline std::atomic<int> mqa_quiet_worker{-1};

// Gaps over 2 us while reading the timer for ms milliseconds.
static inline uint32_t mqa_noise_probe(int ms)
{
    uint64_t frequency, t0;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(frequency));
    asm volatile("isb\nmrs %0, cntvct_el0" : "=r"(t0) :: "memory");
    const uint64_t end = t0 + frequency / 1000 * static_cast<uint64_t>(ms), gap = frequency / 500000;
    uint32_t gaps = 0;
    for (uint64_t prev = t0, now = t0; now < end; prev = now) {
        asm volatile("isb\nmrs %0, cntvct_el0" : "=r"(now) :: "memory");
        gaps += now - prev > gap;
    }
    return gaps;
}

class MqaPool {
public:
    bool try_acquire(int workers) {
        return wake_.workers == workers && !current_worker
            && !mqa_exchange_acquire(busy_, true);
    }
    void release() { busy_.store(false, std::memory_order_release); }

    template<class Body>
    void run(Body &body) {
        // Workers read the function and context from the generation's line.
        wake_.function = [](void *context, int tid, int workers) {
            (*static_cast<Body *>(context))(tid, workers);
        };
        wake_.context = &body;
        const uint32_t generation = wake_.generation.load(std::memory_order_relaxed) + 1;
        wake_.generation.store(generation, std::memory_order_seq_cst);
        if (sleepers_.value.load(std::memory_order_seq_cst) != 0)
            syscall(SYS_futex, &wake_.generation, FUTEX_WAKE_PRIVATE, INT32_MAX, nullptr, nullptr, 0);
        body(0, wake_.workers);
        for (int tid = 1; tid < wake_.workers; ++tid)
            while (done_[tid].value.load(std::memory_order_acquire) != generation)
                asm volatile("yield" ::: "memory");
    }

private:
    static constexpr int max_workers = 256;
    static inline thread_local bool current_worker = false;

    // Everything a worker reads to start a call is on the generation's line.
    struct alignas(128) Wake {
        std::atomic<uint32_t> generation{0};
        int workers = 0;
        void (*function)(void *, int, int) = nullptr;
        void *context = nullptr;
    } wake_;
    MqaPoolFlag sleepers_;
    alignas(128) std::atomic<bool> busy_{false};
    MqaPoolFlag done_[max_workers];
    // Startup gap counts of the start probe, how many reported, and
    // the start signal (after every thread exists: creating threads
    // interrupts all cores).
    std::atomic<uint32_t> noise_[max_workers] = {};
    std::atomic<int> probed_{0};
    std::atomic<bool> probe_go_{false};

    static bool passive_openmp() {
        const char *policy = std::getenv("OMP_WAIT_POLICY");
        if (!policy) return false;
        const char passive[] = "passive";
        for (int i = 0; ; ++i) {
            const char c = policy[i];
            const char lower = c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c;
            if (lower != passive[i]) return false;
            if (c == '\0') return true;
        }
    }

public:
    // Called only by the function-local static in the operator entry.
    // Worker t takes the place an OpenMP team with proc_bind(close) would
    // give thread t. Without one place per worker the team is not used.
    static MqaPool *create(int workers) {
        const bool verbose = std::getenv("MQA_POOL_VERBOSE") != nullptr;
        const int places = omp_get_num_places();
        if (workers < 2 || workers > max_workers || !passive_openmp() || places < workers) {
            if (verbose)
                std::fprintf(stderr, "MQA_POOL disabled workers=%d places=%d passive=%d\n",
                             workers, places, static_cast<int>(passive_openmp()));
            return nullptr;
        }
        const int first_place = std::max(0, omp_get_place_num());
        std::vector<cpu_set_t> sets(static_cast<size_t>(workers));
        for (int tid = 0; tid < workers; ++tid) {
            const int place = (first_place + tid) % places;
            const int count = omp_get_place_num_procs(place);
            if (count <= 0) return nullptr;
            std::vector<int> ids(static_cast<size_t>(count));
            omp_get_place_proc_ids(place, ids.data());
            CPU_ZERO(&sets[tid]);
            for (int id : ids)
                if (id >= 0 && id < CPU_SETSIZE) CPU_SET(id, &sets[tid]);
        }
        auto *pool = new MqaPool();  // Never destroyed: workers may spin at exit.
        pool->wake_.workers = workers;
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &sets[0]);
        for (int tid = 1; tid < workers; ++tid) {
            try {
                std::thread(worker_main, pool, tid, sets[tid]).detach();
            } catch (...) {
                // Started workers only spin on this leaked team and then sleep.
                return nullptr;
            }
        }
        {
            pool->probe_go_.store(true, std::memory_order_release);
            pool->noise_[0].store(mqa_noise_probe(20), std::memory_order_relaxed);
            pool->probed_.fetch_add(1, std::memory_order_acq_rel);
            while (pool->probed_.load(std::memory_order_acquire) < workers)
                asm volatile("yield" ::: "memory");
            int noisiest = -1;
            uint32_t most = 0, next = 0;
            for (int tid = 1; tid < workers; ++tid) {
                const uint32_t n = pool->noise_[tid].load(std::memory_order_relaxed);
                if (n > most) { next = most; most = n; noisiest = tid; }
                else if (n > next) next = n;
            }
            if (noisiest > 0 && most >= 4 && most >= 4 * (next + 1))
                mqa_quiet_worker.store(noisiest, std::memory_order_relaxed);
            if (verbose) {
                std::fprintf(stderr, "MQA_POOL quiet worker %d (gaps %u, next %u); gaps per worker:",
                             mqa_quiet_worker.load(std::memory_order_relaxed), most, next);
                for (int tid = 0; tid < workers; ++tid)
                    std::fprintf(stderr, " %u", pool->noise_[tid].load(std::memory_order_relaxed));
                std::fprintf(stderr, "\n");
            }
        }
        if (verbose)
            std::fprintf(stderr, "MQA_POOL enabled workers=%d first_place=%d\n", workers, first_place);
        return pool;
    }

private:
    static void worker_main(MqaPool *pool, int tid, cpu_set_t set) {
        current_worker = true;
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        {
            while (!pool->probe_go_.load(std::memory_order_acquire))
                asm volatile("yield" ::: "memory");
            pool->noise_[tid].store(mqa_noise_probe(20), std::memory_order_relaxed);
            pool->probed_.fetch_add(1, std::memory_order_acq_rel);
        }
        uint64_t frequency;
        asm volatile("mrs %0, cntfrq_el0" : "=r"(frequency));
        const uint64_t idle_limit = frequency / 50;
        uint32_t seen = 0;
        for (;;) {
            uint64_t idle_begin;
            asm volatile("mrs %0, cntvct_el0" : "=r"(idle_begin));
            uint32_t generation;
            while ((generation = pool->wake_.generation.load(std::memory_order_acquire)) == seen) {
                asm volatile("yield" ::: "memory");
                uint64_t now;
                asm volatile("mrs %0, cntvct_el0" : "=r"(now));
                if (now - idle_begin > idle_limit) {
                    pool->sleepers_.value.fetch_add(1, std::memory_order_seq_cst);
                    if (pool->wake_.generation.load(std::memory_order_seq_cst) == seen)
                        syscall(SYS_futex, &pool->wake_.generation, FUTEX_WAIT_PRIVATE, seen,
                                nullptr, nullptr, 0);
                    pool->sleepers_.value.fetch_sub(1, std::memory_order_seq_cst);
                    asm volatile("mrs %0, cntvct_el0" : "=r"(idle_begin));
                }
            }
            seen = generation;
            pool->wake_.function(pool->wake_.context, tid, pool->wake_.workers);
            pool->done_[tid].value.store(generation, std::memory_order_release);

        }
    }
};

template<class Body>
static inline void mqa_parallel(Body &body, int workers, MqaPool *pool)
{
    if (pool && pool->try_acquire(workers)) {
        pool->run(body);
        pool->release();
        return;
    }
    #pragma omp parallel
    body(omp_get_thread_num(), omp_get_num_threads());
}

// ---- 布局与分块 ----
// High-weight Q/K use 4 KiB panels of interleaved (-A, B) INT8 pairs.
// Low-weight Q panels use 2048 code bytes + 64 scale bytes. K panels add
// another 64 bytes for the high-weight scale: 2176 bytes, cache-line aligned.
namespace mqa_layout {
constexpr int64_t query_group_rows(int64_t queries) { return queries <= 32 ? 16 : 32; }
constexpr int64_t dimensions = 128;
constexpr int64_t panel_rows = 16;
constexpr int64_t panel_words = 1024;
constexpr int64_t row_words = 64;
constexpr int64_t scale_word_offset = 512;
constexpr size_t panel_bytes = 4096;
constexpr int64_t k8_panel_bytes = 2176;
constexpr int64_t k8_panel_words = k8_panel_bytes / sizeof(uint32_t);
constexpr int64_t q8_panel_bytes = 2112;
constexpr int64_t q8_panel_words = q8_panel_bytes / sizeof(uint32_t);
constexpr int64_t query_row_words(int64_t heads) {
    return (heads / 32) * (panel_words + q8_panel_words);
}
constexpr size_t query_ring_slots = 4;
}

#define MQA_PF_Q_STEP \
    "prfm pldl1keep, [x17, x19]\n" \
    "add x19, x19, #64\n" \
    "and x19, x19, #%c[qmask]\n"

// ---- 输出辅助操作与 workspace ----
static constexpr size_t mqa_huge_page_bytes = 2UL << 20;

static inline void mqa_row_fill_nt(float *output, int64_t count)
{
    const svfloat32_t value = svdup_n_f32(-INFINITY);
    const int64_t step = static_cast<int64_t>(svcntw());
    // One partial line, then whole aligned lines under an all-true predicate.
    const int64_t head = std::min<int64_t>(count, ((64 - (reinterpret_cast<uintptr_t>(output) & 63)) & 63) / 4);
    if (head > 0)
        svstnt1_f32(svwhilelt_b32_s64(int64_t(0), head), output, value);
    int64_t i = head;
    const svbool_t all = svptrue_b32();
    for (; i + step <= count; i += step)
        svstnt1_f32(all, output + i, value);
    for (; i < count; i += step)
        svstnt1_f32(svwhilelt_b32_s64(i, count), output + i, value);
}

// Bounds of the private anonymous read-write mapping that contains addr.
static inline bool mqa_anonymous_mapping(uintptr_t addr, uintptr_t &begin, uintptr_t &end)
{
    FILE *maps = std::fopen("/proc/self/maps", "r");
    if (!maps) return false;
    char line[512];
    bool found = false;
    while (std::fgets(line, sizeof(line), maps)) {
        unsigned long lo = 0, hi = 0, offset = 0, inode = 0;
        char perms[8] = {}, device[32] = {};
        if (std::sscanf(line, "%lx-%lx %7s %lx %31s %lu", &lo, &hi, perms, &offset, device, &inode) != 6)
            continue;
        if (addr < lo || addr >= hi) continue;
        found = inode == 0 && perms[0] == 'r' && perms[1] == 'w' && perms[3] == 'p';
        begin = lo;
        end = hi;
        break;
    }
    std::fclose(maps);
    return found;
}

// Allocation capacity may persist between calls; packed input values never do.
// Ownership belongs to the calling thread, including all worker-private slots.
struct MqaBuffer {
    uint32_t *data = nullptr;
    size_t capacity = 0;

    MqaBuffer() = default;
    ~MqaBuffer() { std::free(data); }
    MqaBuffer(const MqaBuffer &) = delete;
    MqaBuffer &operator=(const MqaBuffer &) = delete;
    MqaBuffer(MqaBuffer &&other) noexcept
        : data(std::exchange(other.data, nullptr)),
          capacity(std::exchange(other.capacity, 0)) {}
    MqaBuffer &operator=(MqaBuffer &&other) noexcept {
        if (this != &other) {
            std::free(data);
            data = std::exchange(other.data, nullptr);
            capacity = std::exchange(other.capacity, 0);
        }
        return *this;
    }

    void reserve(size_t bytes) {
        if (bytes <= capacity) return;
        if (bytes > std::numeric_limits<size_t>::max() - (mqa_huge_page_bytes - 1))
            throw std::bad_alloc();
        const size_t rounded = (bytes + mqa_huge_page_bytes - 1) & ~(mqa_huge_page_bytes - 1);
        void *replacement = std::aligned_alloc(mqa_huge_page_bytes, rounded);
        if (!replacement) throw std::bad_alloc();
        (void)madvise(replacement, rounded, MADV_HUGEPAGE);
        std::free(data);
        data = static_cast<uint32_t *>(replacement);
        capacity = rounded;
    }
};

struct MqaBufferView {
    uint32_t *data = nullptr;
    size_t capacity = 0;
};

struct MqaWorkspace {
    MqaBuffer storage;
    MqaBufferView packed_q;
    MqaBufferView packed_kv;
    std::vector<MqaBufferView> worker_kv;
    size_t bound_q_stride = 0;
    size_t bound_stride = 0;
    size_t bound_workers = 0;
    bool in_use = false;
    MqaBuffer q_ring_storage;
    MqaBuffer reordered_scaled_weights;  // Per-row weights with Q scales folded into INT8 heads.
    size_t q_ring_slot = 0;
    // Last populated mapping and the shape that used it (no input/result cache).
    uintptr_t populated_begin = 0, populated_end = 0;
    int64_t populated_shape[3] = {0, 0, 0};
    void prepare(size_t q_bytes, size_t kv_bytes, size_t workers) {
        constexpr size_t alignment = 64;
        const size_t max_bytes = std::numeric_limits<size_t>::max();
        if (q_bytes > max_bytes - (alignment - 1) ||
            kv_bytes > max_bytes - (alignment - 1)) throw std::bad_alloc();
        const size_t q_stride = (q_bytes + alignment - 1) & ~(alignment - 1);
        const size_t stride = (kv_bytes + alignment - 1) & ~(alignment - 1);
        if (stride && workers > max_bytes / stride) throw std::bad_alloc();
        const size_t kv_total = workers * stride;
        if (q_stride > max_bytes - kv_total) throw std::bad_alloc();
        if (q_stride == bound_q_stride && stride == bound_stride &&
            workers == bound_workers) return;
        if (worker_kv.size() < workers) worker_kv.resize(workers);
        // One owner covers Q followed by disjoint KV slots. Nothing updates
        // the binding views until allocation has succeeded.
        storage.reserve(q_stride + kv_total);
        packed_q = {storage.data, q_stride};
        packed_kv = {storage.data ? storage.data + q_stride / sizeof(uint32_t)
                                  : nullptr, kv_total};
        for (size_t worker = 0; worker < workers; ++worker)
            worker_kv[worker] = {
                packed_kv.data ? packed_kv.data + worker * (stride / sizeof(uint32_t))
                               : nullptr, stride};
        for (size_t worker = workers; worker < worker_kv.size(); ++worker)
            worker_kv[worker] = {};
        bound_q_stride = q_stride;
        bound_stride = stride;
        bound_workers = workers;
    }
};

class MqaWorkspaceLease {
    MqaWorkspace nested_workspace_;
    MqaWorkspace &workspace_;
public:
    explicit MqaWorkspaceLease(MqaWorkspace &cached)
        : workspace_(cached.in_use ? nested_workspace_ : cached) {
        workspace_.in_use = true;
    }
    ~MqaWorkspaceLease() { workspace_.in_use = false; }
    MqaWorkspaceLease(const MqaWorkspaceLease &) = delete;
    MqaWorkspaceLease &operator=(const MqaWorkspaceLease &) = delete;
    MqaWorkspace &get() { return workspace_; }
};

// ---- BF16 / INT8 量化与打包 ----
#define MQA_Q8_QUANT2 \
    "movprfx z4, z30\n" "bfmlalb z4.s, z2.h, z29.h\n" \
    "movprfx z12, z30\n" "bfmlalb z12.s, z10.h, z29.h\n" \
    "movprfx z5, z30\n" "bfmlalt z5.s, z2.h, z29.h\n" \
    "movprfx z13, z30\n" "bfmlalt z13.s, z10.h, z29.h\n" \
    "movprfx z6, z30\n" "bfmlalb z6.s, z3.h, z29.h\n" \
    "movprfx z14, z30\n" "bfmlalb z14.s, z11.h, z29.h\n" \
    "movprfx z7, z30\n" "bfmlalt z7.s, z3.h, z29.h\n" \
    "movprfx z15, z30\n" "bfmlalt z15.s, z11.h, z29.h\n" \
    "trn1 z4.b, z4.b, z5.b\n" \
    "trn1 z12.b, z12.b, z13.b\n" \
    "trn1 z6.b, z6.b, z7.b\n" \
    "trn1 z14.b, z14.b, z15.b\n" \
    "trn1 z4.h, z4.h, z6.h\n" \
    "trn1 z12.h, z12.h, z14.h\n"
#define MQA_Q8_MOVA4(T) \
    "mova z2.s, p0/m, za" #T "v.s[w12, 0]\n" \
    "mova z3.s, p0/m, za" #T "v.s[w12, 1]\n" \
    "mova z10.s, p0/m, za" #T "v.s[w12, 2]\n" \
    "mova z11.s, p0/m, za" #T "v.s[w12, 3]\n"
// Per-token absmax of the packer. The loop's four token vectors
// (z0, z4, z8, z12) are halved twice into one vector of four tokens x 8 lanes,
// kept in z16-z19; after the loop three more halvings leave token t in lane t.
#define LAT_ABSMAX_ITER \
    "uzp1 z1.h, z0.h, z4.h\n" "uzp2 z2.h, z0.h, z4.h\n" "umax z1.h, p0/m, z1.h, z2.h\n" \
    "uzp1 z5.h, z8.h, z12.h\n" "uzp2 z6.h, z8.h, z12.h\n" "umax z5.h, p0/m, z5.h, z6.h\n" \
    "uzp1 z0.h, z1.h, z5.h\n" "uzp2 z2.h, z1.h, z5.h\n" "umax z0.h, p0/m, z0.h, z2.h\n" \
    "mov z19.d, z18.d\n" "mov z18.d, z17.d\n" "mov z17.d, z16.d\n" "mov z16.d, z0.d\n"
#define LAT_ABSMAX_FINAL \
    "uzp1 z0.h, z19.h, z18.h\n" "uzp2 z1.h, z19.h, z18.h\n" "umax z0.h, p0/m, z0.h, z1.h\n" \
    "uzp1 z2.h, z17.h, z16.h\n" "uzp2 z3.h, z17.h, z16.h\n" "umax z2.h, p0/m, z2.h, z3.h\n" \
    "uzp1 z4.h, z0.h, z2.h\n" "uzp2 z5.h, z0.h, z2.h\n" "umax z4.h, p0/m, z4.h, z5.h\n" \
    "uzp1 z31.h, z4.h, z4.h\n" "uzp2 z5.h, z4.h, z4.h\n" "umax z31.h, p0/m, z31.h, z5.h\n"
#define LAT_ABSMAX_REV
#define LAT_KPACK_PFS_STEP(T) "add x9, x10, x12, lsl #6\n" \
    "prfm pldl1keep, [x9, #" #T "*1024]\n" "prfm pldl1keep, [x9, #" #T "*1024+64]\n" \
    "prfm pldl1keep, [x9, #" #T "*1024+128]\n" "prfm pldl1keep, [x9, #" #T "*1024+192]\n"
// Equal-weight lattice: T=RNE(x*inv), C=RNE(9*T/41), A=5*C-T, B=T-4*C.
// Store (-A,B); 41*(Aq*Ak+Bq*Bk) = Tq*Tk+Rq*Rk, R=5*A-4*B.
// inv=BF16_truncate(1025/absmax); |A|<=102, |B|<=126, |R|<=20.
// SQRDMULH with 7193 implements RNE(9*T/41) over this finite domain.
// Four independent encode chains share one ZA transpose / absmax scan.
// Low=true also emits the single-INT8 K codes and both K scales. Otherwise
// high Q scales are returned separately and folded into the head weights.
template <bool Low, bool FoldWeights = false>
static inline void mqa_lattice_pack(const bfloat16_t *source, const bfloat16_t *const *rows,
                                    uint32_t *high, uint32_t *low, float *scales, int64_t groups,
                                    const float *weights = nullptr, const int *order = nullptr,
                                    const void *next_rows = nullptr)
{
    // next_rows (rows path, 1): 4 KiB a group prefetched from it.
    const void *args[] = {source, rows, high, low, scales, weights, order, next_rows};
    asm volatile(
        "smstart\n"
        "ptrue p0.b\n"
        "ldp x14, x16, [%[args]]\n"
        "ldp x13, x15, [%[args], #16]\n"
        "ldr x4, [%[args], #32]\n"
        "mov x8, %[groups]\n"
        "mov x5, #16\n"
        "mov x7, #32\n"
        "mov x17, #48\n"
        "ldr x19, [%[args], #56]\n"
        "cbz x8, 9f\n"
        "1:\n"
        "mov z31.h, #0\n"
        "mov w12, #0\n"
        "2:\n"
        "cbz x16, 3f\n"
        "ldr x10, [x16], #8\n"
        "b 4f\n"
        "3:\n"
        "mov x10, x14\n"
        "add x14, x14, #256\n"
        "4:\n"
        "ld1w {z0.s}, p0/z, [x10]\n"
        "ld1w {z1.s}, p0/z, [x10, x5, lsl #2]\n"
        "ld1w {z2.s}, p0/z, [x10, x7, lsl #2]\n"
        "ld1w {z3.s}, p0/z, [x10, x17, lsl #2]\n"
        "mova za0h.s[w12, 0], p0/m, z0.s\n"
        "mova za1h.s[w12, 0], p0/m, z1.s\n"
        "mova za2h.s[w12, 0], p0/m, z2.s\n"
        "mova za3h.s[w12, 0], p0/m, z3.s\n"
        "add w12, w12, #1\n"
        "cbz x16, 3f\n"
        "ldr x10, [x16], #8\n"
        "b 4f\n"
        "3:\n"
        "mov x10, x14\n"
        "add x14, x14, #256\n"
        "4:\n"
        "ld1w {z4.s}, p0/z, [x10]\n"
        "ld1w {z5.s}, p0/z, [x10, x5, lsl #2]\n"
        "ld1w {z6.s}, p0/z, [x10, x7, lsl #2]\n"
        "ld1w {z7.s}, p0/z, [x10, x17, lsl #2]\n"
        "mova za0h.s[w12, 0], p0/m, z4.s\n"
        "mova za1h.s[w12, 0], p0/m, z5.s\n"
        "mova za2h.s[w12, 0], p0/m, z6.s\n"
        "mova za3h.s[w12, 0], p0/m, z7.s\n"
        "add w12, w12, #1\n"
        "cbz x16, 3f\n"
        "ldr x10, [x16], #8\n"
        "b 4f\n"
        "3:\n"
        "mov x10, x14\n"
        "add x14, x14, #256\n"
        "4:\n"
        "ld1w {z8.s}, p0/z, [x10]\n"
        "ld1w {z9.s}, p0/z, [x10, x5, lsl #2]\n"
        "ld1w {z10.s}, p0/z, [x10, x7, lsl #2]\n"
        "ld1w {z11.s}, p0/z, [x10, x17, lsl #2]\n"
        "mova za0h.s[w12, 0], p0/m, z8.s\n"
        "mova za1h.s[w12, 0], p0/m, z9.s\n"
        "mova za2h.s[w12, 0], p0/m, z10.s\n"
        "mova za3h.s[w12, 0], p0/m, z11.s\n"
        "add w12, w12, #1\n"
        "cbz x16, 3f\n"
        "ldr x10, [x16], #8\n"
        "b 4f\n"
        "3:\n"
        "mov x10, x14\n"
        "add x14, x14, #256\n"
        "4:\n"
        "ld1w {z12.s}, p0/z, [x10]\n"
        "ld1w {z13.s}, p0/z, [x10, x5, lsl #2]\n"
        "ld1w {z14.s}, p0/z, [x10, x7, lsl #2]\n"
        "ld1w {z15.s}, p0/z, [x10, x17, lsl #2]\n"
        "mova za0h.s[w12, 0], p0/m, z12.s\n"
        "mova za1h.s[w12, 0], p0/m, z13.s\n"
        "mova za2h.s[w12, 0], p0/m, z14.s\n"
        "mova za3h.s[w12, 0], p0/m, z15.s\n"
        "add w12, w12, #1\n"
        "fabs z0.h, p0/m, z0.h\n"
        "fabs z1.h, p0/m, z1.h\n"
        "fabs z2.h, p0/m, z2.h\n"
        "fabs z3.h, p0/m, z3.h\n"
        "fabs z4.h, p0/m, z4.h\n"
        "fabs z5.h, p0/m, z5.h\n"
        "fabs z6.h, p0/m, z6.h\n"
        "fabs z7.h, p0/m, z7.h\n"
        "fabs z8.h, p0/m, z8.h\n"
        "fabs z9.h, p0/m, z9.h\n"
        "fabs z10.h, p0/m, z10.h\n"
        "fabs z11.h, p0/m, z11.h\n"
        "fabs z12.h, p0/m, z12.h\n"
        "fabs z13.h, p0/m, z13.h\n"
        "fabs z14.h, p0/m, z14.h\n"
        "fabs z15.h, p0/m, z15.h\n"
        "umax z0.h, p0/m, z0.h, z1.h\n"
        "umax z2.h, p0/m, z2.h, z3.h\n"
        "umax z4.h, p0/m, z4.h, z5.h\n"
        "umax z6.h, p0/m, z6.h, z7.h\n"
        "umax z8.h, p0/m, z8.h, z9.h\n"
        "umax z10.h, p0/m, z10.h, z11.h\n"
        "umax z12.h, p0/m, z12.h, z13.h\n"
        "umax z14.h, p0/m, z14.h, z15.h\n"
        "umax z0.h, p0/m, z0.h, z2.h\n"
        "umax z4.h, p0/m, z4.h, z6.h\n"
        "umax z8.h, p0/m, z8.h, z10.h\n"
        "umax z12.h, p0/m, z12.h, z14.h\n"
        LAT_ABSMAX_ITER
        "cmp w12, #16\n"
        "b.ne 2b\n"
        LAT_ABSMAX_FINAL
        // the next group's raw K, 4 lines each quantization iteration (x10;
        // the hot output when there is none: the Q path or the last group)
        "mov x10, x13\n"
        "cbnz x16, 24f\n"
        "cmp x8, #1\n"
        "b.eq 23f\n"
        "mov x10, x14\n"
        "b 23f\n"
        "24:\n"
        "cbz x19, 23f\n"
        "mov x10, x19\n"
        "add x19, x19, #4096\n"
        "23:\n"
        "uunpklo z31.s, z31.h\n"
        LAT_ABSMAX_REV
        "lsl z31.s, z31.s, #16\n"
        "fcmeq p1.s, p0/z, z31.s, #0.0\n"
        "movz w9, #0x4480, lsl #16\n"
        "movk w9, #0x2000\n"
        "dup z21.s, w9\n"
        "fdiv z21.s, p0/m, z21.s, z31.s\n"
        "and z21.s, z21.s, #0xffff0000\n"
        "mov z21.s, p1/m, #0\n"
        "fmov z22.s, #1.0\n"
        "fdiv z22.s, p0/m, z22.s, z21.s\n"
        "mov z22.s, p1/m, #0\n"
        "lsr z1.s, z21.s, #16\n"
        "orr z21.d, z21.d, z1.d\n"
        ".if %c[low]\n"
        "movz w9, #0x42fe, lsl #16\n"
        "dup z29.s, w9\n"
        "fdiv z29.s, p0/m, z29.s, z31.s\n"
        "and z29.s, z29.s, #0xffff0000\n"
        "mov z29.s, p1/m, #0\n"
        "fmov z24.s, #1.0\n"
        "fdiv z24.s, p0/m, z24.s, z29.s\n"
        "mov z24.s, p1/m, #0\n"
        "lsr z1.s, z29.s, #16\n"
        "orr z29.d, z29.d, z1.d\n"
        ".endif\n"
        "movz w9, #0x4b40, lsl #16\n"
        "dup z30.s, w9\n"
        "mov w9, #7193\n"
        "dup z27.h, w9\n"
        "mov z28.h, #5\n"
        "mov x6, #0\n"
        "mov x11, x15\n"
        "mov w12, #0\n"
        "60:\n"
        MQA_Q8_MOVA4(0)
        LAT_KPACK_PFS_STEP(0)
        "movprfx z4, z30\n"
        "bfmlalb z4.s, z2.h, z21.h\n"
        "movprfx z5, z30\n"
        "bfmlalt z5.s, z2.h, z21.h\n"
        "movprfx z6, z30\n"
        "bfmlalb z6.s, z3.h, z21.h\n"
        "movprfx z7, z30\n"
        "bfmlalt z7.s, z3.h, z21.h\n"
        "movprfx z12, z30\n"
        "bfmlalb z12.s, z10.h, z21.h\n"
        "movprfx z13, z30\n"
        "bfmlalt z13.s, z10.h, z21.h\n"
        "movprfx z14, z30\n"
        "bfmlalb z14.s, z11.h, z21.h\n"
        "movprfx z15, z30\n"
        "bfmlalt z15.s, z11.h, z21.h\n"
        "trn1 z4.h, z4.h, z5.h\n"
        "trn1 z6.h, z6.h, z7.h\n"
        "trn1 z12.h, z12.h, z13.h\n"
        "trn1 z14.h, z14.h, z15.h\n"
        "sqrdmulh z5.h, z4.h, z27.h\n"
        "sqrdmulh z7.h, z6.h, z27.h\n"
        "sqrdmulh z13.h, z12.h, z27.h\n"
        "sqrdmulh z15.h, z14.h, z27.h\n"
        "mls z4.h, p0/m, z5.h, z28.h\n"
        "mls z6.h, p0/m, z7.h, z28.h\n"
        "mls z12.h, p0/m, z13.h, z28.h\n"
        "mls z14.h, p0/m, z15.h, z28.h\n"
        "add z5.h, z4.h, z5.h\n"
        "add z7.h, z6.h, z7.h\n"
        "add z13.h, z12.h, z13.h\n"
        "add z15.h, z14.h, z15.h\n"
        "trn1 z4.b, z4.b, z5.b\n"
        "trn1 z6.b, z6.b, z7.b\n"
        "trn1 z12.b, z12.b, z13.b\n"
        "trn1 z14.b, z14.b, z15.b\n"
        "st1w {z4.s}, p0, [x13, x6, lsl #2]\n"
        "add x6, x6, #16\n"
        "st1w {z6.s}, p0, [x13, x6, lsl #2]\n"
        "add x6, x6, #16\n"
        "st1w {z12.s}, p0, [x13, x6, lsl #2]\n"
        "add x6, x6, #16\n"
        "st1w {z14.s}, p0, [x13, x6, lsl #2]\n"
        "add x6, x6, #16\n"
        ".if %c[low]\n"
        MQA_Q8_QUANT2
        "st1w {z4.s}, p0, [x11]\n"
        "st1w {z12.s}, p0, [x11, #1, mul vl]\n"
        "add x11, x11, #128\n"
        ".endif\n"
        "add w12, w12, #4\n"
        "cmp w12, #16\n"
        "b.ne 60b\n"
        "mov w12, #0\n"
        "61:\n"
        MQA_Q8_MOVA4(1)
        LAT_KPACK_PFS_STEP(1)
        "movprfx z4, z30\n"
        "bfmlalb z4.s, z2.h, z21.h\n"
        "movprfx z5, z30\n"
        "bfmlalt z5.s, z2.h, z21.h\n"
        "movprfx z6, z30\n"
        "bfmlalb z6.s, z3.h, z21.h\n"
        "movprfx z7, z30\n"
        "bfmlalt z7.s, z3.h, z21.h\n"
        "movprfx z12, z30\n"
        "bfmlalb z12.s, z10.h, z21.h\n"
        "movprfx z13, z30\n"
        "bfmlalt z13.s, z10.h, z21.h\n"
        "movprfx z14, z30\n"
        "bfmlalb z14.s, z11.h, z21.h\n"
        "movprfx z15, z30\n"
        "bfmlalt z15.s, z11.h, z21.h\n"
        "trn1 z4.h, z4.h, z5.h\n"
        "trn1 z6.h, z6.h, z7.h\n"
        "trn1 z12.h, z12.h, z13.h\n"
        "trn1 z14.h, z14.h, z15.h\n"
        "sqrdmulh z5.h, z4.h, z27.h\n"
        "sqrdmulh z7.h, z6.h, z27.h\n"
        "sqrdmulh z13.h, z12.h, z27.h\n"
        "sqrdmulh z15.h, z14.h, z27.h\n"
        "mls z4.h, p0/m, z5.h, z28.h\n"
        "mls z6.h, p0/m, z7.h, z28.h\n"
        "mls z12.h, p0/m, z13.h, z28.h\n"
        "mls z14.h, p0/m, z15.h, z28.h\n"
        "add z5.h, z4.h, z5.h\n"
        "add z7.h, z6.h, z7.h\n"
        "add z13.h, z12.h, z13.h\n"
        "add z15.h, z14.h, z15.h\n"
        "trn1 z4.b, z4.b, z5.b\n"
        "trn1 z6.b, z6.b, z7.b\n"
        "trn1 z12.b, z12.b, z13.b\n"
        "trn1 z14.b, z14.b, z15.b\n"
        "st1w {z4.s}, p0, [x13, x6, lsl #2]\n"
        "add x6, x6, #16\n"
        "st1w {z6.s}, p0, [x13, x6, lsl #2]\n"
        "add x6, x6, #16\n"
        "st1w {z12.s}, p0, [x13, x6, lsl #2]\n"
        "add x6, x6, #16\n"
        "st1w {z14.s}, p0, [x13, x6, lsl #2]\n"
        "add x6, x6, #16\n"
        ".if %c[low]\n"
        MQA_Q8_QUANT2
        "st1w {z4.s}, p0, [x11]\n"
        "st1w {z12.s}, p0, [x11, #1, mul vl]\n"
        "add x11, x11, #128\n"
        ".endif\n"
        "add w12, w12, #4\n"
        "cmp w12, #16\n"
        "b.ne 61b\n"
        "mov w12, #0\n"
        "62:\n"
        MQA_Q8_MOVA4(2)
        LAT_KPACK_PFS_STEP(2)
        "movprfx z4, z30\n"
        "bfmlalb z4.s, z2.h, z21.h\n"
        "movprfx z5, z30\n"
        "bfmlalt z5.s, z2.h, z21.h\n"
        "movprfx z6, z30\n"
        "bfmlalb z6.s, z3.h, z21.h\n"
        "movprfx z7, z30\n"
        "bfmlalt z7.s, z3.h, z21.h\n"
        "movprfx z12, z30\n"
        "bfmlalb z12.s, z10.h, z21.h\n"
        "movprfx z13, z30\n"
        "bfmlalt z13.s, z10.h, z21.h\n"
        "movprfx z14, z30\n"
        "bfmlalb z14.s, z11.h, z21.h\n"
        "movprfx z15, z30\n"
        "bfmlalt z15.s, z11.h, z21.h\n"
        "trn1 z4.h, z4.h, z5.h\n"
        "trn1 z6.h, z6.h, z7.h\n"
        "trn1 z12.h, z12.h, z13.h\n"
        "trn1 z14.h, z14.h, z15.h\n"
        "sqrdmulh z5.h, z4.h, z27.h\n"
        "sqrdmulh z7.h, z6.h, z27.h\n"
        "sqrdmulh z13.h, z12.h, z27.h\n"
        "sqrdmulh z15.h, z14.h, z27.h\n"
        "mls z4.h, p0/m, z5.h, z28.h\n"
        "mls z6.h, p0/m, z7.h, z28.h\n"
        "mls z12.h, p0/m, z13.h, z28.h\n"
        "mls z14.h, p0/m, z15.h, z28.h\n"
        "add z5.h, z4.h, z5.h\n"
        "add z7.h, z6.h, z7.h\n"
        "add z13.h, z12.h, z13.h\n"
        "add z15.h, z14.h, z15.h\n"
        "trn1 z4.b, z4.b, z5.b\n"
        "trn1 z6.b, z6.b, z7.b\n"
        "trn1 z12.b, z12.b, z13.b\n"
        "trn1 z14.b, z14.b, z15.b\n"
        "st1w {z4.s}, p0, [x13, x6, lsl #2]\n"
        "add x6, x6, #16\n"
        "st1w {z6.s}, p0, [x13, x6, lsl #2]\n"
        "add x6, x6, #16\n"
        "st1w {z12.s}, p0, [x13, x6, lsl #2]\n"
        "add x6, x6, #16\n"
        "st1w {z14.s}, p0, [x13, x6, lsl #2]\n"
        "add x6, x6, #16\n"
        ".if %c[low]\n"
        MQA_Q8_QUANT2
        "st1w {z4.s}, p0, [x11]\n"
        "st1w {z12.s}, p0, [x11, #1, mul vl]\n"
        "add x11, x11, #128\n"
        ".endif\n"
        "add w12, w12, #4\n"
        "cmp w12, #16\n"
        "b.ne 62b\n"
        "mov w12, #0\n"
        "63:\n"
        MQA_Q8_MOVA4(3)
        LAT_KPACK_PFS_STEP(3)
        "movprfx z4, z30\n"
        "bfmlalb z4.s, z2.h, z21.h\n"
        "movprfx z5, z30\n"
        "bfmlalt z5.s, z2.h, z21.h\n"
        "movprfx z6, z30\n"
        "bfmlalb z6.s, z3.h, z21.h\n"
        "movprfx z7, z30\n"
        "bfmlalt z7.s, z3.h, z21.h\n"
        "movprfx z12, z30\n"
        "bfmlalb z12.s, z10.h, z21.h\n"
        "movprfx z13, z30\n"
        "bfmlalt z13.s, z10.h, z21.h\n"
        "movprfx z14, z30\n"
        "bfmlalb z14.s, z11.h, z21.h\n"
        "movprfx z15, z30\n"
        "bfmlalt z15.s, z11.h, z21.h\n"
        "trn1 z4.h, z4.h, z5.h\n"
        "trn1 z6.h, z6.h, z7.h\n"
        "trn1 z12.h, z12.h, z13.h\n"
        "trn1 z14.h, z14.h, z15.h\n"
        "sqrdmulh z5.h, z4.h, z27.h\n"
        "sqrdmulh z7.h, z6.h, z27.h\n"
        "sqrdmulh z13.h, z12.h, z27.h\n"
        "sqrdmulh z15.h, z14.h, z27.h\n"
        "mls z4.h, p0/m, z5.h, z28.h\n"
        "mls z6.h, p0/m, z7.h, z28.h\n"
        "mls z12.h, p0/m, z13.h, z28.h\n"
        "mls z14.h, p0/m, z15.h, z28.h\n"
        "add z5.h, z4.h, z5.h\n"
        "add z7.h, z6.h, z7.h\n"
        "add z13.h, z12.h, z13.h\n"
        "add z15.h, z14.h, z15.h\n"
        "trn1 z4.b, z4.b, z5.b\n"
        "trn1 z6.b, z6.b, z7.b\n"
        "trn1 z12.b, z12.b, z13.b\n"
        "trn1 z14.b, z14.b, z15.b\n"
        "st1w {z4.s}, p0, [x13, x6, lsl #2]\n"
        "add x6, x6, #16\n"
        "st1w {z6.s}, p0, [x13, x6, lsl #2]\n"
        "add x6, x6, #16\n"
        "st1w {z12.s}, p0, [x13, x6, lsl #2]\n"
        "add x6, x6, #16\n"
        "st1w {z14.s}, p0, [x13, x6, lsl #2]\n"
        "add x6, x6, #16\n"
        ".if %c[low]\n"
        MQA_Q8_QUANT2
        "st1w {z4.s}, p0, [x11]\n"
        "st1w {z12.s}, p0, [x11, #1, mul vl]\n"
        "add x11, x11, #128\n"
        ".endif\n"
        "add w12, w12, #4\n"
        "cmp w12, #16\n"
        "b.ne 63b\n"
        ".if %c[low]\n"
        "add x11, x15, #2048\n"
        "st1w {z24.s}, p0, [x11]\n"
        "st1w {z22.s}, p0, [x11, #1, mul vl]\n"
        "add x15, x15, #2176\n"
        ".else\n"
        ".if %c[fold]\n"
        "ldp x10, x11, [%[args], #40]\n"
        "ldr x9, [%[args], #32]\n"
        "sub x9, x4, x9\n"
        "add x11, x11, x9\n"
        "ld1w {z0.s}, p0/z, [x11]\n"
        "ld1w {z1.s}, p0/z, [x10, z0.s, uxtw #2]\n"
        "movz w9, #0x4224, lsl #16\n"
        "dup z0.s, w9\n"
        "fmul z22.s, z22.s, z0.s\n"
        "fmul z22.s, z22.s, z1.s\n"
        ".endif\n"
        "st1w {z22.s}, p0, [x4]\n"
        "add x4, x4, #64\n"
        ".endif\n"
        "add x13, x13, #4096\n"
        "subs x8, x8, #1\n"
        "b.ne 1b\n"
        "9:\n"
        "smstop\n"
        :
        : [args] "r"(args), [groups] "r"(groups), [low] "i"(Low ? 1 : 0), [fold] "i"(FoldWeights ? 1 : 0)
        : "memory", "cc", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14",
          "x15", "x16", "x17", "x19", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10",
          "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p1", "p2", "p3", "p4",
          "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15");
}

// Pack low Q and fold its scale into the reordered weights while in streaming mode.
// H64's four-pair correction follows after this returns and retains the same codes.
#define LAT_QLOW_PF_STEP "cbz x7, 97f\n" "prfm pldl1keep, [x7]\n" "prfm pldl1keep, [x7, #64]\n" \
    "prfm pldl1keep, [x7, #128]\n" "prfm pldl1keep, [x7, #192]\n" "add x7, x7, #256\n" "97:\n"
static inline void mqa_pack_low_query(const bfloat16_t *const *rows, const int *indices, const float *weights,
                                      uint32_t *out, float *alpha, int64_t groups,
                                      const void *next_rows = nullptr)
{
    // next_rows (1): 4 KiB a group prefetched from it, 4 lines
    // each quantization iteration (x7; none when it is null: an address next to
    // the row pointers can run past the stack into unmapped pages).
    const void *args[] = {rows, indices, weights, out, alpha, next_rows};
    asm volatile("smstart\n"
                 "ptrue p0.b\n"
                 "ldp x16, x15, [%[args]]\n"
                 "ldp x14, x13, [%[args], #16]\n"
                 "ldr x4, [%[args], #32]\n"
                 "ldr x7, [%[args], #40]\n"
                 "mov x8, %[groups]\n"
                 "movz w9, #0x4b40, lsl #16\n"
                 "dup z30.s, w9\n"
                 "cbz x8, 99f\n"
                 "1:\n"
                 "mov z31.h, #0\n"
                 "mov w12, #0\n"
                 "2:\n"
                 "ldr x10, [x16], #8\n"
                 "ld1w {z0.s}, p0/z, [x10, #0, mul vl]\n"
                 "ld1w {z1.s}, p0/z, [x10, #1, mul vl]\n"
                 "ld1w {z2.s}, p0/z, [x10, #2, mul vl]\n"
                 "ld1w {z3.s}, p0/z, [x10, #3, mul vl]\n"
                 "mova za0h.s[w12, 0], p0/m, z0.s\n"
                 "mova za1h.s[w12, 0], p0/m, z1.s\n"
                 "mova za2h.s[w12, 0], p0/m, z2.s\n"
                 "mova za3h.s[w12, 0], p0/m, z3.s\n"
                 "add w12, w12, #1\n"
                 "ldr x10, [x16], #8\n"
                 "ld1w {z4.s}, p0/z, [x10, #0, mul vl]\n"
                 "ld1w {z5.s}, p0/z, [x10, #1, mul vl]\n"
                 "ld1w {z6.s}, p0/z, [x10, #2, mul vl]\n"
                 "ld1w {z7.s}, p0/z, [x10, #3, mul vl]\n"
                 "mova za0h.s[w12, 0], p0/m, z4.s\n"
                 "mova za1h.s[w12, 0], p0/m, z5.s\n"
                 "mova za2h.s[w12, 0], p0/m, z6.s\n"
                 "mova za3h.s[w12, 0], p0/m, z7.s\n"
                 "add w12, w12, #1\n"
                 "ldr x10, [x16], #8\n"
                 "ld1w {z8.s}, p0/z, [x10, #0, mul vl]\n"
                 "ld1w {z9.s}, p0/z, [x10, #1, mul vl]\n"
                 "ld1w {z10.s}, p0/z, [x10, #2, mul vl]\n"
                 "ld1w {z11.s}, p0/z, [x10, #3, mul vl]\n"
                 "mova za0h.s[w12, 0], p0/m, z8.s\n"
                 "mova za1h.s[w12, 0], p0/m, z9.s\n"
                 "mova za2h.s[w12, 0], p0/m, z10.s\n"
                 "mova za3h.s[w12, 0], p0/m, z11.s\n"
                 "add w12, w12, #1\n"
                 "ldr x10, [x16], #8\n"
                 "ld1w {z12.s}, p0/z, [x10, #0, mul vl]\n"
                 "ld1w {z13.s}, p0/z, [x10, #1, mul vl]\n"
                 "ld1w {z14.s}, p0/z, [x10, #2, mul vl]\n"
                 "ld1w {z15.s}, p0/z, [x10, #3, mul vl]\n"
                 "mova za0h.s[w12, 0], p0/m, z12.s\n"
                 "mova za1h.s[w12, 0], p0/m, z13.s\n"
                 "mova za2h.s[w12, 0], p0/m, z14.s\n"
                 "mova za3h.s[w12, 0], p0/m, z15.s\n"
                 "add w12, w12, #1\n"
                 "fabs z0.h, p0/m, z0.h\n"
                 "fabs z1.h, p0/m, z1.h\n"
                 "fabs z2.h, p0/m, z2.h\n"
                 "fabs z3.h, p0/m, z3.h\n"
                 "fabs z4.h, p0/m, z4.h\n"
                 "fabs z5.h, p0/m, z5.h\n"
                 "fabs z6.h, p0/m, z6.h\n"
                 "fabs z7.h, p0/m, z7.h\n"
                 "fabs z8.h, p0/m, z8.h\n"
                 "fabs z9.h, p0/m, z9.h\n"
                 "fabs z10.h, p0/m, z10.h\n"
                 "fabs z11.h, p0/m, z11.h\n"
                 "fabs z12.h, p0/m, z12.h\n"
                 "fabs z13.h, p0/m, z13.h\n"
                 "fabs z14.h, p0/m, z14.h\n"
                 "fabs z15.h, p0/m, z15.h\n"
                 "umax z0.h, p0/m, z0.h, z1.h\n"
                 "umax z2.h, p0/m, z2.h, z3.h\n"
                 "umax z4.h, p0/m, z4.h, z5.h\n"
                 "umax z6.h, p0/m, z6.h, z7.h\n"
                 "umax z8.h, p0/m, z8.h, z9.h\n"
                 "umax z10.h, p0/m, z10.h, z11.h\n"
                 "umax z12.h, p0/m, z12.h, z13.h\n"
                 "umax z14.h, p0/m, z14.h, z15.h\n"
                 "umax z0.h, p0/m, z0.h, z2.h\n"
                 "umax z4.h, p0/m, z4.h, z6.h\n"
                 "umax z8.h, p0/m, z8.h, z10.h\n"
                 "umax z12.h, p0/m, z12.h, z14.h\n"
                 LAT_ABSMAX_ITER
                 "cmp w12, #16\n"
                 "b.ne 2b\n"
                 LAT_ABSMAX_FINAL
                 "uunpklo z31.s, z31.h\n"
                 LAT_ABSMAX_REV
                 "lsl z31.s, z31.s, #16\n"
                 "fcmeq p1.s, p0/z, z31.s, #0.0\n"
                 "movz w9, #0x42fe, lsl #16\n"
                 "dup z29.s, w9\n"
                 "fdiv z29.s, p0/m, z29.s, z31.s\n"
                 "and z29.s, z29.s, #0xffff0000\n"
                 "mov z29.s, p1/m, #0\n"
                 "fmov z24.s, #1.0\n"
                 "fdiv z24.s, p0/m, z24.s, z29.s\n"
                 "mov z24.s, p1/m, #0\n"
                 "lsr z1.s, z29.s, #16\n"
                 "orr z29.d, z29.d, z1.d\n"
                 "ld1w {z0.s}, p0/z, [x15]\n"
                 "ld1w {z26.s}, p0/z, [x14, z0.s, uxtw #2]\n"
                 "fmul z26.s, z26.s, z24.s\n"
                 "st1w {z26.s}, p0, [x4]\n"
                 "mov x11, x13\n"
                 "mov w12, #0\n"
                 "60:\n" MQA_Q8_MOVA4(0) LAT_QLOW_PF_STEP MQA_Q8_QUANT2 "st1w {z4.s}, p0, [x11]\n"
                                                       "st1w {z12.s}, p0, [x11, #1, mul vl]\n"
                                                       "add x11, x11, #128\n"
                                                       "add w12, w12, #4\n"
                                                       "cmp w12, #16\n"
                                                       "b.ne 60b\n"
                                                       "mov w12, #0\n"
                                                       "61:\n" MQA_Q8_MOVA4(1) LAT_QLOW_PF_STEP MQA_Q8_QUANT2
                 "st1w {z4.s}, p0, [x11]\n"
                 "st1w {z12.s}, p0, [x11, #1, mul vl]\n"
                 "add x11, x11, #128\n"
                 "add w12, w12, #4\n"
                 "cmp w12, #16\n"
                 "b.ne 61b\n"
                 "mov w12, #0\n"
                 "62:\n" MQA_Q8_MOVA4(2) LAT_QLOW_PF_STEP MQA_Q8_QUANT2 "st1w {z4.s}, p0, [x11]\n"
                                                       "st1w {z12.s}, p0, [x11, #1, mul vl]\n"
                                                       "add x11, x11, #128\n"
                                                       "add w12, w12, #4\n"
                                                       "cmp w12, #16\n"
                                                       "b.ne 62b\n"
                                                       "mov w12, #0\n"
                                                       "63:\n" MQA_Q8_MOVA4(3) LAT_QLOW_PF_STEP MQA_Q8_QUANT2
                 "st1w {z4.s}, p0, [x11]\n"
                 "st1w {z12.s}, p0, [x11, #1, mul vl]\n"
                 "add x11, x11, #128\n"
                 "add w12, w12, #4\n"
                 "cmp w12, #16\n"
                 "b.ne 63b\n"
                 "add x11, x13, #2048\n"
                 "st1w {z24.s}, p0, [x11]\n"
                 "add x13, x13, #2112\n"
                 "add x15, x15, #64\n"
                 "add x4, x4, #64\n"
                 "subs x8, x8, #1\n"
                 "b.ne 1b\n"
                 "99:\n"
                 "smstop\n"
                 :
                 : [args] "r"(args), [groups] "r"(groups)
                 : "memory", "cc", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14",
                   "x15", "x16", "x17", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10",
                   "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
                   "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p1", "p2", "p3", "p4", "p5",
                   "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15");
}

// Compensate low-branch Q rounding error within one 16-head panel. Pick the
// 2*Pairs largest |alpha| (ties: lower index), pairing a target with a donor.
// alpha = weight * Q scale; gain = 0.5 * alpha_donor / alpha_target. Re-encode
// the target as RNE(u_target - gain * (RNE(u_donor) - u_donor)); the donor
// stays unchanged. This input-only correction adds no dot products.
// Case 1 uses four pairs; the non-streaming SVE layout requires 512 bits.

// Selection keys are unique: abs(weight) bits followed by the original head ID.
// Keep the same low/high head sets; within-half order may differ from nth_element.
static inline uint64_t mqa_select_pivot(const uint64_t *a, int n)
{
    const uint64_t x = a[0], y = a[n / 2], z = a[n - 1];
    return std::max(std::min(x, y), std::min(std::max(x, y), z));
}
static inline void mqa_select_vector(uint64_t *a, int n, int kth)
{
    alignas(64) uint64_t tmp[64];
    while (n > 8) {
        const uint64_t pivot = mqa_select_pivot(a, n);
        int lo = 0, hi = n;
        for (int j = 0; j < n; j += 8) {
            auto pg = svwhilelt_b64(uint64_t(j), uint64_t(n));
            auto x = svld1_u64(pg, a + j);
            auto lt = svcmplt_n_u64(pg, x, pivot), gt = svcmpgt_n_u64(pg, x, pivot);
            int nl = svcntp_b64(pg, lt), ng = svcntp_b64(pg, gt);
            hi -= ng;
            svst1_u64(svwhilelt_b64(uint64_t(0), uint64_t(nl)), tmp + lo, svcompact_u64(lt, x));
            svst1_u64(svwhilelt_b64(uint64_t(0), uint64_t(ng)), tmp + hi, svcompact_u64(gt, x));
            lo += nl;
        }
        tmp[lo] = pivot;
        for (int j = 0; j < n; j += 8) {
            auto pg = svwhilelt_b64(uint64_t(j), uint64_t(n));
            svst1_u64(pg, a + j, svld1_u64(pg, tmp + j));
        }
        if (kth == lo)
            return;
        if (kth < lo)
            n = lo;
        else {
            a += lo + 1;
            kth -= lo + 1;
            n -= lo + 1;
        }
    }
    // Insertion sort for the remaining at most eight unique keys.
    for (int j = 1; j < n; ++j) {
        uint64_t x = a[j];
        int k = j;
        while (k && a[k - 1] > x) {
            a[k] = a[k - 1];
            --k;
        }
        a[k] = x;
    }
}
// Abs FP32 bits plus reverse head ID form an exact unsigned key. One UMAXV
// yields both magnitude and tie-broken head, replacing FMAXV + UMINV.
// The 16 finite alpha values use smaller head IDs to break magnitude ties.
template <int Pairs> static inline void mqa_feedback_maxkey(const float *alpha, int *pick)
{
    const auto pg = svptrue_b64();
    auto lo = svand_n_u64_x(pg, svld1uw_u64(pg, reinterpret_cast<const uint32_t *>(alpha)), 0x7fffffffu);
    auto hi = svand_n_u64_x(pg, svld1uw_u64(pg, reinterpret_cast<const uint32_t *>(alpha + 8)), 0x7fffffffu);
    lo = svorr_u64_x(pg, svlsl_n_u64_x(pg, lo, 4), svindex_u64(15, -1));
    hi = svorr_u64_x(pg, svlsl_n_u64_x(pg, hi, 4), svindex_u64(7, -1));
    for (int i = 0; i < 2 * Pairs; ++i) {
        const uint64_t key = svmaxv_u64(pg, svmax_u64_x(pg, lo, hi));
        pick[i] = 15 - int(key & 15);
        lo = svsel_u64(svcmpeq_n_u64(pg, lo, key), svdup_u64(0), lo);
        hi = svsel_u64(svcmpeq_n_u64(pg, hi, key), svdup_u64(0), hi);
    }
}

template <int Pairs>
static inline void mqa_lattice_balance(const bfloat16_t *const *rows, uint32_t *panel,
                                       const float *alpha)
{
    int pick[2 * Pairs];
    mqa_feedback_maxkey<Pairs>(alpha, pick);
    const auto *scales = reinterpret_cast<const float *>(panel + 512);
    for (int pair = 0; pair < Pairs; ++pair) {
        int target = pick[2 * pair], donor = pick[2 * pair + 1];
        if (alpha[target] == 0 || scales[target] == 0 || scales[donor] == 0)
            continue;
        float gain = .5f * (alpha[donor] / alpha[target]);
        float inv0 = to_float(to_bf16(1.f / scales[donor])),
              inv1 = to_float(to_bf16(1.f / scales[target]));
        uint32_t i0, i1;
        memcpy(&i0, &inv0, 4);
        memcpy(&i1, &inv1, 4);
        i0 = (i0 >> 16) | (i0 & 0xffff0000u);
        i1 = (i1 >> 16) | (i1 & 0xffff0000u);
        const bfloat16_t *s0 = rows[donor], *s1 = rows[target];
        uint32_t *dst = panel + target;
        asm volatile("ptrue p0.b\n ptrue p1.s, vl8\n dup z20.s, %w[i0]\n dup z21.s, %w[i1]\n"
                     "ld1rw {z22.s}, p0/z, [%[gain]]\n movz w9, #0x4b40, lsl #16\n dup z23.s, w9\n"
                     "index z24.s, #0, #1\n lsl z24.s, z24.s, #6\n mov w9, #4\n"
                     "1:\n ld1h {z16.h}, p0/z, [%[s0]]\n ld1h {z17.h}, p0/z, [%[s1]]\n"
                     "mov z0.s, #0\n mov z1.s, #0\n mov z2.s, #0\n mov z3.s, #0\n"
                     "bfmlalb z0.s, z16.h, z20.h\n bfmlalt z1.s, z16.h, z20.h\n"
                     "bfmlalb z2.s, z17.h, z21.h\n bfmlalt z3.s, z17.h, z21.h\n"
                     "fadd z4.s, z0.s, z23.s\n fadd z5.s, z1.s, z23.s\n"
                     "fsub z4.s, z4.s, z23.s\n fsub z5.s, z5.s, z23.s\n"
                     "fsub z4.s, z4.s, z0.s\n fsub z5.s, z5.s, z1.s\n"
                     "fmls z2.s, p0/m, z4.s, z22.s\n fmls z3.s, p0/m, z5.s, z22.s\n"
                     "fadd z2.s, z2.s, z23.s\n fadd z3.s, z3.s, z23.s\n"
                     "trn1 z2.h, z2.h, z3.h\n uzp1 z2.b, z2.b, z2.b\n"
                     "st1w {z2.s}, p1, [%[dst], z24.s, uxtw]\n"
                     "add %[s0], %[s0], #64\n add %[s1], %[s1], #64\n add %[dst], %[dst], #512\n"
                     "subs w9, w9, #1\n b.ne 1b\n"
                     : [s0] "+&r"(s0), [s1] "+&r"(s1), [dst] "+&r"(dst)
                     : [i0] "r"(i0), [i1] "r"(i1), [gain] "r"(&gain)
                     : "cc", "memory", "x9", "p0", "p1", "z0", "z1", "z2", "z3", "z4", "z5", "z16",
                       "z17", "z20", "z21", "z22", "z23", "z24");
    }
}

static inline void pack_query_rows(const bfloat16_t *q_data, const float *weights_data,
                                   uint32_t *packed_q, float *reordered_scaled_weights,
                                   int64_t heads, int64_t qpanel_begin, int64_t qpanel_end)
{
    if (qpanel_begin < qpanel_end) {
        // Per query row: the first half of its panels holds the heads with
        // the largest |w| in dual INT8, the second half the others in single INT8
        // (INT8 payload plus row scales occupies at least 2112 bytes).
        const int64_t panels = heads / 16, half = heads / 2;
        const bfloat16_t *rows[64];
        int order[64];
        int64_t ordered = -1;
        for (int64_t p = qpanel_begin; p < qpanel_end;) {
            const int64_t i = p / panels, g = p % panels;
            const bool int8 = g >= panels / 2;
            const int64_t end = std::min(qpanel_end, i * panels + (int8 ? panels : panels / 2));
            const float *const w = weights_data + i * heads;
            if (i != ordered) {
                // the half with the smallest |w| (ties: lower head first): keys
                // are the |w| bits (monotonic for non-negative floats) then the head
                ordered = i;
                uint64_t keys[64];
                for (int h = 0; h < heads; ++h) {
                    uint32_t bits;
                    std::memcpy(&bits, &w[h], 4);
                    keys[h] = (uint64_t(bits & 0x7fffffffu) << 32) | uint32_t(h);
                }
                mqa_select_vector(keys, heads, half);
                for (int h = 0; h < heads; ++h)
                    order[h] = int(keys[h] & 0xffffffffu);
            }
            const bfloat16_t *const q_row = q_data + i * heads * mqa_layout::dimensions;
            float *const w_out = reordered_scaled_weights + i * heads;
            // High slots use order[half + j]; low slots use order[j - half].
            const int64_t slot0 = g * 16, slots = (end - p) * 16;
            for (int64_t j = slot0; j < slot0 + slots; ++j)
                rows[j - slot0] =
                    q_row + order[j < half ? half + j : j - half] * mqa_layout::dimensions;
            // 1: this worker's next query row, dual half from its
            // start, single half from 4 KiB a dual group further
            const bfloat16_t *const next_row = (i + 1) * panels < qpanel_end
                ? q_data + (i + 1) * heads * mqa_layout::dimensions : nullptr;
            if (int8) {
                uint32_t *const i8_out = packed_q + i * mqa_layout::query_row_words(heads) +
                                         (panels / 2) * mqa_layout::panel_words +
                                         (g - panels / 2) * mqa_layout::q8_panel_words;
                mqa_pack_low_query(rows, order + slot0 - half, w, i8_out, w_out + slot0, end - p,
                                   next_row ? reinterpret_cast<const char *>(next_row) + 4096 * (panels / 2)
                                                + 4096 * (g - panels / 2) : nullptr);
                if (heads == 64)
                    for (int64_t panel = 0; panel < end - p; ++panel)
                        mqa_lattice_balance<4>(rows + panel * 16, i8_out + panel * 528,
                                               w_out + slot0 + panel * 16);
            } else {
                mqa_lattice_pack<false, true>(
                    nullptr, rows, packed_q + i * mqa_layout::query_row_words(heads) + g * 1024,
                    nullptr, w_out + slot0, end - p, w, order + half + slot0,
                    next_row ? reinterpret_cast<const char *>(next_row) + 4096 * g : nullptr);
            }
            if (end == (i + 1) * panels) {
                // Low-Q row scales are dead after weight folding and feedback.
                // Ordered summation matches the kernel's zero-start FMLA sequence,
                // so a wholly non-positive panel still reduces to exactly zero.
                float* biases = reinterpret_cast<float*>(packed_q + i * mqa_layout::query_row_words(heads)
                    + (panels / 2) * mqa_layout::panel_words + mqa_layout::scale_word_offset);
                const auto pg = svptrue_b32();
                for (int64_t group = 0; group < panels; ++group)
                    biases[group] = 8388608.f * svadda_f32(pg, 0.f, svld1_f32(pg, w_out + group * 16));
            }
            p = end;
        }
    }
}

static inline void pack_key_range(const bfloat16_t *kv_data, uint32_t *local_k, int64_t token_begin,
                                  int64_t token_end)
{
    // Both representations of this call's range share one worker-private slot.
    mqa_lattice_pack<true>(kv_data + token_begin * 128, nullptr, local_k,
                           local_k + 256 * 64, nullptr, (token_end - token_begin) / 16);
}
// ---- 通用 SME key-half 核 ----
// count and every descriptor's blocks must be positive; masked bounds satisfy
// 0 <= first < last <= blocks * 32 relative to output.
#define KEYHALF_STEP(Target0, Target1) \
    "ld1h {z0.h}, p0/z, [x9, x22, lsl #1]\n" \
    "ld1h {z1.h}, p0/z, [x10, x22, lsl #1]\n" \
    "ld1h {z4.h}, p0/z, [x14, x22, lsl #1]\n" \
    "smopa za" #Target0 ".s, p0/m, p0/m, z0.b, z4.b\n" \
    "smopa za" #Target1 ".s, p0/m, p0/m, z1.b, z4.b\n" \
    "add x22, x22, #32\n"

// one step of an int8 transition: four dimensions per product, with
// x22 counting bytes (the int8 Q panels and K halves are 64 bytes per step).

// 32 heads: one 16-head Q operand (x9) against both key halves
// (x14 half 0 into Target0, x10 half 1 into Target1), dual or single INT8.
// CI8: 0 dual-INT8 rows (four steps), 1 single-INT8 rows (two steps),
// 3 single-INT8 shared-Q rows, 4 dual-INT8 shared-Q rows. All tiles are int32.
#define MQA_ZSTR_(x) #x
#define MQA_ZSTR(x) MQA_ZSTR_(x)
#define KEYHALF_DRAIN_ROW(Prev0, Prev1, Acc0, Acc1, Row, T0, T1) \
    "mova z" MQA_ZSTR(T0) ".s, p0/m, za" #Prev0 "h.s[w12, " #Row "]\n" \
    "mova z" MQA_ZSTR(T1) ".s, p0/m, za" #Prev1 "h.s[w12, " #Row "]\n" \
    "smax z" MQA_ZSTR(T0) ".s, p0/m, z" MQA_ZSTR(T0) ".s, z30.s\n" \
    "smax z" MQA_ZSTR(T1) ".s, p0/m, z" MQA_ZSTR(T1) ".s, z30.s\n" \
    "fmla z" #Acc0 ".s, z" MQA_ZSTR(T0) ".s, z6.s[" #Row "]\n" \
    "fmla z" #Acc1 ".s, z" MQA_ZSTR(T1) ".s, z7.s[" #Row "]\n"

// ZA cells begin at the integer bit pattern of M=2^23, initialized by ADDHA.
// For every reachable dot d, as_float(max(bits(M)+d,bits(M))) - M = max(d,0).
// SVE therefore reads an already usable biased float, without SCVTF. Subtract
// M*sum(alpha) once per 16-head panel, preserving head-wise FMLA order.
// Four pairs of readout vectors alternate FMLA with next-tile SMOPA steps.

#define KEYHALF_DRAIN_X(Prev0, Prev1, W0, W1, Acc0, Acc1) \
    "mov z" #Acc0 ".s, #0\n" \
    "mov z" #Acc1 ".s, #0\n" \
    "mov z28.d, z" #W0 ".d\n" \
    "mov z29.d, z" #W1 ".d\n" \
    "mov w12, #0\n" \
    "91:\n" \
    "dup z6.q, z28.q[0]\n" \
    "dup z7.q, z29.q[0]\n" \
    KEYHALF_DRAIN_ROW(Prev0, Prev1, Acc0, Acc1, 0, 8, 9) \
    KEYHALF_DRAIN_ROW(Prev0, Prev1, Acc0, Acc1, 1, 10, 11) \
    KEYHALF_DRAIN_ROW(Prev0, Prev1, Acc0, Acc1, 2, 8, 9) \
    KEYHALF_DRAIN_ROW(Prev0, Prev1, Acc0, Acc1, 3, 10, 11) \
    "ext z28.b, z28.b, z28.b, #16\n" \
    "ext z29.b, z29.b, z29.b, #16\n" \
    "add w12, w12, #4\n" \
    "cmp w12, #16\n" \
    "b.ne 91b\n" \
    "dup z0.s, z5.s[(" #W0 ") - 24]\n" \
    "dup z1.s, z5.s[(" #W1 ") - 24]\n" \
    "fsub z" #Acc0 ".s, z" #Acc0 ".s, z0.s\n" \
    "fsub z" #Acc1 ".s, z" #Acc1 ".s, z1.s\n"

// one 64-head transition as the assembler macro lat_r3 (defined at the
// start of the kernel's asm, purged at its end). Products into tiles t0 / t1 (ci8
// 0: dual INT8, four steps a read-out row; 1: single INT8, two), read-out of rows
// 0-15 of the previous tiles p0 / p1 with weights w0 / w1 into a0 / a1: in row
// slot r, mova of row r (each before its own product step), smax of row r - 1 and
// fmla of row r - 2; temporaries z8-z13 rotate by row modulo 3.
#define LAT_R3_STEP \
    ".if \\ci8 == 0\n" \
    "ld1h {z0.h}, p0/z, [x9, x22, lsl #1]\n" "ld1h {z1.h}, p0/z, [x10, x22, lsl #1]\n" \
    "ld1h {z4.h}, p0/z, [x14, x22, lsl #1]\n" \
    "smopa za\\t0\\().s, p0/m, p0/m, z0.b, z4.b\n" "smopa za\\t1\\().s, p0/m, p0/m, z1.b, z4.b\n" \
    "add x22, x22, #32\n" \
    ".else\n" \
    "ld1b {z0.b}, p0/z, [x9, x22]\n" "ld1b {z1.b}, p0/z, [x10, x22]\n" "ld1b {z4.b}, p0/z, [x14, x22]\n" \
    "smopa za\\t0\\().s, p0/m, p0/m, z0.b, z4.b\n" "smopa za\\t1\\().s, p0/m, p0/m, z1.b, z4.b\n" \
    "add x22, x22, #64\n" \
    ".endif\n"
#define LAT_R3_MOVA(P, A, B, C) \
    ".if (\\r %% 3) == 0\n" "mova z" A ".s, p0/m, za\\" P "\\()h.s[w12, \\r %% 4]\n" \
    ".elseif (\\r %% 3) == 1\n" "mova z" B ".s, p0/m, za\\" P "\\()h.s[w12, \\r %% 4]\n" \
    ".else\n" "mova z" C ".s, p0/m, za\\" P "\\()h.s[w12, \\r %% 4]\n" ".endif\n"
#define LAT_R3_SMAX2(X, Y) "smax z" X ".s, p0/m, z" X ".s, z30.s\n" "smax z" Y ".s, p0/m, z" Y ".s, z30.s\n"
#define LAT_R3_SMAX(ROW) \
    ".if ((" ROW ") %% 3) == 0\n" LAT_R3_SMAX2("8", "9") \
    ".elseif ((" ROW ") %% 3) == 1\n" LAT_R3_SMAX2("10", "11") \
    ".else\n" LAT_R3_SMAX2("12", "13") ".endif\n"
#define LAT_R3_FMLA2(ROW, X, Y) \
    "fmla z\\a0\\().s, z" X ".s, z6.s[(" ROW ") %% 4]\n" "fmla z\\a1\\().s, z" Y ".s, z7.s[(" ROW ") %% 4]\n"
#define LAT_R3_FMLA(ROW) \
    ".if ((" ROW ") %% 4) == 0\n" \
    "dup z6.q, z\\w0\\().q[(" ROW ") / 4]\n" "dup z7.q, z\\w1\\().q[(" ROW ") / 4]\n" \
    ".endif\n" \
    ".if ((" ROW ") %% 3) == 0\n" LAT_R3_FMLA2(ROW, "8", "9") \
    ".elseif ((" ROW ") %% 3) == 1\n" LAT_R3_FMLA2(ROW, "10", "11") \
    ".else\n" LAT_R3_FMLA2(ROW, "12", "13") ".endif\n"
#define LAT_R3_DEFINE \
    ".macro lat_r3 t0, t1, p0, p1, w0, w1, a0, a1, ci8\n" \
    "zero {za\\t0\\().s, za\\t1\\().s}\n" \
    "addha za\\t0\\().s, p0/m, p0/m, z30.s\n" \
    "addha za\\t1\\().s, p0/m, p0/m, z30.s\n" \
    "mov x22, #0\n" \
    "mov z\\a0\\().s, #0\n" \
    "mov z\\a1\\().s, #0\n" \
    "mov w12, #0\n" \
    ".irp r, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15\n" \
    ".if (\\r > 0) && ((\\r %% 4) == 0)\n" "add w12, w12, #4\n" ".endif\n" \
    LAT_R3_MOVA("p0", "8", "10", "12") \
    LAT_R3_STEP \
    LAT_R3_MOVA("p1", "9", "11", "13") \
    LAT_R3_STEP \
    ".if \\r >= 1\n" LAT_R3_SMAX("\\r - 1") ".endif\n" \
    ".if \\ci8 == 0\n" LAT_R3_STEP ".endif\n" \
    ".if \\r >= 2\n" LAT_R3_FMLA("\\r - 2") ".endif\n" \
    ".if \\ci8 == 0\n" LAT_R3_STEP ".endif\n" \
    MQA_PF_Q_STEP \
    ".endr\n" \
    LAT_R3_SMAX("15") LAT_R3_FMLA("14") LAT_R3_FMLA("15") \
    "dup z0.s, z5.s[\\w0 - 24]\n" \
    "dup z1.s, z5.s[\\w1 - 24]\n" \
    "fsub z\\a0\\().s, z\\a0\\().s, z0.s\n" \
    "fsub z\\a1\\().s, z\\a1\\().s, z1.s\n" \
    ".endm\n"
#define LAT_R3_PROLOGUE LAT_R3_DEFINE
#define LAT_R3_EPILOGUE ".purgem lat_r3\n"
#define LAT_TR64(Target0, Target1, Prev0, Prev1, W0, W1, Acc0, Acc1, CI8, RI8) \
    "lat_r3 " #Target0 ", " #Target1 ", " #Prev0 ", " #Prev1 ", " #W0 ", " #W1 ", " #Acc0 ", " #Acc1 ", " CI8 "\n"

// 普通输出存储，也用于初始 -inf fill。
#define KH_ST(Z, P, ADDR) \
    "st1w {" Z ".s}, " P ", " ADDR "\n"
#define KEYHALF_SUM_STORE \
    "fadd z16.s, z16.s, z17.s\n" \
    "fadd z20.s, z20.s, z21.s\n" \
    "fadd z18.s, z18.s, z19.s\n" \
    "fadd z22.s, z22.s, z23.s\n" \
    "fmla z16.s, p0/m, z18.s, z2.s\n" \
    "fmla z20.s, p0/m, z22.s, z3.s\n" \
    "cmp x11, #0\n" \
    "b.gt 8f\n" \
    "cmp x13, #32\n" \
    "b.lt 8f\n" \
    KH_ST("z16", "p0", "[x24]") \
    KH_ST("z20", "p0", "[x24, #1, mul vl]") \
    "b 9f\n" \
    "8:\n" \
    "index z0.s, #0, #1\n" \
    "dup z1.s, w11\n" \
    "cmpge p1.s, p0/z, z0.s, z1.s\n" \
    "dup z1.s, w13\n" \
    "cmplt p2.s, p0/z, z0.s, z1.s\n" \
    "and p1.b, p0/z, p1.b, p2.b\n" \
    "sel z16.s, p1, z16.s, z31.s\n" \
    KH_ST("z16", "p0", "[x24]") \
    "index z0.s, #0, #1\n" \
    "add z0.s, z0.s, #16\n" \
    "dup z1.s, w11\n" \
    "cmpge p1.s, p0/z, z0.s, z1.s\n" \
    "dup z1.s, w13\n" \
    "cmplt p2.s, p0/z, z0.s, z1.s\n" \
    "and p1.b, p0/z, p1.b, p2.b\n" \
    "sel z20.s, p1, z20.s, z31.s\n" \
    KH_ST("z20", "p0", "[x24, #1, mul vl]") \
    "9:\n" \
    "sub x11, x11, #32\n" \
    "sub x13, x13, #32\n"


// ZA0/1 and ZA2/3 alternate compute/readout. z16..23 hold weighted
// reductions; z2/3 hold low K scales. z8..15 stage four rows from two tiles.
// 64 heads use two Q panels per representation. Dual/single INT8 transitions
// stay interleaved.
// One task context, rather than one 64-byte descriptor per active query.
struct MqaDirectRows {
    const uint32_t *q, *k;
    const float *weights;
    float *output;
    const int64_t *ks, *ke;
    int64_t query_begin, query_end, token_begin, token_end, output_row_bytes;
    const uint32_t *next_q;
    const void *ready;
};
static_assert(sizeof(MqaDirectRows) == 104);
static_assert(offsetof(MqaDirectRows, query_begin) == 48);
static_assert(offsetof(MqaDirectRows, next_q) == 88);


static inline void mqa_direct64(const MqaDirectRows *args)
{
    asm volatile(
        LAT_R3_PROLOGUE
        ".macro direct_scan\n"
        "ldp x14, x15, [%[args], #64]\n"
        "701:\n"
        "cmp x25, x26\n"
        "b.ge 703f\n"
        "ldr x4, [x3, x25, lsl #3]\n"
        "ldr x5, [x6, x25, lsl #3]\n"
        "cmp x4, x14\n"
        "csel x4, x4, x14, gt\n"
        "cmp x5, x15\n"
        "csel x5, x5, x15, lt\n"
        "cmp x4, x5\n"
        "b.lt 703f\n"
        "add x25, x25, #1\n"
        "b 701b\n"
        "703:\n"
        ".endm\n"
        "\n"
        ".macro direct_materialize\n"
        "mov x9, #12416\n"
        "ldr x27, [%[args], #0]\n"
        "madd x27, x25, x9, x27\n"
        "ldr x28, [%[args], #16]\n"
        "add x28, x28, x25, lsl #8\n"
        "ldr x20, [%[args], #24]\n"
        "ldr x9, [%[args], #80]\n"
        "madd x20, x25, x9, x20\n"
        "and x9, x4, #-32\n"
        "add x20, x20, x9, lsl #2\n"
        "sub x2, x4, x9\n"
        "sub x10, x5, x9\n"
        "orr x2, x2, x10, lsl #32\n"
        "add x21, x5, #31\n"
        "and x21, x21, #-32\n"
        "sub x21, x21, x9\n"
        "lsr x21, x21, #5\n"
        "ldr x10, [%[args], #64]\n"
        "sub x9, x9, x10\n"
        "ldr x23, [%[args], #8]\n"
        "add x23, x23, x9, lsl #8\n"
        "ldr x8, [%[args], #8]\n"
        "add x8, x8, #16, lsl #12\n"
        "lsr x9, x9, #4\n"
        "mov x10, #2176\n"
        "madd x8, x9, x10, x8\n"
        ".endm\n"
        "\n"
        ".macro direct_lookahead\n"
        "add x25, x25, #1\n"
        "direct_scan\n"
        "mov x19, #0\n"
        "cmp x25, x26\n"
        "b.ge 704f\n"
        "ldr x17, [%[args], #0]\n"
        "mov x16, #12416\n"
        "madd x17, x25, x16, x17\n"
        "ldr x16, [%[args], #16]\n"
        "add x16, x16, x25, lsl #8\n"
        "prfm pldl1keep, [x16]\n"
        "prfm pldl1keep, [x16, #64]\n"
        "prfm pldl1keep, [x16, #128]\n"
        "prfm pldl1keep, [x16, #192]\n"
        "b 705f\n"
        "704:\n"
        "ldr x17, [%[args], #88]\n"
        "cmp x17, #0\n"
        "csel x17, x27, x17, eq\n"
        "705:\n"
        ".endm\n"
        "smstart\n"
        "ptrue p0.b\n"
        "movz w9, #0x4b00, lsl #16\n"
        "dup z30.s, w9\n"
        // Streaming-mode -inf fill of whole lines (fill_count may be 0).
        "movz w9, #0xff80, lsl #16\n"
        "dup z31.s, w9\n"
        "mov x7, #%c[k8_pair_bytes]\n"
        "ldp x3, x6, [%[args], #32]\n"
        "ldp x25, x26, [%[args], #48]\n"
        "direct_scan\n"
        "cmp x25, x26\n"
        "b.ge 92f\n"
        "direct_materialize\n"
        "direct_lookahead\n"
        "mov w11, w2\n"
        "lsr x13, x2, #32\n"
        "mov x24, x20\n"
        "ld1w {z24.s}, p0/z, [x28]\n"
        "ld1w {z25.s}, p0/z, [x28, #1, mul vl]\n"
        "ld1w {z26.s}, p0/z, [x28, #2, mul vl]\n"
        "ld1w {z27.s}, p0/z, [x28, #3, mul vl]\n"
        "add x15, x27, #2, lsl #12\n"
        "add x15, x15, #2048\n"
        "ld1rqw {z5.s}, p0/z, [x15]\n"
        "zero {za}\n"
        "addha za0.s, p0/m, p0/m, z30.s\n"
        "addha za1.s, p0/m, p0/m, z30.s\n"
        "mov x9, x27\n"
        "add x10, x9, #4096\n"
        "mov x14, x23\n"
        "mov x22, #0\n"
        "mov w15, #32\n"
        "1:\n"
        KEYHALF_STEP(0, 1)
        KEYHALF_STEP(0, 1)
        "subs w15, w15, #1\n"
        "b.ne 1b\n"
        "2:\n"
        // Alternate dual/single INT8 tiles. Four-row readout overlaps SMOPA,
        // integer clamp of biased float bits and weighted FP32 accumulation. High K
        // scales follow the low scales in the compact K panel.
        "add x9, x27, #8192\n"
        "add x10, x9, #%c[q8_panel_bytes]\n"
        "mov x14, x8\n"
        "add x15, x14, #2048\n"
        "ld1w {z2.s}, p0/z, [x15]\n"
        LAT_TR64(2, 3, 0, 1, 24, 25, 16, 17, "1", "0")
        "ld1w {z0.s}, p0/z, [x15, #1, mul vl]\n"
        "fmul z16.s, z16.s, z0.s\n"
        "fmul z17.s, z17.s, z0.s\n"
        "mov x9, x27\n"
        "add x10, x9, #4096\n"
        "add x14, x23, #4096\n"
        LAT_TR64(0, 1, 2, 3, 26, 27, 18, 19, "0", "1")
        "add x9, x27, #8192\n"
        "add x10, x9, #%c[q8_panel_bytes]\n"
        "add x14, x8, #%c[k8_panel_bytes]\n"
        "add x15, x14, #2048\n"
        "ld1w {z3.s}, p0/z, [x15]\n"
        LAT_TR64(2, 3, 0, 1, 24, 25, 20, 21, "1", "0")
        "ld1w {z0.s}, p0/z, [x15, #1, mul vl]\n"
        "fmul z20.s, z20.s, z0.s\n"
        "fmul z21.s, z21.s, z0.s\n"
        "subs x21, x21, #1\n"
        "b.eq 6f\n"
        "add x23, x23, #8192\n"
        "add x8, x8, x7\n"
        "mov x9, x27\n"
        "add x10, x9, #4096\n"
        "mov x14, x23\n"
        LAT_TR64(0, 1, 2, 3, 26, 27, 22, 23, "0", "1")
        KEYHALF_SUM_STORE
        "add x24, x24, #128\n"
        "b 2b\n"
        "6:\n"
        "cmp x25, x26\n"
        "b.ge 5f\n"
        "direct_materialize\n"
        "direct_lookahead\n"
        "mov x9, x27\n"
        "add x10, x9, #4096\n"
        "mov x14, x23\n"
        LAT_TR64(0, 1, 2, 3, 26, 27, 22, 23, "0", "1")
        KEYHALF_SUM_STORE
        "mov x24, x20\n"
        "mov w11, w2\n"
        "lsr x13, x2, #32\n"
        "ld1w {z24.s}, p0/z, [x28]\n"
        "ld1w {z25.s}, p0/z, [x28, #1, mul vl]\n"
        "ld1w {z26.s}, p0/z, [x28, #2, mul vl]\n"
        "ld1w {z27.s}, p0/z, [x28, #3, mul vl]\n"
        "add x15, x27, #2, lsl #12\n"
        "add x15, x15, #2048\n"
        "ld1rqw {z5.s}, p0/z, [x15]\n"
        "b 2b\n"
        "5:\n"
        KEYHALF_DRAIN_X(2, 3, 26, 27, 22, 23)
        KEYHALF_SUM_STORE
        "92:\n"
        "smstop\n"
        ".purgem direct_scan\n"
        ".purgem direct_materialize\n"
        ".purgem direct_lookahead\n"
        LAT_R3_EPILOGUE
        : : [args] "r"(args),
            [qmask] "i"(64 * 256 - 1),
            [q8_panel_bytes] "i"(mqa_layout::q8_panel_bytes),
            [k8_panel_bytes] "i"(mqa_layout::k8_panel_bytes),
            [k8_pair_bytes] "i"(2 * mqa_layout::k8_panel_bytes)
        : "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22",
          "x23", "x24", "x25", "x26", "x27", "x28", "cc", "memory",
          "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7",
          "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31");
}

// ---- H32 专用核：descriptor 生命周期内缓存权重 quarters ----
#define R35_32_PF_Q_STEP \
    "prfm pldl1keep, [x17, x19]\n" \
    "add x19, x19, #64\n" \
    "and x19, x19, #%c[qmask]\n"

// 32 heads: one 16-head Q operand (x9) against both key halves
// (x14 half 0 into Target0, x10 half 1 into Target1), dual or single INT8.
#define R35_32_KEYHALF_STEP_QK2(Target0, Target1) \
    "ld1h {z0.h}, p0/z, [x9, x22, lsl #1]\n" \
    "ld1h {z1.h}, p0/z, [x10, x22, lsl #1]\n" \
    "ld1h {z18.h}, p0/z, [x14, x22, lsl #1]\n" \
    "smopa za" #Target0 ".s, p0/m, p0/m, z0.b, z18.b\n" \
    "smopa za" #Target1 ".s, p0/m, p0/m, z0.b, z1.b\n" \
    "add x22, x22, #32\n"
#define R35_32_KEYHALF_STEP_QK2_I8(Target0, Target1) \
    "ld1b {z0.b}, p0/z, [x9, x22]\n" \
    "ld1b {z1.b}, p0/z, [x10, x22]\n" \
    "ld1b {z18.b}, p0/z, [x14, x22]\n" \
    "smopa za" #Target0 ".s, p0/m, p0/m, z0.b, z18.b\n" \
    "smopa za" #Target1 ".s, p0/m, p0/m, z0.b, z1.b\n" \
    "add x22, x22, #64\n"
// CI8: 3 = single INT8, 4 = dual INT8; one Q panel covers both K halves.
#define R35_32_KEYHALF_CSTEP(Target0, Target1, CI8) \
    ".if (" CI8 ") == 4\n" R35_32_KEYHALF_STEP_QK2(Target0, Target1) \
    ".else\n" R35_32_KEYHALF_STEP_QK2_I8(Target0, Target1) ".endif\n"
#define R35_32_MQA_ZSTR_(x) #x
#define R35_32_MQA_ZSTR(x) R35_32_MQA_ZSTR_(x)

#define R35_32_KEYHALF_CACHED_QUARTER_0(Target0, Target1, Prev0, Prev1, W0, Acc0, Acc1, CI8, RI8) \
    ".if (" #W0 ") == 24\n" \
    R35_32_KEYHALF_QUARTER_ROWS(Target0, Target1, Prev0, Prev1, Acc0, Acc1, CI8, RI8, 2, 2) \
    ".else\n" \
    R35_32_KEYHALF_QUARTER_ROWS(Target0, Target1, Prev0, Prev1, Acc0, Acc1, CI8, RI8, 7, 7) \
    ".endif\n"

#define R35_32_KEYHALF_QUARTER_ROWS(Target0, Target1, Prev0, Prev1, Acc0, Acc1, CI8, RI8, Weight0, Weight1) \
    "mova z8.s, p0/m, za" #Prev0 "h.s[w12, 0]\n" \
    "mova z9.s, p0/m, za" #Prev1 "h.s[w12, 0]\n" \
    ".if (" CI8 ") != 9\n" \
    R35_32_KEYHALF_CSTEP(Target0, Target1, CI8) \
    ".if ((" CI8 ") & 1) == 0\n" \
    R35_32_KEYHALF_CSTEP(Target0, Target1, CI8) \
    ".endif\n" \
    ".endif\n" \
    ".if (" CI8 ") != 9\n" \
    R35_32_PF_Q_STEP \
    ".endif\n" \
    "mova z10.s, p0/m, za" #Prev0 "h.s[w12, 1]\n" \
    "mova z11.s, p0/m, za" #Prev1 "h.s[w12, 1]\n" \
    ".if (" CI8 ") != 9\n" \
    R35_32_KEYHALF_CSTEP(Target0, Target1, CI8) \
    ".if ((" CI8 ") & 1) == 0\n" \
    R35_32_KEYHALF_CSTEP(Target0, Target1, CI8) \
    ".endif\n" \
    ".endif\n" \
    ".if (" CI8 ") != 9\n" \
    R35_32_PF_Q_STEP \
    ".endif\n" \
    "mova z12.s, p0/m, za" #Prev0 "h.s[w12, 2]\n" \
    "mova z13.s, p0/m, za" #Prev1 "h.s[w12, 2]\n" \
    ".if (" CI8 ") != 9\n" \
    R35_32_KEYHALF_CSTEP(Target0, Target1, CI8) \
    ".if ((" CI8 ") & 1) == 0\n" \
    R35_32_KEYHALF_CSTEP(Target0, Target1, CI8) \
    ".endif\n" \
    ".endif\n" \
    ".if (" CI8 ") != 9\n" \
    R35_32_PF_Q_STEP \
    ".endif\n" \
    "mova z14.s, p0/m, za" #Prev0 "h.s[w12, 3]\n" \
    "mova z15.s, p0/m, za" #Prev1 "h.s[w12, 3]\n" \
    ".if (" CI8 ") != 9\n" \
    R35_32_KEYHALF_CSTEP(Target0, Target1, CI8) \
    ".if ((" CI8 ") & 1) == 0\n" \
    R35_32_KEYHALF_CSTEP(Target0, Target1, CI8) \
    ".endif\n" \
    ".endif\n" \
    ".if (" CI8 ") != 9\n" \
    R35_32_PF_Q_STEP \
    ".endif\n" \
    "smax z8.s, p0/m, z8.s, z30.s\n" \
    "smax z9.s, p0/m, z9.s, z30.s\n" \
    "smax z10.s, p0/m, z10.s, z30.s\n" \
    "smax z11.s, p0/m, z11.s, z30.s\n" \
    "smax z12.s, p0/m, z12.s, z30.s\n" \
    "smax z13.s, p0/m, z13.s, z30.s\n" \
    "smax z14.s, p0/m, z14.s, z30.s\n" \
    "smax z15.s, p0/m, z15.s, z30.s\n" \
    "fmla z" #Acc0 ".s, z8.s, z" R35_32_MQA_ZSTR(Weight0) ".s[0]\n" \
    "fmla z" #Acc1 ".s, z9.s, z" R35_32_MQA_ZSTR(Weight1) ".s[0]\n" \
    ".if (" CI8 ") != 9\n" \
    R35_32_KEYHALF_CSTEP(Target0, Target1, CI8) \
    ".if ((" CI8 ") & 1) == 0\n" \
    R35_32_KEYHALF_CSTEP(Target0, Target1, CI8) \
    ".endif\n" \
    ".endif\n" \
    "fmla z" #Acc0 ".s, z10.s, z" R35_32_MQA_ZSTR(Weight0) ".s[1]\n" \
    "fmla z" #Acc1 ".s, z11.s, z" R35_32_MQA_ZSTR(Weight1) ".s[1]\n" \
    ".if (" CI8 ") != 9\n" \
    R35_32_KEYHALF_CSTEP(Target0, Target1, CI8) \
    ".if ((" CI8 ") & 1) == 0\n" \
    R35_32_KEYHALF_CSTEP(Target0, Target1, CI8) \
    ".endif\n" \
    ".endif\n" \
    "fmla z" #Acc0 ".s, z12.s, z" R35_32_MQA_ZSTR(Weight0) ".s[2]\n" \
    "fmla z" #Acc1 ".s, z13.s, z" R35_32_MQA_ZSTR(Weight1) ".s[2]\n" \
    ".if (" CI8 ") != 9\n" \
    R35_32_KEYHALF_CSTEP(Target0, Target1, CI8) \
    ".if ((" CI8 ") & 1) == 0\n" \
    R35_32_KEYHALF_CSTEP(Target0, Target1, CI8) \
    ".endif\n" \
    ".endif\n" \
    "fmla z" #Acc0 ".s, z14.s, z" R35_32_MQA_ZSTR(Weight0) ".s[3]\n" \
    "fmla z" #Acc1 ".s, z15.s, z" R35_32_MQA_ZSTR(Weight1) ".s[3]\n" \
    ".if (" CI8 ") != 9\n" \
    R35_32_KEYHALF_CSTEP(Target0, Target1, CI8) \
    ".if ((" CI8 ") & 1) == 0\n" \
    R35_32_KEYHALF_CSTEP(Target0, Target1, CI8) \
    ".endif\n" \
    ".endif\n"

#define R35_32_KEYHALF_CACHED_QUARTER_1(Target0, Target1, Prev0, Prev1, W0, Acc0, Acc1, CI8, RI8) \
    ".if (" #W0 ") == 24\n" \
    R35_32_KEYHALF_QUARTER_ROWS(Target0, Target1, Prev0, Prev1, Acc0, Acc1, CI8, RI8, 3, 3) \
    ".else\n" \
    "mov z6.d, z22.d\n" \
    R35_32_KEYHALF_QUARTER_ROWS(Target0, Target1, Prev0, Prev1, Acc0, Acc1, CI8, RI8, 6, 6) \
    ".endif\n"

#define R35_32_KEYHALF_CACHED_QUARTER_2(Target0, Target1, Prev0, Prev1, W0, Acc0, Acc1, CI8, RI8) \
    ".if (" #W0 ") == 24\n" \
    R35_32_KEYHALF_QUARTER_ROWS(Target0, Target1, Prev0, Prev1, Acc0, Acc1, CI8, RI8, 4, 4) \
    ".else\n" \
    "mov z6.d, z23.d\n" \
    R35_32_KEYHALF_QUARTER_ROWS(Target0, Target1, Prev0, Prev1, Acc0, Acc1, CI8, RI8, 6, 6) \
    ".endif\n"

#define R35_32_KEYHALF_CACHED_QUARTER_3(Target0, Target1, Prev0, Prev1, W0, Acc0, Acc1, CI8, RI8) \
    ".if (" #W0 ") == 24\n" \
    R35_32_KEYHALF_QUARTER_ROWS(Target0, Target1, Prev0, Prev1, Acc0, Acc1, CI8, RI8, 5, 5) \
    ".else\n" \
    "mov z6.d, z26.d\n" \
    R35_32_KEYHALF_QUARTER_ROWS(Target0, Target1, Prev0, Prev1, Acc0, Acc1, CI8, RI8, 6, 6) \
    ".endif\n"

// 3 bit 2: the 32-head transition (one Q panel against both K halves,
// ci8 4: dual INT8, four steps a read-out row; 3: single INT8, two) as the
// assembler macro lat_r3q, rows read out as lat_r3 does. Both tiles of a row
// share its head weight: w0 24 (reading the dual tiles) takes lane r % 4 of
// z2-z5 (cached quarters 0-3), w0 25 takes z7 for rows 0-3 and z6 (copied from
// z22 / z23 / z26) for the others, as the quarter macros do.
#define LAT_R3Q_STEP \
    ".if \\ci8 == 4\n" R35_32_KEYHALF_STEP_QK2(\\t0\\(), \\t1\\()) \
    ".else\n" R35_32_KEYHALF_STEP_QK2_I8(\\t0\\(), \\t1\\()) ".endif\n"
#define LAT_R3Q_FMLA2(ROW, X, Y, W) \
    "fmla z\\a0\\().s, z" X ".s, z" W ".s[(" ROW ") %% 4]\n" "fmla z\\a1\\().s, z" Y ".s, z" W ".s[(" ROW ") %% 4]\n"
#define LAT_R3Q_FMLA_W(ROW, W) \
    ".if ((" ROW ") %% 3) == 0\n" LAT_R3Q_FMLA2(ROW, "8", "9", W) \
    ".elseif ((" ROW ") %% 3) == 1\n" LAT_R3Q_FMLA2(ROW, "10", "11", W) \
    ".else\n" LAT_R3Q_FMLA2(ROW, "12", "13", W) ".endif\n"
#define LAT_R3Q_FMLA(ROW) \
    ".if \\w0 == 24\n" \
    ".if ((" ROW ") / 4) == 0\n" LAT_R3Q_FMLA_W(ROW, "2") \
    ".elseif ((" ROW ") / 4) == 1\n" LAT_R3Q_FMLA_W(ROW, "3") \
    ".elseif ((" ROW ") / 4) == 2\n" LAT_R3Q_FMLA_W(ROW, "4") \
    ".else\n" LAT_R3Q_FMLA_W(ROW, "5") ".endif\n" \
    ".else\n" \
    ".if ((" ROW ") / 4) == 0\n" LAT_R3Q_FMLA_W(ROW, "7") \
    ".else\n" \
    ".if ((" ROW ") %% 4) == 0\n" \
    ".if ((" ROW ") / 4) == 1\n" "mov z6.d, z22.d\n" \
    ".elseif ((" ROW ") / 4) == 2\n" "mov z6.d, z23.d\n" \
    ".else\n" "mov z6.d, z26.d\n" ".endif\n" \
    ".endif\n" \
    LAT_R3Q_FMLA_W(ROW, "6") \
    ".endif\n" \
    ".endif\n"
#define LAT_R3Q_DEFINE \
    ".macro lat_r3q t0, t1, p0, p1, w0, a0, a1, ci8\n" \
    "zero {za\\t0\\().s, za\\t1\\().s}\n" \
    "addha za\\t0\\().s, p0/m, p0/m, z30.s\n" \
    "addha za\\t1\\().s, p0/m, p0/m, z30.s\n" \
    "mov x22, #0\n" \
    "mov z\\a0\\().s, #0\n" \
    "mov z\\a1\\().s, #0\n" \
    "mov w12, #0\n" \
    ".irp r, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15\n" \
    ".if (\\r > 0) && ((\\r %% 4) == 0)\n" "add w12, w12, #4\n" ".endif\n" \
    LAT_R3_MOVA("p0", "8", "10", "12") \
    LAT_R3Q_STEP \
    LAT_R3_MOVA("p1", "9", "11", "13") \
    LAT_R3Q_STEP \
    ".if \\r >= 1\n" LAT_R3_SMAX("\\r - 1") ".endif\n" \
    ".if \\ci8 == 4\n" LAT_R3Q_STEP ".endif\n" \
    ".if \\r >= 2\n" LAT_R3Q_FMLA("\\r - 2") ".endif\n" \
    ".if \\ci8 == 4\n" LAT_R3Q_STEP ".endif\n" \
    R35_32_PF_Q_STEP \
    ".endr\n" \
    LAT_R3_SMAX("15") LAT_R3Q_FMLA("14") LAT_R3Q_FMLA("15") \
    "dup z0.s, z27.s[\\w0 - 24]\n" \
    "fsub z\\a0\\().s, z\\a0\\().s, z0.s\n" \
    "fsub z\\a1\\().s, z\\a1\\().s, z0.s\n" \
    ".endm\n"
#define LAT_R3Q_PROLOGUE LAT_R3Q_DEFINE
#define LAT_R3Q_EPILOGUE ".purgem lat_r3q\n"
#define LAT_TR32(Target0, Target1, Prev0, Prev1, W0, Acc0, Acc1, CI8, RI8) \
    "lat_r3q " #Target0 ", " #Target1 ", " #Prev0 ", " #Prev1 ", " #W0 ", " #Acc0 ", " #Acc1 ", " CI8 "\n"

#define R35_32_KEYHALF_DRAIN_X(Prev0, Prev1, W0, Acc0, Acc1) \
    "mov z" #Acc0 ".s, #0\n" \
    "mov z" #Acc1 ".s, #0\n" \
    "mov w12, #0\n" \
    R35_32_KEYHALF_CACHED_QUARTER_0(0, 1, Prev0, Prev1, W0, Acc0, Acc1, "9", "1") \
    "mov w12, #4\n" \
    R35_32_KEYHALF_CACHED_QUARTER_1(0, 1, Prev0, Prev1, W0, Acc0, Acc1, "9", "1") \
    "mov w12, #8\n" \
    R35_32_KEYHALF_CACHED_QUARTER_2(0, 1, Prev0, Prev1, W0, Acc0, Acc1, "9", "1") \
    "mov w12, #12\n" \
    R35_32_KEYHALF_CACHED_QUARTER_3(0, 1, Prev0, Prev1, W0, Acc0, Acc1, "9", "1") \
    "dup z0.s, z27.s[(" #W0 ") - 24]\n" \
    "fsub z" #Acc0 ".s, z" #Acc0 ".s, z0.s\n" \
    "fsub z" #Acc1 ".s, z" #Acc1 ".s, z0.s\n"

// Descriptor-lifetime quarter caches; reload only AFTER previous output is stored.
#define R35_32_KEYHALF_CACHE_WEIGHTS \
    "ld1w {z24.s}, p0/z, [x28]\n" \
    "ld1w {z25.s}, p0/z, [x28, #1, mul vl]\n" \
    "dup z2.q, z24.q[0]\n" \
    "dup z3.q, z24.q[1]\n" \
    "dup z4.q, z24.q[2]\n" \
    "dup z5.q, z24.q[3]\n" \
    "dup z7.q, z25.q[0]\n" \
    "dup z22.q, z25.q[1]\n" \
    "dup z23.q, z25.q[2]\n" \
    "dup z26.q, z25.q[3]\n" \
    "add x15, x27, #1, lsl #12\n" \
    "add x15, x15, #2048\n" \
    "ld1rqw {z27.s}, p0/z, [x15]\n"

// Selected non-temporal output stores (including the initial fill list).
#define R35_32_KH_ST(Z, P, ADDR) \
    "stnt1w {" Z ".s}, " P ", " ADDR "\n"
#define R35_32_KEYHALF_SUM_STORE \
    "fmla z16.s, p0/m, z17.s, z28.s\n" \
    "fmla z20.s, p0/m, z21.s, z29.s\n" \
    "cmp x11, #0\n" \
    "b.gt 8f\n" \
    "cmp x13, #32\n" \
    "b.lt 8f\n" \
    R35_32_KH_ST("z16", "p0", "[x24]") \
    R35_32_KH_ST("z20", "p0", "[x24, #1, mul vl]") \
    "b 9f\n" \
    "8:\n" \
    "index z0.s, #0, #1\n" \
    "dup z1.s, w11\n" \
    "cmpge p1.s, p0/z, z0.s, z1.s\n" \
    "dup z1.s, w13\n" \
    "cmplt p2.s, p0/z, z0.s, z1.s\n" \
    "and p1.b, p0/z, p1.b, p2.b\n" \
    "sel z16.s, p1, z16.s, z31.s\n" \
    R35_32_KH_ST("z16", "p0", "[x24]") \
    "index z0.s, #0, #1\n" \
    "add z0.s, z0.s, #16\n" \
    "dup z1.s, w11\n" \
    "cmpge p1.s, p0/z, z0.s, z1.s\n" \
    "dup z1.s, w13\n" \
    "cmplt p2.s, p0/z, z0.s, z1.s\n" \
    "and p1.b, p0/z, p1.b, p2.b\n" \
    "sel z20.s, p1, z20.s, z31.s\n" \
    R35_32_KH_ST("z20", "p0", "[x24, #1, mul vl]") \
    "9:\n" \
    "sub x11, x11, #32\n" \
    "sub x13, x13, #32\n"

// ZA0/1 and ZA2/3 alternate compute/readout. z16/20 reduce dual INT8;
// z17/21 reduce single INT8. z28/29 hold low K scales; z8..15 stage four
// rows from both tiles. z2..5 cache high weights, z7/z22/z23/z26 low
// weights. z6 is a weight scratch and z18 is a product operand.
static inline void mqa_direct32(const MqaDirectRows *args)
{
    asm volatile(
        LAT_R3Q_PROLOGUE
        ".macro direct_scan\n"
        "ldp x14, x15, [%[args], #64]\n"
        "701:\n"
        "cmp x25, x26\n"
        "b.ge 703f\n"
        "ldr x4, [x3, x25, lsl #3]\n"
        "ldr x5, [x6, x25, lsl #3]\n"
        "cmp x4, x14\n"
        "csel x4, x4, x14, gt\n"
        "cmp x5, x15\n"
        "csel x5, x5, x15, lt\n"
        "cmp x4, x5\n"
        "b.lt 703f\n"
        "add x25, x25, #1\n"
        "b 701b\n"
        "703:\n"
        ".endm\n"
        "\n"
        ".macro direct_materialize\n"
        "lsr x16, x7, #32\n"
        "cmp x25, x16\n"
        "b.lt 706f\n"
        "lsr x16, x25, #5\n"
        "ldr x9, [%[args], #96]\n"
        "add x9, x9, x16, lsl #6\n"
        "mov x10, #64\n"
        "707:\n"
        "ldar x14, [x9]\n"
        "cmp x14, x10\n"
        "b.eq 708f\n"
        "yield\n"
        "b 707b\n"
        "708:\n"
        "add x16, x16, #1\n"
        "lsl x16, x16, #5\n"
        "bfi x7, x16, #32, #32\n"
        "706:\n"
        "mov x9, #6208\n"
        "ldr x27, [%[args], #0]\n"
        "madd x27, x25, x9, x27\n"
        "ldr x28, [%[args], #16]\n"
        "add x28, x28, x25, lsl #7\n"
        "ldr x20, [%[args], #24]\n"
        "ldr x9, [%[args], #80]\n"
        "madd x20, x25, x9, x20\n"
        "and x9, x4, #-32\n"
        "add x20, x20, x9, lsl #2\n"
        "sub x2, x4, x9\n"
        "sub x10, x5, x9\n"
        "orr x2, x2, x10, lsl #32\n"
        "add x21, x5, #31\n"
        "and x21, x21, #-32\n"
        "sub x21, x21, x9\n"
        "lsr x21, x21, #5\n"
        "ldr x10, [%[args], #64]\n"
        "sub x9, x9, x10\n"
        "ldr x23, [%[args], #8]\n"
        "add x23, x23, x9, lsl #8\n"
        "ldr x8, [%[args], #8]\n"
        "add x8, x8, #16, lsl #12\n"
        "lsr x9, x9, #4\n"
        "mov x10, #2176\n"
        "madd x8, x9, x10, x8\n"
        ".endm\n"
        "\n"
        ".macro direct_lookahead\n"
        "add x25, x25, #1\n"
        "direct_scan\n"
        "mov x19, #0\n"
        "cmp x25, x26\n"
        "b.ge 704f\n"
        "ldr x17, [%[args], #0]\n"
        "mov x16, #6208\n"
        "madd x17, x25, x16, x17\n"
        "ldr x16, [%[args], #16]\n"
        "add x16, x16, x25, lsl #7\n"
        "prfm pldl1keep, [x16]\n"
        "prfm pldl1keep, [x16, #64]\n"
        "b 705f\n"
        "704:\n"
        "ldr x17, [%[args], #88]\n"
        "cmp x17, #0\n"
        "csel x17, x27, x17, eq\n"
        "705:\n"
        ".endm\n"
        "smstart\n"
        "ptrue p0.b\n"
        "movz w9, #0x4b00, lsl #16\n"
        "dup z30.s, w9\n"
        // Streaming-mode -inf fill of whole lines (fill_count may be 0).
        "movz w9, #0xff80, lsl #16\n"
        "dup z31.s, w9\n"
        "mov x7, #%c[k8_pair_bytes]\n"
        "ldp x3, x6, [%[args], #32]\n"
        "ldp x25, x26, [%[args], #48]\n"
        "direct_scan\n"
        "cmp x25, x26\n"
        "b.ge 92f\n"
        "direct_materialize\n"
        "direct_lookahead\n"
        "mov w11, w2\n"
        "lsr x13, x2, #32\n"
        "mov x24, x20\n"
        R35_32_KEYHALF_CACHE_WEIGHTS
        "zero {za}\n"
        "addha za0.s, p0/m, p0/m, z30.s\n"
        "addha za1.s, p0/m, p0/m, z30.s\n"
        "mov x9, x27\n"
        "add x10, x9, #4096\n"
        "mov x14, x23\n"
        "mov x22, #0\n"
        "add x10, x23, #4096\n"
        "mov w15, #32\n"
        "1:\n"
        R35_32_KEYHALF_STEP_QK2(0, 1)
        R35_32_KEYHALF_STEP_QK2(0, 1)
        "subs w15, w15, #1\n"
        "b.ne 1b\n"
        "2:\n"
        // 32 heads, int8: B (int8 heads 16-31, both key halves, tiles 2/3)
        // reads out A (dual-INT8 heads 0-15, tiles 0/1) into z16 / z20; the next
        // block's A reads out B into z17 / z21; token scales in z28 / z29.
        "add x9, x27, #4096\n"
        "mov x14, x8\n"
        "add x10, x8, #%c[k8_panel_bytes]\n"
        "add x15, x14, #2048\n"
        "ld1w {z28.s}, p0/z, [x15]\n"
        "add x15, x10, #2048\n"
        "ld1w {z29.s}, p0/z, [x15]\n"
        LAT_TR32(2, 3, 0, 1, 24, 16, 20, "3", "0")
        "ld1w {z0.s}, p0/z, [x15, #1, mul vl]\n"
        "fmul z20.s, z20.s, z0.s\n"
        "sub x15, x15, #%c[k8_panel_bytes]\n"
        "ld1w {z0.s}, p0/z, [x15, #1, mul vl]\n"
        "fmul z16.s, z16.s, z0.s\n"
        "subs x21, x21, #1\n"
        "b.eq 6f\n"
        "add x23, x23, #8192\n"
        "add x8, x8, w7, uxtw\n"
        "mov x9, x27\n"
        "mov x14, x23\n"
        "add x10, x23, #4096\n"
        LAT_TR32(0, 1, 2, 3, 25, 17, 21, "4", "1")
        R35_32_KEYHALF_SUM_STORE
        "add x24, x24, #128\n"
        "b 2b\n"
        "6:\n"
        "cmp x25, x26\n"
        "b.ge 5f\n"
        "direct_materialize\n"
        "direct_lookahead\n"
        "mov x9, x27\n"
        "add x10, x9, #4096\n"
        "mov x14, x23\n"
        "add x10, x23, #4096\n"
        LAT_TR32(0, 1, 2, 3, 25, 17, 21, "4", "1")
        R35_32_KEYHALF_SUM_STORE
        "mov x24, x20\n"
        "mov w11, w2\n"
        "lsr x13, x2, #32\n"
        R35_32_KEYHALF_CACHE_WEIGHTS
        "b 2b\n"
        "5:\n"
        R35_32_KEYHALF_DRAIN_X(2, 3, 25, 17, 21)
        R35_32_KEYHALF_SUM_STORE
        "92:\n"
        "smstop\n"
        ".purgem direct_scan\n"
        ".purgem direct_materialize\n"
        ".purgem direct_lookahead\n"
        LAT_R3Q_EPILOGUE
        : : [args] "r"(args),
            [qmask] "i"(32 * 256 - 1),
            [k8_panel_bytes] "i"(mqa_layout::k8_panel_bytes),
            [k8_pair_bytes] "i"(2 * mqa_layout::k8_panel_bytes)
        : "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22",
          "x23", "x24", "x25", "x26", "x27", "x28", "cc", "memory",
          "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7",
          "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31");
}

// ---- KV 任务拆分及算子执行 ----
struct TaskRange {
    int64_t token_begin, token_end, query_block_begin, query_block_end;
};

// Tasks own disjoint output ranges. Case 1 keeps one whole K256 block per
// worker (blocks 0-36), halves blocks 37-55 and quarters blocks 56-63: smaller
// last claims even out the finish times, but each split pays the per-row Q
// reload again, so only the last 8 blocks are quartered.
// Case 2 uses a fixed 81-task table: 37 full-Q K256 tasks, then 44 K128 x Q64
// tail tasks, heaviest first (the halves / eighths below are not used for it).
struct TaskSchedule {
    int64_t tokens, key_blocks, query_blocks;
    int64_t split_after, half_end, eighth_begin_task, task_count;
    int64_t quarter_begin_task = -1;

    TaskSchedule(int64_t token_count, int64_t queries) : tokens(token_count) {
        key_blocks = (tokens + 255) / 256;
        const int64_t rows = mqa_layout::query_group_rows(queries);
        query_blocks = (queries + rows - 1) / rows;
        split_after = query_blocks >= 2
            ? std::max<int64_t>(0, key_blocks - (queries <= 32 ? 24 : 10)) : key_blocks;
        const int64_t eighths = queries > 32 && query_blocks >= 4
            ? std::min<int64_t>(4, key_blocks - split_after) : 0;
        half_end = key_blocks - eighths;
        eighth_begin_task = split_after + 2 * (half_end - split_after);
        task_count = eighth_begin_task + 8 * eighths;
        if (tokens == 12288 && queries == 128) task_count = 81;
        if (tokens == 16384 && queries == 30) {
            split_after = 37;
            half_end = 56;
            quarter_begin_task = split_after + 2 * (half_end - split_after);
            task_count = quarter_begin_task + 4 * (key_blocks - half_end);
        }
    }

    TaskRange map(int64_t task) const {
        if (tokens == 12288 && query_blocks == 4 && task_count == 81) {
            static constexpr uint16_t order[] = {16992,16988,16996,16984,17000,16980,17004,16976,17008,16972,17012,16968,17016,16964,16960,17020,17024,16956,17028,16952,17032,16948,16944,17036,17040,16940,17044,16936,16932,17048,17052,16928,17056,16924,17060,16920,17064,8470,18710,8468,18708,8620,18860,8466,18706,8464,18704,8622,18862,8624,18864,8462,18702,8626,18866,8460,18700,8458,18698,8628,18868,8630,18870,8456,18696,8632,18872,8454,18694,8452,18692,8634,18874,8636,18876,8450,18690,8638,18878,8448,18688};
            const uint16_t value = order[task];
            const int64_t begin = (value & 255) * 64;
            return {begin, begin + (64LL << ((value >> 8) & 3)),
                    (value >> 10) & 3, (value >> 12) & 7};
        }
        if (task < split_after)
            return {task * 256, std::min((task + 1) * 256, tokens), 0, query_blocks};
        if (quarter_begin_task >= 0) {
            const int64_t begin = task < quarter_begin_task
                ? split_after * 256 + (task - split_after) * 128
                : half_end * 256 + (task - quarter_begin_task) * 64;
            return {begin, begin + (task < quarter_begin_task ? 128 : 64), 0, query_blocks};
        }
        if (task < eighth_begin_task) {
            const int64_t begin = split_after * 256 + (task - split_after) * 128;
            return {begin, std::min(begin + 128, tokens), 0, query_blocks};
        }
        const int64_t block = half_end + (task - eighth_begin_task) / 8;
        const int64_t part = (task - eighth_begin_task) & 7;
        const int64_t begin = block * 256 + (part & 3) * 64;
        return {begin, std::min(begin + 64, tokens),
                query_blocks * (part >> 2) / 2, query_blocks * ((part >> 2) + 1) / 2};
    }
};

// One call of a target shape: the Q-row owners pack this call's Q and publish
// it per query block; every worker then claims K tasks, packs the task's K
// privately and runs the direct kernel over the task's query rows; whole masked
// 32-token blocks are filled at the end.
static inline void mqa_indexer_body(
    const Tensor<bfloat16_t, 3> &q,
    const Tensor<bfloat16_t, 2> &kv,
    const Tensor<float, 2> &weights,
    const Tensor<int64_t, 1> &cu_seqlen_ks,
    const Tensor<int64_t, 1> &cu_seqlen_ke,
    const Tensor<float, 2> &logits,
    MqaWorkspace &workspace, MqaPool *pool, int max_workers)
{
    const int64_t queries = q.size(0), heads = q.size(1), tokens = kv.size(0);
    const int64_t qpanels = queries * heads / 16;
    // Each worker keeps dual/single INT8 panels for its current task's K.
    constexpr int64_t k_slots = 2;
    workspace.prepare(static_cast<size_t>(qpanels) * mqa_layout::panel_bytes,
                      static_cast<size_t>(256) * 256 * k_slots,
                      static_cast<size_t>(max_workers));
    // Rotate folded weights with Q: workers must not repeatedly share a line
    // that is immediately overwritten by the next call's Q pack.
    workspace.reordered_scaled_weights.reserve(mqa_layout::query_ring_slots * static_cast<size_t>(queries) * heads * sizeof(float));
    const size_t q_ring_stride = static_cast<size_t>(queries) * mqa_layout::query_row_words(heads) * sizeof(uint32_t);
    const uint32_t *const ring_before = workspace.q_ring_storage.data;
    workspace.q_ring_storage.reserve(mqa_layout::query_ring_slots * q_ring_stride + 64);
    if (workspace.q_ring_storage.data != ring_before)
        // New ring: fault it in now (the first call), not slot by slot in later calls.
        std::memset(workspace.q_ring_storage.data, 0, workspace.q_ring_storage.capacity);
    // Rotate write targets to delay reuse of addresses read by all workers.
    const size_t slot = workspace.q_ring_slot;
    workspace.q_ring_slot = (slot + 1) % mqa_layout::query_ring_slots;
    uint32_t *packed_q = workspace.q_ring_storage.data
        + 64 / sizeof(uint32_t) + slot * q_ring_stride / sizeof(uint32_t);
    const uintptr_t output_begin = reinterpret_cast<uintptr_t>(logits.data_ptr());
    const uintptr_t output_end = output_begin + queries * tokens * sizeof(float);
    // Write-once flags and claim counters only: a line that many workers
    // poll while others update it costs more than the work it protects.
    // Each query block publishes its packed panel count with release semantics.
    constexpr int64_t max_qb = 16;
    struct alignas(64) Counter { std::atomic<int64_t> value; };
    static_assert(sizeof(Counter) == 64, "direct kernel Q-ready stride");
    struct Sync {
        // The unused lines keep every counter at its measured offset: 64-byte
        // counters share 128-byte L3 lines in pairs.
        unsigned char unused0[10 * 64];
        Counter next_task;
        Counter qb_panels[max_qb];
        unsigned char unused1[64];
        alignas(64) uintptr_t populate_begin, populate_end;
        Counter populate_arrived;
        Counter next_fill;
    };
    Sync sync;
    sync.next_task.value.store(0, std::memory_order_relaxed);
    sync.next_fill.value.store(0, std::memory_order_relaxed);
    const int64_t query_block_count = (queries + mqa_layout::query_group_rows(queries) - 1) / mqa_layout::query_group_rows(queries);
    for (int64_t qb = 0; qb < query_block_count; ++qb)
        sync.qb_panels[qb].value.store(0, std::memory_order_relaxed);

    // Output arena prefault: once per mapping and shape, and only
    // for mappings up to 8 GiB.
    uintptr_t &populate_begin = sync.populate_begin, &populate_end = sync.populate_end;
    populate_begin = populate_end = 0;
    const bool known = output_begin >= workspace.populated_begin && output_end <= workspace.populated_end
        && workspace.populated_shape[0] == queries && workspace.populated_shape[1] == heads
        && workspace.populated_shape[2] == tokens;
    if (!known && mqa_anonymous_mapping(output_begin, populate_begin, populate_end)
        && populate_end - populate_begin <= (uintptr_t(8) << 30)) {
        workspace.populated_begin = populate_begin;
        workspace.populated_end = populate_end;
        workspace.populated_shape[0] = queries;
        workspace.populated_shape[1] = heads;
        workspace.populated_shape[2] = tokens;
    }
    const bool populate = populate_end > populate_begin;
    if (populate)
        sync.populate_arrived.value.store(0, std::memory_order_relaxed);
    // Everything a worker reads at startup is this closure (two cache lines,
    // captured by value): every master-written line that all workers read
    // costs each of them a serialized transfer.
    struct alignas(64) CallArgs {
        const bfloat16_t *q_data, *kv_data;
        const float *weights_data;
        const int64_t *ks_data, *ke_data;
        float *logits_data;
        uint32_t *packed_q, *local_k_base;
        Sync *sync;
        MqaWorkspace *workspace;
        int32_t queries, heads, tokens;
        uint32_t populate;  // sync->populate_* holds a span to populate
        float *folded_weights;
    };
    static_assert(sizeof(CallArgs) == 128);
    const CallArgs call{q.data_ptr(), kv.data_ptr(), weights.data_ptr(), cu_seqlen_ks.data_ptr(),
                        cu_seqlen_ke.data_ptr(), logits.data_ptr(), packed_q,
                        workspace.worker_kv[0].data, &sync, &workspace,
                        static_cast<int32_t>(queries), static_cast<int32_t>(heads), static_cast<int32_t>(tokens),
                        populate ? 1u : 0u,
                        reinterpret_cast<float*>(workspace.reordered_scaled_weights.data) + slot * queries * heads};
    auto body = [a = call](int64_t tid, int64_t workers) {
        const int64_t queries = a.queries, heads = a.heads, tokens = a.tokens;
        const bfloat16_t *const q_data = a.q_data, *const kv_data = a.kv_data;
        const float *const weights_data = a.weights_data;
        const int64_t *const ks_data = a.ks_data, *const ke_data = a.ke_data;
        float *const logits_data = a.logits_data;
        uint32_t *const packed_q = a.packed_q;
        const int64_t rows_per_block = mqa_layout::query_group_rows(queries);
        const int64_t query_block_count = (queries + rows_per_block - 1) / rows_per_block;
        Sync &sync = *a.sync;
        auto &next_task = sync.next_task;
        auto &qb_panels = sync.qb_panels;
        // The rare first-call populate span lives in sync.
        const uintptr_t populate_begin = a.populate ? sync.populate_begin : 0;
        const uintptr_t populate_end = a.populate ? sync.populate_end : 0;
        float *const reordered_scaled_weights = a.folded_weights;
        // The worker with the most timer gaps at startup packs no Q rows, takes no static task and claims
        // none near the end; rank numbers the others.
        const int quiet_worker = workers >= 3
            ? mqa_quiet_worker.load(std::memory_order_relaxed) : -1;
        const bool quiet = tid == quiet_worker;
        const bool has_quiet = quiet_worker >= 0 && quiet_worker < workers;
        const int64_t rank = tid - (has_quiet && tid > quiet_worker ? 1 : 0);
        const int64_t q_owners = has_quiet ? workers - 1 : workers;
        // Case 2 ends with the heaviest static tasks (centre blocks, the first
        // ranks): those workers pack no Q rows, so their tasks start earlier
        // while the lighter-loaded ranks pack all of Q.
        const int64_t q_skip = queries == 128 && workers >= 24 ? 8 : 0;
        const int64_t q_packers = q_owners - q_skip;
        const bool q_packer = !quiet && rank >= q_skip;
        const int64_t query_row_begin = q_packer ? queries * std::min(rank - q_skip, q_packers) / q_packers : 0;
        const int64_t query_row_end = q_packer ? queries * std::min(rank - q_skip + 1, q_packers) / q_packers : 0;
        for (const char *line = reinterpret_cast<const char *>(q_data + query_row_begin * heads * 128),
                        *rows_end = reinterpret_cast<const char *>(q_data + std::min(query_row_end,
                            query_row_begin + 1) * heads * 128);
             line < rows_end; line += 64)
            asm volatile("prfm pldl2keep, [%0]" : : "r"(line));
        const TaskSchedule schedule(tokens, queries);
        const int64_t task_count = schedule.task_count;
        // Every worker except the quiet one starts with one static task.
        const int64_t static_tasks = std::min<int64_t>(q_owners, task_count);
        int64_t first_task = !quiet && rank < static_tasks ? rank : -1;
        const int64_t qpanel_begin = query_row_begin * (heads / 16);
        const int64_t qpanel_end = query_row_end * (heads / 16);
        if (populate_end > populate_begin) {
            // 2 MiB-aligned slices per worker. Atomic +0 write-faults absent
            // pages and leaves present data (and concurrent writes) intact.
            const uintptr_t span = (populate_end - populate_begin + (2UL << 20) - 1) / (2UL << 20);
            const uintptr_t lo = populate_begin + (span * tid / workers) * (2UL << 20);
            const uintptr_t hi = std::min(populate_end, populate_begin + (span * (tid + 1) / workers) * (2UL << 20));
            for (uintptr_t page = lo; page < hi; page += 4096)
                __atomic_fetch_add(reinterpret_cast<uint32_t *>(page), 0u, __ATOMIC_RELAXED);
            // Complete the mapping-wide prefault before any output store.
            mqa_fetch_add_acq_rel(sync.populate_arrived.value, 1);
            while (sync.populate_arrived.value.load(std::memory_order_acquire) != workers)
                asm volatile("yield" ::: "memory");
        }
        pack_query_rows(q_data, weights_data, packed_q, reordered_scaled_weights,
                        heads, qpanel_begin, qpanel_end);
        const int64_t panels_per_query = heads / 16;
        for (int64_t qb = 0; qb < query_block_count; ++qb) {
            const int64_t lo = qb * rows_per_block * panels_per_query;
            const int64_t hi = std::min(queries, (qb + 1) * rows_per_block) * panels_per_query;
            const int64_t n = std::min(hi, qpanel_end) - std::max(lo, qpanel_begin);
            if (n <= 0) continue;
            // Release-add publishes packed panels without waiting for a return value.
            asm volatile(".arch_extension lse\n" "staddl %x[v], [%[p]]" : : [v] "r"(n),
                         [p] "r"(&qb_panels[qb].value) : "memory");
        }

        // The task's K range, packed into this worker's private slot.
        uint32_t *const local_k = a.local_k_base + tid * (256 * 64 * k_slots);
        int64_t packed_token_begin = -1, packed_token_end = -1;
        // Case 1 waits for a query block's Q before its kernel call (the 32-head
        // kernel acquires the ready counts itself, per Q32 group).
        uint32_t qb_seen = 0;
        auto wait_q = [&](int64_t qb) {
            const int64_t expect = (std::min(queries, (qb + 1) * rows_per_block) - qb * rows_per_block) * (heads / 16);
            while ((qb_seen & (1u << qb)) == 0) {
                if (qb_panels[qb].value.load(std::memory_order_acquire) == expect)
                    qb_seen |= 1u << qb;
                else
                    asm volatile("yield" ::: "memory");
            }
        };
        // The first active Q row of a query block, prefetched by the kernel
        // during the block before it.
        auto first_row_q = [&](int64_t token_begin, int64_t token_end, int64_t query_block) -> const uint32_t * {
            const int64_t nb = query_block * rows_per_block;
            const int64_t ne = std::min(nb + rows_per_block, queries);
            for (int64_t i = nb; i < ne; ++i)
                if (std::max(token_begin, ks_data[i]) < std::min(token_end, ke_data[i]))
                    return packed_q + i * mqa_layout::query_row_words(heads);
            return nullptr;
        };
        auto claim = [&]() -> int64_t {
            if (first_task >= 0) {
                const int64_t task = first_task;
                first_task = -1;
                return task < task_count ? task : -1;
            }
            // The quiet worker leaves the last tasks to the others.
            if (quiet && static_tasks + next_task.value.load(std::memory_order_relaxed)
                             >= task_count - 20)
                return -1;
            const int64_t task = static_tasks + mqa_fetch_add_relaxed(next_task.value, 1);
            return task < task_count ? task : -1;
        };
        for (int64_t task = claim(); task >= 0; task = claim()) {
            const TaskRange range = schedule.map(task);
            const int64_t token_begin = range.token_begin, token_end = range.token_end;
            if (token_begin >= token_end) continue;
            if (token_begin != packed_token_begin || token_end != packed_token_end) {
                pack_key_range(kv_data, local_k, token_begin, token_end);
                packed_token_begin = token_begin;
                packed_token_end = token_end;
            }
            // Static tasks start together, each first reading all Q rows that
            // the other workers just packed: starting them at different query
            // blocks spreads those first reads.
            const int64_t qb_count = range.query_block_end - range.query_block_begin;
            const int64_t rot = task < static_tasks && qb_count > 1 ? rank % qb_count : 0;
            if (heads == 32) {
                // Rows [split, end), then [begin, split).
                const int64_t rows_begin = range.query_block_begin * rows_per_block;
                const int64_t rows_end = std::min(queries, range.query_block_end * rows_per_block);
                const int64_t split = std::min(rows_end, rows_begin + rot * rows_per_block);
                for (int part = 0; part < (split != rows_begin ? 2 : 1); ++part) {
                    const MqaDirectRows direct{packed_q, local_k, reordered_scaled_weights, logits_data,
                        ks_data, ke_data, part ? rows_begin : split, part ? split : rows_end,
                        token_begin, token_end, tokens * 4, nullptr, &qb_panels[0].value};
                    mqa_direct32(&direct);
                }
                continue;
            }
            for (int64_t k = 0; k < qb_count; ++k) {
                const int64_t qb = range.query_block_begin + (k + rot) % qb_count;
                wait_q(qb);
                const int64_t query_begin = qb * rows_per_block;
                const uint32_t *const next_q = k + 1 < qb_count
                    ? first_row_q(token_begin, token_end, range.query_block_begin + (k + 1 + rot) % qb_count) : nullptr;
                const MqaDirectRows direct{packed_q, local_k, reordered_scaled_weights, logits_data,
                    ks_data, ke_data, query_begin, std::min(query_begin + rows_per_block, queries),
                    token_begin, token_end, tokens * 4, next_q, &qb_panels[0].value};
                mqa_direct64(&direct);
            }
        }

        if (!quiet) {
            auto fill_row = [&](int64_t row) {
                float *out = logits_data + row * tokens;
                const int64_t lo = std::clamp<int64_t>(ks_data[row], 0, tokens);
                const int64_t hi = std::clamp<int64_t>(ke_data[row], 0, tokens);
                if (lo >= hi) { mqa_row_fill_nt(out, tokens); return; }
                mqa_row_fill_nt(out, lo / 32 * 32);
                const int64_t aligned_hi = (hi + 31) / 32 * 32;
                mqa_row_fill_nt(out + aligned_hi, tokens - aligned_hi);
            };
            // Case 2 takes four rows a claim: with one row a claim ~30 workers keep
            // the counter's line busy while the last tasks still compute (~1.5 us a
            // call). Case 1's 30 rows need single rows to balance.
            const int64_t fill_rows = queries == 128 ? 4 : 1;
            for (;;) {
                const int64_t begin = mqa_fetch_add_relaxed(sync.next_fill.value, fill_rows);
                if (begin >= queries) break;
                for (int64_t row = begin; row < std::min(begin + fill_rows, queries); ++row)
                    fill_row(row);
            }
        }
        if (tid == 0) {
            // The caller's next call starts with omp_get_max_threads() and the
            // workspace fields; after a long call both are a chain of cold
            // misses (~1 us), so touch them while waiting for the others.
            (void)omp_get_max_threads();
            const char *ws = reinterpret_cast<const char *>(a.workspace);
            for (size_t offset = 0; offset < sizeof(MqaWorkspace); offset += 64)
                asm volatile("prfm pldl1keep, [%0]" : : "r"(ws + offset));
        }
    };
    mqa_parallel(body, max_workers, pool);
}

#undef MQA_PF_Q_STEP
#undef MQA_Q8_QUANT2
#undef MQA_Q8_MOVA4
#undef LAT_ABSMAX_ITER
#undef LAT_ABSMAX_FINAL
#undef LAT_ABSMAX_REV
#undef LAT_KPACK_PFS_STEP
#undef LAT_QLOW_PF_STEP
#undef KEYHALF_STEP
#undef MQA_ZSTR_
#undef MQA_ZSTR
#undef KEYHALF_DRAIN_ROW
#undef KEYHALF_DRAIN_X
#undef LAT_R3_STEP
#undef LAT_R3_MOVA
#undef LAT_R3_SMAX2
#undef LAT_R3_SMAX
#undef LAT_R3_FMLA2
#undef LAT_R3_FMLA
#undef LAT_R3_DEFINE
#undef LAT_R3_PROLOGUE
#undef LAT_R3_EPILOGUE
#undef LAT_TR64
#undef KH_ST
#undef KEYHALF_SUM_STORE
#undef R35_32_PF_Q_STEP
#undef R35_32_KEYHALF_STEP_QK2
#undef R35_32_KEYHALF_STEP_QK2_I8
#undef R35_32_KEYHALF_CSTEP
#undef R35_32_MQA_ZSTR_
#undef R35_32_MQA_ZSTR
#undef R35_32_KEYHALF_CACHED_QUARTER_0
#undef R35_32_KEYHALF_QUARTER_ROWS
#undef R35_32_KEYHALF_CACHED_QUARTER_1
#undef R35_32_KEYHALF_CACHED_QUARTER_2
#undef R35_32_KEYHALF_CACHED_QUARTER_3
#undef LAT_R3Q_STEP
#undef LAT_R3Q_FMLA2
#undef LAT_R3Q_FMLA_W
#undef LAT_R3Q_FMLA
#undef LAT_R3Q_DEFINE
#undef LAT_R3Q_PROLOGUE
#undef LAT_R3Q_EPILOGUE
#undef LAT_TR32
#undef R35_32_KEYHALF_DRAIN_X
#undef R35_32_KEYHALF_CACHE_WEIGHTS
#undef R35_32_KH_ST
#undef R35_32_KEYHALF_SUM_STORE

}  // namespace mqa_lat
inline void indexer_bf16_mqa_logits(
    const Tensor<bfloat16_t, 3> &q,                             // [seq_len, num_heads, dim]
    const Tensor<bfloat16_t, 2> &kv,                            // [seq_len_kv, dim]
    const Tensor<float, 2> &weights,                            // [seq_len, num_heads]
    const Tensor<int64_t, 1> &cu_seqlen_ks,                     // [seq_len]
    const Tensor<int64_t, 1> &cu_seqlen_ke,                     // [seq_len]
    const Tensor<float, 2> &logits                              // [seq_len, seq_len_kv]
) {

    const int64_t queries = q.size(0), heads = q.size(1), tokens = kv.size(0);
    if (queries == 0 || tokens == 0) return;
    int64_t svl;
    asm volatile("rdsvl %0, #1" : "=r"(svl));
    // Only the two official shapes take the INT8 path.
    const bool target = (queries == 30 && heads == 64 && tokens == 16384)
        || (queries == 128 && heads == 32 && tokens == 12288);
    if (!target || svl != 64 || svcntw() != 16 || q.size(2) != 128
        || !q.is_contiguous() || !kv.is_contiguous() || !weights.is_contiguous()
        || !cu_seqlen_ks.is_contiguous() || !cu_seqlen_ke.is_contiguous()
        || !logits.is_contiguous()) {
        ref_bf16_mqa_logits<float>(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, logits);
        return;
    }
    const int workers = omp_get_max_threads();
    // Created on the first supported call, inside this operator. Subsequent
    // calls reuse the workers; run() waits for this call's work before return.
    static mqa_lat::MqaPool *const pool = mqa_lat::MqaPool::create(workers);

    // A concurrent caller owns another TLS workspace. Reentry on this caller
    // thread receives an independent temporary workspace until its call ends.
    // The lease outlives the entire OpenMP region, including its final join.
    static thread_local mqa_lat::MqaWorkspace workspace;
    mqa_lat::MqaWorkspaceLease lease(workspace);
    mqa_lat::mqa_indexer_body(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, logits, lease.get(), pool, workers);
}
