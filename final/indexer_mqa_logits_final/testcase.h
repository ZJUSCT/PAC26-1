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

#include <vector>
#include <optional>
#include <omp.h>

#include "Tensor.h"
#include "utils.h"
#include "allocator.h"

struct TestcaseParams {
    int64_t seq_len;
    int64_t seq_len_kv;

    int64_t num_heads;
    int64_t dim;

    bool check_correctness;
    int64_t num_runs;
};

struct TestcaseData {
    Tensor<bfloat16_t, 3> q;                                    // [seq_len, num_heads, dim]
    Tensor<bfloat16_t, 2> kv;                                   // [seq_len_kv, dim]
    Tensor<float, 2> weights;                                   // [seq_len, num_heads]
    Tensor<int64_t, 1> cu_seqlen_ks;                            // [seq_len]
    Tensor<int64_t, 1> cu_seqlen_ke;                            // [seq_len]

    Tensor<float, 2> logits;                                    // [seq_len, seq_len_kv]
    Tensor<float, 2> ref_logits;                                // [seq_len, seq_len_kv]

    int64_t total_k_len;
};

inline int64_t calculate_testcase_data_max_size(const TestcaseParams &p)
{
    int64_t total_size = 0;

    total_size += sizeof(bfloat16_t) * p.seq_len * p.num_heads * p.dim;         // q
    total_size += sizeof(bfloat16_t) * p.seq_len_kv * p.dim;                    // kv
    total_size += sizeof(float) * p.seq_len * p.num_heads;                      // weights
    total_size += sizeof(int64_t) * p.seq_len;                                  // cu_seqlen_ks
    total_size += sizeof(int64_t) * p.seq_len;                                  // cu_seqlen_ke

    total_size += sizeof(float) * p.seq_len * p.seq_len_kv;                     // logits
    total_size += sizeof(float) * p.seq_len * p.seq_len_kv;                     // ref_logits

    return total_size + DEFAULT_ALIGNMENT * 7;
}

inline TestcaseData generate_testcase_data(const TestcaseParams &p, Allocator &allocator, std::mt19937 &rng_base)
{
    std::vector<std::mt19937> rng_mt;
    for (int64_t i = 0; i < omp_get_max_threads(); ++i) {
        auto seed = rng_base();
        rng_mt.push_back(std::mt19937(seed));
    }
    auto gen_float = [&]() -> float {
        std::normal_distribution<> distr;
        int64_t tid = omp_get_thread_num();
        return std::min(std::max(-3.0, distr(rng_mt[tid])), 3.0) / 3;
    };
    auto gen_bf16 = [&]() -> bfloat16_t {
        return to_bf16(gen_float());
    };
    auto gen_int = [&](int l, int r) -> int {
        std::uniform_int_distribution<> distr(l, r);
        int64_t tid = omp_get_thread_num();
        return distr(rng_mt[tid]);
    };

    auto q = generate_tensor<bfloat16_t, 3, true>(allocator, {p.seq_len, p.num_heads, p.dim}, gen_bf16);

    auto kv = generate_tensor<bfloat16_t, 2, true>(allocator, {p.seq_len_kv, p.dim}, gen_bf16);

    auto weights = generate_tensor<float, 2, true>(allocator, {p.seq_len, p.num_heads}, gen_float);

    auto cu_seqlen_ks = create_tensor<int64_t, 1>(allocator, {p.seq_len});
    auto cu_seqlen_ke = create_tensor<int64_t, 1>(allocator, {p.seq_len});

    int64_t total_k_len = 0;

    for (int64_t i = 0; i < p.seq_len; ++i) {
        int64_t len = gen_int(1, p.seq_len_kv);
        int64_t begin = gen_int(0, p.seq_len_kv - len);
        int64_t end = begin + len;
        cu_seqlen_ks.index({i}) = begin;
        cu_seqlen_ke.index({i}) = end;
        total_k_len += len;
    }

    auto logits = create_tensor<float, 2>(allocator, {p.seq_len, p.seq_len_kv});

    auto ref_logits = create_tensor<float, 2>(allocator, {p.seq_len, p.seq_len_kv});
    
    return {q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, logits, ref_logits, total_k_len};
}

inline std::vector<TestcaseData> generate_datalist(const TestcaseParams &p, Allocator &allocator, std::mt19937 &rng)
{
    std::vector<TestcaseData> data_list;

    int64_t single_size = calculate_testcase_data_max_size(p);
    while (allocator.remain() >= single_size && data_list.size() < p.num_runs) {
        TestcaseData data = generate_testcase_data(p, allocator, rng);
        data_list.push_back(data);
    }

    return data_list;
}