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
#include <iomanip>
#include <cstring>
#include <fstream>
#include <vector>
#include <numeric>
#include <iostream>
#include <random>

#include "Tensor.h"
#include "utils.h"
#include "allocator.h"
#include "testcase.h"
#include "ref_mqa_logits.h"
#include "indexer_mqa_logits.h"

constexpr int64_t MAX_ALLOC_SIZE = 2000LL * 1024 * 1024;

std::vector<TestcaseParams> gen_testcase()
{
    constexpr int64_t DIM = 128;

    std::vector<TestcaseParams> testcases;
    std::vector<std::tuple<int64_t, int64_t, int64_t>> params_list = {
        {30, 16384, 64},
        {128, 12288, 32},
    };
    for (auto [seq_len, seq_len_kv, num_heads] : params_list) {
        TestcaseParams p;
        p.seq_len = seq_len;
        p.seq_len_kv = seq_len_kv;
        p.num_heads = num_heads;
        p.dim = DIM;
        p.check_correctness = true;
        p.num_runs = 100;
        
        testcases.push_back(p);
    }
    return testcases;
}

void run_test(const TestcaseParams &p, int64_t rand_seed = 0)
{
    static int64_t testcase_count = 0;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Running on Testcase " << (++testcase_count) << ":" << std::endl;
    std::cout << "Parameters: seq_len = " << p.seq_len
              << ", seq_len_kv = " << p.seq_len_kv
              << ", num_heads = " << p.num_heads
              << ", dim = " << p.dim << std::endl;

    std::mt19937 rng(rand_seed);

    Allocator allocator(MAX_ALLOC_SIZE);
    auto datalist = generate_datalist(p, allocator, rng);

    double total_time_usage_us = 0;
    int64_t total_k_len = 0;
    for (int64_t i = 0; i < p.num_runs; ++i) {
        auto &d = datalist[i % datalist.size()];
        auto begin = get_clock_us();
        indexer_bf16_mqa_logits(d.q, d.kv, d.weights, d.cu_seqlen_ks, d.cu_seqlen_ke, d.logits);
        auto end = get_clock_us();
        if (i > 0) {
            total_time_usage_us += end - begin;
            total_k_len += d.total_k_len;
        }
    }

    if (p.check_correctness) {
        auto &d = datalist[rng() % datalist.size()];
        ref_bf16_mqa_logits<double>(d.q, d.kv, d.weights, d.cu_seqlen_ks, d.cu_seqlen_ke, d.ref_logits);
        FLASH_ASSERT(check_mask_and_replace(d.logits, d.ref_logits));
        auto cos_diff = calc_cos_diff(d.logits, d.ref_logits);
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "cos_diff: " << cos_diff << std::endl;
        FLASH_ASSERT(cos_diff < 5e-6);
    }

    if (p.num_runs > 1) {
        auto avg_k_len = (double)total_k_len / (p.num_runs - 1);

        double flops = 2.0 * avg_k_len * p.num_heads * p.dim;
        double bytes = 
            sizeof(bfloat16_t) * p.seq_len * p.num_heads * p.dim +
            sizeof(bfloat16_t) * p.seq_len_kv * p.dim +
            sizeof(float) * p.seq_len * p.seq_len_kv;

        double avg_time_usage_us = total_time_usage_us / (p.num_runs - 1);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Performance: " << (flops / avg_time_usage_us) / (1e12 / 1e6) << " TFLOPS" << std::endl;
    }
}

int main(int argc, char *argv[])
{
    auto test_cases = gen_testcase();

    std::cout << "Total Number of Testcases: " << test_cases.size() << std::endl;
    
    for (const auto &p : test_cases) {
        run_test(p);
    }

    std::cout << std::string(80, '=') << std::endl;
    std::cout << "All Tests Ends" << std::endl;

    return 0;
}