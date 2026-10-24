/*
 * Copyright (c) 2026 Huawei Technologies Co., Lt All Rights Reserve
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

#include <iostream>
#include <iomanip>
#include <vector>
#include <optional>
#include <limits>
#include <string>
#include <sstream>
#include <cmath>
#include <omp.h>

#include "Tensor.h"
#include "utils.h"
#include "testcase.h"

template <typename AccumType>
inline void ref_bf16_mqa_logits(
    const Tensor<bfloat16_t, 3> &q,                             // [seq_len, num_heads, dim]
    const Tensor<bfloat16_t, 2> &kv,                            // [seq_len_kv, dim]
    const Tensor<float, 2> &weights,                            // [seq_len, num_heads]
    const Tensor<int64_t, 1> &cu_seqlen_ks,                     // [seq_len]
    const Tensor<int64_t, 1> &cu_seqlen_ke,                     // [seq_len]
    const Tensor<float, 2> &logits                              // [seq_len, seq_len_kv]
) {
    int64_t seq_len = q.size(0);
    int64_t num_heads = q.size(1);
    int64_t dim = q.size(2);
    int64_t seq_len_kv = kv.size(0);

    #pragma omp parallel for
    for (int64_t i = 0; i < seq_len; ++i) {
        int64_t begin = cu_seqlen_ks.index({i});
        int64_t end = cu_seqlen_ke.index({i});
        for (int j = 0; j < seq_len_kv; ++j) {
            if (j >= begin && j < end) {
                AccumType sum = 0;
                for (int64_t h = 0; h < num_heads; ++h) {
                    AccumType dot = 0;
                    for (int64_t d = 0; d < dim; ++d) {
                        dot += to_float(q.index({i, h, d})) * to_float(kv.index({j, d}));
                    }
                    sum += std::max<AccumType>(0, dot) * (AccumType)weights.index({i, h});
                }
                logits.index({i, j}) = sum;
            } else {
                logits.index({i, j}) = -INFINITY;
            }
        }
    }
}