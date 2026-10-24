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

#include <cmath>
#include <cstdint>
#include <arm_sme.h>
#include <arm_neon.h>
#include <vector>
#include <numeric>
#include <limits>
#include <time.h>
#include <iostream>
#include <random>

#include "Tensor.h"

#define FLASH_ASSERT(cond)                                                                                      \
    do {                                                                                                        \
        if (__builtin_expect(!(cond), 0)) {                                                                     \
            std::cerr << "Assertion failed (" << __FILE__ << ":" << __LINE__ << "): " << #cond << std::endl;    \
            exit(1);                                                                                            \
        }                                                                                                       \
    } while (0)

template <typename T>
constexpr T ceil_div(const T &x, const T &y)
{
    return (x + y - 1) / y;
}

template <typename T>
constexpr T ceil(const T &x, const T &y)
{
    return ceil_div(x, y) * y;
}

inline auto get_clock_us()
{
    struct timespec nowtime;
    clock_gettime(CLOCK_MONOTONIC, &nowtime);
    return nowtime.tv_sec * 1e6 + nowtime.tv_nsec * 1e-3;
}

static inline bfloat16_t to_bf16(float x)
{
    return vcvth_bf16_f32(x);
}

static inline float to_float(bfloat16_t x)
{
    return vcvtah_f32_bf16(x);
}