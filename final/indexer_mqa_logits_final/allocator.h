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

#include <sys/mman.h>
#include <cstdlib>
#include <unistd.h>
#include <iostream>

#include "utils.h"

constexpr int64_t DEFAULT_ALIGNMENT = 64;

const int64_t PAGE_SIZE = sysconf(_SC_PAGESIZE);

struct Allocator {
    void *begin_addr;
    void *cur_addr;
    int64_t max_size = 0;

    Allocator(int64_t max_size_)
    {
        max_size = ceil(max_size_, PAGE_SIZE);
        begin_addr = cur_addr = aligned_alloc(PAGE_SIZE, max_size);
    }

    ~Allocator()
    {
	free(begin_addr);
    }

    int64_t remain()
    {
        return max_size - (reinterpret_cast<int64_t>(cur_addr) - reinterpret_cast<int64_t>(begin_addr));
    }

    void *alloc(int64_t size, int64_t alignment = DEFAULT_ALIGNMENT)
    {
        void *addr = reinterpret_cast<void *>(ceil(reinterpret_cast<int64_t>(cur_addr), alignment));
        FLASH_ASSERT((int8_t *)addr + size <= (int8_t *)begin_addr + max_size);
        cur_addr = (int8_t *)addr + size;
        return addr;
    }

    void reset()
    {
        cur_addr = begin_addr;
    }
};
