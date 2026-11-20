# ZJUSCT PAC26-1 Archive

本仓库收录了 **HwHiAIUser** 队在 **第十三届并行应用挑战赛（PAC 2026）** 初赛和决赛中的优化方案与答辩材料，获优化组 **一等奖（2/242）**。

Authors (in no particular order): **[Liheng Hu](https://github.com/ckyasb)**, **[Hao Yi](https://github.com/eWloYW8)**, **[Tianyang Liu](https://github.com/jjsnam)**, **[Guanyi Zhou](https://github.com/Kwannyyy)**  

### 初赛

[初赛答辩材料](quals/slides/PAC26-1_quals.pdf)

#### 赛题一：Tensor Contraction

| 测试用例 | 基线耗时 | 基线吞吐率（TFLOPS） | 优化后耗时 | 优化后吞吐率（TFLOPS） | 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Case 1 | 17.6150 s | 0.227079 | 0.174445 s | 22.9313 | 100.98× |
| Case 2 | 354.085 s | 0.009037 | 0.141787 s | 22.5697 | 2497.48× |

#### 赛题二：Indexer MQA Logits

| 测试用例 | 基线耗时 | 基线吞吐率（TFLOPS） | 优化后耗时 | 优化后吞吐率（TFLOPS） | 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Case 1 | 26.964565 ms | 0.039781 | 58.532379 μs | 18.326338 | 460.68× |
| Case 2 | 195.983984 ms | 0.043850 | 398.316911 μs | 21.576886 | 492.06× |

### 决赛

[决赛答辩材料](final/slides/PAC-1.pdf)

#### 赛题一：Tensor Contraction 3Way

| 测试用例 | 基线耗时 | 基线吞吐率（TFLOPS） | 优化后耗时 | 优化后吞吐率（TFLOPS） | 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Case 1 | 20.8991 s | 0.196 | 88.261 ms | 46.408 | 236.78× |
| Case 2 | 13.6636 s | 0.225 | 65.225 ms | 47.099 | 209.33× |

#### 赛题二：Indexer MQA Logits

| 测试用例 | 基线耗时 | 基线吞吐率（TFLOPS） | 优化后耗时 | 优化后吞吐率（TFLOPS） | 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Case 1 | 215.566 ms | 0.01868 | 0.122 ms | 33.068 | 1770.24× |
| Case 2 | 268.089 ms | 0.02403 | 0.187 ms | 34.460 | 1433.86× |

