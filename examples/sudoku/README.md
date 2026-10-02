# examples/sudoku：用 Winnow 的 typed decisions 批量解数独

这个示例把参考仓库 `jeff/example/` 里的数独求解器改造成 `examples/ask.py` 的形式：

* 模型调用只有 `typesafe_sdk.TypeSafeClient.system_one(model=..., state=..., questions=...)`；
* `state` 是棋盘文本，问题用 `choice` / `noul` 两种 primitive **动态构造**；
* 求解器默认走 **批量打分**：第一次需要模型时，把当前盘面里一批格子的问题
  一次性打包进一次 `system_one`，之后整棵搜索树复用这批分数；
* 约束传播、合法性校验、回溯搜索全部在本地 Python 里完成。

模型只负责给分支候选排序（必要时剪枝），**不是最终裁判**：剪枝剪错了会自动
关掉剪枝重搜，返回前还会做完整的数独合法性校验。

## 为什么是「批量」而不是「并发请求」

服务端一次只处理一个 decision 请求（多个 HTTP 请求只会排队），真正的并行粒度
在请求内部：一次请求里的问题按 `--decision-parallel` 个分支分波并行评估（见
`docs/API.md` 的 Chat and concurrency）。所以把 N 个问题拆成 N 个请求只会串行
排队；把一批问题塞进**一次** `system_one` 才能吃满并行度，也省掉每个请求都
重复预处理的 state 和题面前缀。

`sudoku_solver.py` 的默认策略 `score_strategy="batch"`：

1. 约束传播先跑；只有传播卡住、需要给候选排序时才问模型；
2. 第一次把当前盘面里候选数 ≤ 当前最小候选数 + `--batch-slack`（默认 2）的格子
   打包成一次请求。`SUDOKU_RUBRIC` 放在每个问题最前面，服务端会把同一请求里
   所有问题的公共前缀只预处理一次，所以后面每个问题只多付「格子 + 候选」那一小段；
3. 分数在整棵搜索树复用（同一格的候选集合只会越传播越小）；后面遇到没打过分的
   格子再补一批。hard 盘面通常 1 批 27 问，evil 通常 1 批 45 问。

`--strategy node` 保留旧的「每个分支点问一格」行为，方便对比。

## 文件

| 文件 | 作用 |
| --- | --- |
| `sudoku_solver.py` | 纯约束求解核心。不依赖 SDK、不联网；打分器是鸭子类型，批量策略开箱即用，可用打桩 scorer 离线测试（`SudokuSolver(None).solve(board)` 就是纯约束基线） |
| `ask.py` | ask.py 风格入口：`TypeSafeClient` 接线、typed questions 构造、命令行 |
| `README.md` | 本文件 |

## 快速开始

先启动服务（默认 `127.0.0.1:8091`，见仓库 README 的 Run 一节）：

```sh
./run.sh
# 或
python3 scripts/serve.py --model ../Winnow-12B/gguf/Winnow-12B-Q8_0.gguf \
  --mmproj ../Winnow-12B/gguf/mmproj-Winnow-12B.gguf \
  --context 65536 --decision-parallel 4 --chat-parallel 1 --cache q8_0 --memory exclusive
```

然后在仓库根目录运行：

```sh
# 默认 medium：约束传播直接解出，0 次模型调用（适合冒烟测试）
python3 examples/sudoku/ask.py

# hard/evil 必须搜索：默认一次批量请求覆盖一批格子，整棵搜索树复用
python3 examples/sudoku/ask.py hard
python3 examples/sudoku/ask.py evil --quiet

# 和旧策略对比：每个分支点单独问一次
python3 examples/sudoku/ask.py hard --strategy node

# 每个候选值单独问一次 P(yes)（问题数更多，仍是一次批量请求）
python3 examples/sudoku/ask.py evil --mode noul --noul-normalize

# 打印第一轮批量请求的 state/questions/answers（长输出只展示前 3 个问题）
python3 examples/sudoku/ask.py hard --explain --max-nodes 3

# 从文件读盘面（81 个数字或 '.'）
python3 examples/sudoku/ask.py --puzzle-file grid.txt
```

也可以 `cd examples/sudoku && python3 ask.py hard`，或 `python3 -m examples.sudoku.ask hard`。

地址/密钥/模型可以用环境变量覆盖：`WINNOW_API`、`WINNOW_API_KEY`、
`WINNOW_MODEL`、`WINNOW_TIMEOUT`，或命令行 `--server/--api-key/--model/--timeout`。
本地服务不校验密钥（`aaa` 只是让 SDK 有值可用）。

`--help` 里还有 `--strategy`、`--batch-slack`、`--threshold`、`--no-prune`、
`--no-fallback`、`--parallel`、`--max-nodes` 等开关。

## 模型被问了什么

一次批量请求里所有问题共用同一个 state 和同一个题面前缀。`--explain` 的真实输出
（只展示前 3 个问题）：

```
[explain] state:
4.....8.5
.3.......
...7.....
.2.....6.
....8.4..
.4..1....
...6.3.7.
5.32.1...
1.4......

[explain] questions = (27 个问题，只展示前 3 个)
{
  "R1C2": {
    "type": "choice",
    "instructions": "Sudoku puzzle. The grid is in the state: 9 lines for rows R1-R9 ...\n
      Cell R1C2 (row 1, column 2, 3x3 box 1). Candidates: 1, 6, 7, 9. Which digit is the value of that cell?",
    "criteria": {"1": null, "6": null, "7": null, "9": null}
  },
  "R1C4": { "... Cell R1C4 ... Candidates: 1, 3, 9 ..." },
  "R1C5": { "... Cell R1C5 ... Candidates: 2, 3, 6, 9 ..." }
  // ... 其余 24 个问题结构相同，只是格子和候选不同
}
[explain] answers:
  R1C2: choice='6' confidence=0.0079 [1=0.2029 6=0.3012 7=0.2657 9=0.2303]
  R1C4: choice='9' confidence=0.0120 [1=0.2702 3=0.3272 9=0.4027]
  R1C5: choice='6' confidence=0.0140 [2=0.1832 3=0.2239 6=0.2991 9=0.2938]
```

`noul` 模式：每个 `(格子, 候选值)` 一个问题，返回独立的 `P(yes)`（不是分布，
除非加 `--noul-normalize`）。criteria 键只能是 `true` / `false`，这是 Winnow
服务端协议规定的；同样走批量请求。

## 设计要点

1. **本地先行**：裸单、隐性唯一、"某数字在本单位已无处可放"等约束传播先把能
   确定的格子填掉；只有剩余候选 ≥ 2 的格子才问模型（最小剩余值选格）。
2. **批量提问，全树复用**：第一次需要打分时把「候选数最少的那批格子」一次问完，
   分数在整棵搜索树里复用；搜索后面遇到更宽的格子再补一批。hard 从 22 次串行
   请求 / 12k token 降到 1 次请求 / 5.5k token，evil 从 60 次 / 33k token 降到
   1 次 / 9k token。
3. **题面公共前缀**：`SUDOKU_RUBRIC` 放在每个问题最前面，服务端只预处理一次
   公共前缀；criteria 用数字本身作选项（null 描述），state 用紧凑 9 行棋盘，
   每个问题只剩短尾巴。
4. **剪枝是"有则剪、无则全试"**：只有存在达到 `--threshold` 的候选时才剪掉低分
   候选；一个都没有就按分数顺序全部尝试，避免模型整体不自信就把有解盘面判死。
5. **两遍搜索兜底**：剪枝那一遍如果没解出来，自动关掉剪枝再跑一遍完整搜索；
   已经打过分的结果会命中缓存，第二遍通常不额外问模型。返回前校验 81 格、
   行/列/宫 1-9 各一次。
6. **模型失败可生存**：服务掉线、返回缺字段时退化为均匀分布并计数，求解继续；
   整批失败只记一次错误，不会对同一批格子反复重试。
7. **请求内并行，而不是并发请求**：求解器的 `--parallel` 只是进程内并行探索分支，
   真正的并行是「一次请求很多问题 + 服务端 `--decision-parallel`」，见上一节。

## 与参考实现（`jeff/example`）的差异

| | 参考实现 | 这里 |
| --- | --- | --- |
| 传输 | `HttpJeffClient` 直接 POST `/v1/systemone` | `typesafe_sdk.TypeSafeClient.system_one`（`examples/ask.py` 风格） |
| 问题类型 | 自定义 `ChoiceQuestion` / `NoulQuestion` 对象 | 普通 dict：`{"type": "choice"/"noul", "instructions": ..., "criteria": ...}` |
| noul criteria 键 | `yes` / `no` | `true` / `false`（Winnow 只接受这两个键） |
| 批量策略 | 一次请求打包待打分格子 | 默认同样批量，但按候选宽度懒加载（先问最窄的一批，后续补问），并复用整棵搜索树 |
| 题面 | 逐格长描述 | 公共 rubric + 短尾巴，让服务端共享前缀；仍可用 `--strategy node` 逐点问 |
| 求解逻辑 | 几何修正、剪枝回退、错误容忍、解校验 | 原样保留（含 `peers()` 优先级回归和两遍搜索） |

## 实测

本机当前服务：4×NVIDIA A2、Winnow-12B-Q8、`--decision-parallel 4`（绝对时间和
5070 Ti 不同，但下面所有对比都在同一台机器、同一个服务上）。`batch` / `node`
两行来自同一份代码，只是 `--strategy` 不同；"改动前的原始实现"是旧代码
（verbose 问题 + 每分支点一问），也在同一台机器上复测：

| 盘面 / 策略 | system_one 请求 | 问题数 | 分支点 | 墙钟 | 结果 |
| --- | --- | --- | --- | --- | --- |
| easy、medium / batch | 0 | 0 | 0 | ~0.00s | 约束传播直接解出 |
| hard / node（逐点策略） | 41 | 41 | 43 | 29.2s | 解出，校验通过 |
| **hard / batch（默认）** | **1** | **27** | 43 | **5.5s** | 解出，校验通过 |
| evil / node（逐点策略） | 69 | 69 | 79 | 50.3s | 解出，校验通过 |
| **evil / batch（默认）** | **1** | **45** | 96 | **8.9s** | 解出，校验通过 |
| hard / batch + noul | 1 | 93 | 52 | 12.6s | 解出，校验通过 |
| hard / 改动前的原始实现 | 22 | 22 | 22 | 26.4s | 解出，校验通过 |
| evil / 改动前的原始实现 | 60 | 60 | 72 | 72.9s | 解出，校验通过 |

**诚实说明**：Winnow-12B 不是数独模型。本机实测在 hard 盘面的分支点上，它对
2 选 1 的概率只有 ~0.51–0.57，接近随机；`noul` 模式的区分度也不更好；换一套
prompt 就会改变排序，分支点数量随之波动（上表 hard 的 43 和 22 就是同一盘面、
不同提问格式的结果）。所以这个示例展示的是 **typed decisions API 的批量接线
方式**和"模型只做启发式、程序保证正确性"的架构：省下的是模型往返次数和 token，
不是"模型把数独变简单了"。真正保证解正确的是本地约束传播 + 完整搜索兜底。

## 已知局限

* `batch` 策略会问一些搜索最终没用到的格子（换请求内并行），`--batch-slack`
  调小可以少问但会多发请求（`--batch-slack 0` 在 hard 上变成 6 批请求），
  需要按服务端 prefill 速度取舍。
* `noul` 模式每个候选一次提问，一格最多 9 问，比 `choice` 慢（hard 93 问 vs
  27 问）。
* 服务端限制：每次请求 1–256 个问题、每个问题 2–64 个选项；示例已按问题数分片。
* 服务端一次只处理一个 decision 请求，`--parallel` 并发多个请求不会提高吞吐；
  真正的并行在批量请求内部。
* 搜索节点数、请求数都受模型延迟支配，`--max-nodes` 可以在演示/调试时限制预算。
