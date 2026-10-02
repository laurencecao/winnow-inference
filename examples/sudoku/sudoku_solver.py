"""数独求解器：本地约束传播 + 外部打分器批量给候选排序（+ 可选并行分支）。

分工
----
* **约束传播**完全在本地做：裸单（候选数只剩一个）、隐性唯一（某数字在本行/
  列/宫里只剩一个可放的位置）、以及“某数字在本单位已无处可放 → 矛盾”。
* **打分器**（本示例里是 Winnow，接线见 ``ask.py``）只在无法继续传播时打分，
  拿到的概率只用来决定尝试顺序，必要时剪枝。默认走 **批量策略**：

  1. 第一次需要打分时，把当前盘面里候选数最少的一批格子（候选数
     ``<= 当前格子候选数 + batch_slack``）打包成 **一次** 请求；
  2. 这批分数在整棵搜索树里复用（同一格的候选只会变少，不会变多）；
  3. 搜索后面遇到没打过分的格子时再补一次批量请求。

  相比“每个分支点单独问一次”，这把 N 次串行模型调用压缩成 1～2 次批量调用，
  直接吃满服务端 ``--decision-parallel`` 的请求内并行；``score_strategy="node"``
  保留旧的逐分支点打分行为，方便对比。

* **程序化合法性检查是最终裁判**：关掉剪枝时，模型打分再离谱也不会让有解
  盘面求解失败，最多是搜索慢一点。

本模块只依赖标准库：打分器是鸭子类型，只要有

    score_one_cell(board, cell, values) -> {value: probability}

即可（可选实现 ``score_candidates(board, {cell: values}) -> {(cell, value): p}``
来支持批量打分，没有就退化成逐格调用）；``scorer=None`` 就退化成纯约束求解器，
可以作为基线。与 Winnow 的接线、typed questions 的构造都在 ``ask.py`` 里，
方便离线测试求解逻辑。

刻意保留的几个设计点
--------------------
1. ``peers()`` 由 27 个单位统一推导：``frozenset(含该格的 3 个单位) - {格}``，
   保证 peer 里不含自己（原始版本的 ``row | col | box - {cell}`` 运算符优先级
   有误，格子把自己也算了进去，落子后 ``is_legal()`` 恒为 False，搜索永远失败）。
2. 剪枝是“有则剪、无则全试”：只有当**至少一个候选**达到阈值时才剪掉低分
   候选；否则按分数顺序全部尝试，避免模型整体不自信就把有解盘面判死。
3. 剪枝那一遍失败后，``fallback_complete=True`` 会自动关闭剪枝再跑一遍完整
   搜索（第二遍通常直接命中打分缓存），保证“有解必得”。
4. 打分失败（服务掉线、返回缺字段）不抛异常打断求解，而是退化为均匀分布
   并计入 ``scorer_errors``；批量请求失败时整批退化为均匀分布，不重复重试
   同一批格子。
5. 返回前校验解：81 格、行/列/宫 1-9 各一次，避免返回“没有 0 但有冲突”的盘面。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass

# 批量打分时，除当前最小候选数外再多包含几档宽度的格子。
DEFAULT_BATCH_SLACK = 2

ALL_DIGITS: tuple[int, ...] = tuple(range(1, 10))

# ---------------------------------------------------------------------------
# 几何：27 个单位（9 行 + 9 列 + 9 宫），peers 由单位导出
# ---------------------------------------------------------------------------

_ROW_UNITS: tuple[tuple[int, ...], ...] = tuple(tuple(range(r * 9, r * 9 + 9)) for r in range(9))
_COL_UNITS: tuple[tuple[int, ...], ...] = tuple(
    tuple(c + 9 * i for i in range(9)) for c in range(9)
)
_BOX_UNITS: tuple[tuple[int, ...], ...] = tuple(
    tuple((br * 3 + i) * 9 + (bc * 3 + j) for i in range(3) for j in range(3))
    for br in range(3)
    for bc in range(3)
)
UNITS: tuple[tuple[int, ...], ...] = _ROW_UNITS + _COL_UNITS + _BOX_UNITS

CELL_UNITS: tuple[tuple[tuple[int, ...], ...], ...] = tuple(
    tuple(unit for unit in UNITS if cell in unit) for cell in range(81)
)
PEERS: tuple[frozenset[int], ...] = tuple(
    frozenset().union(*[set(unit) for unit in CELL_UNITS[cell]]) - {cell} for cell in range(81)
)
assert all(len(p) == 20 and cell not in p for cell, p in enumerate(PEERS)), "peer geometry is wrong"


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def get_row(cell: int) -> list[int]:
    return list(_ROW_UNITS[cell // 9])


def get_col(cell: int) -> list[int]:
    return list(_COL_UNITS[cell % 9])


def get_box(cell: int) -> list[int]:
    r, c = divmod(cell, 9)
    return list(_BOX_UNITS[(r // 3) * 3 + (c // 3)])


def peers(cell: int) -> frozenset[int]:
    """同格所在行/列/宫的其他 20 个格子（**不含自己**）。"""
    return PEERS[cell]


def is_solved(board: Sequence[int]) -> bool:
    return all(v != 0 for v in board)


def is_consistent(board: Sequence[int]) -> bool:
    """检查已填数字是否有冲突。"""
    for cell in range(81):
        v = board[cell]
        if v == 0:
            continue
        for p in PEERS[cell]:
            if board[p] == v:
                return False
    return True


def is_legal(board: Sequence[int], cell: int, value: int) -> bool:
    """把 value 放进 cell 是否违反行/列/宫约束（cell 已填别的值时也为 False）。"""
    if value not in ALL_DIGITS:
        return False
    if board[cell] not in (0, value):
        return False
    return all(board[p] != value for p in PEERS[cell])


def validate_solution(board: Sequence[int]) -> bool:
    """完整解校验：81 格填满、无冲突、每个单位的 1-9 恰好各一次。"""
    if len(board) != 81 or not is_solved(board):
        return False
    if not is_consistent(board):
        return False
    return all(sorted(board[c] for c in unit) == list(ALL_DIGITS) for unit in UNITS)


def empty_cells(board: Sequence[int]) -> list[int]:
    return [cell for cell in range(81) if board[cell] == 0]


# ---------------------------------------------------------------------------
# 约束传播
# ---------------------------------------------------------------------------


def compute_candidates(board: Sequence[int]) -> dict[int, list[int]]:
    """为每个空格计算合法候选值（程序化，不问模型）。"""
    candidates: dict[int, list[int]] = {}
    for cell in range(81):
        if board[cell] != 0:
            continue
        used = {board[p] for p in PEERS[cell] if board[p] != 0}
        candidates[cell] = [v for v in ALL_DIGITS if v not in used]
    return candidates


def fill_naked_singles(
    board: Sequence[int], candidates: Mapping[int, Sequence[int]] | None = None
) -> list[int]:
    """把候选数 == 1 的格子填上，循环到没有新的裸单为止。

    这是个便捷函数：**不做矛盾检测**，遇到无解盘面可能返回内部不一致的结果；
    求解器内部用的是 :meth:`SudokuSolver._propagate`。
    """
    board = list(board)
    cand: Mapping[int, Sequence[int]] | None = candidates
    while True:
        if cand is None:
            cand = compute_candidates(board)
        singles = {cell: vals[0] for cell, vals in cand.items() if len(vals) == 1}
        if not singles:
            return board
        for cell, v in singles.items():
            board[cell] = v
        cand = None


def find_hidden_singles(
    board: Sequence[int], candidates: Mapping[int, Sequence[int]]
) -> dict[int, int]:
    """隐性唯一：某数字在某行/列/宫中只有一个格子可填，返回 ``{cell: value}``。

    同一格被两个数字同时独占属于矛盾，这种条目会被丢掉而不是随意覆盖
    （求解器内部的 ``_propagate`` 会把这种情况判为死分支）。
    """
    fills: dict[int, int] = {}
    conflicted: set[int] = set()
    for unit in UNITS:
        present = {board[c] for c in unit}
        for v in ALL_DIGITS:
            if v in present:
                continue
            spots = [c for c in unit if board[c] == 0 and v in candidates.get(c, ())]
            if len(spots) == 1:
                cell = spots[0]
                if cell in fills and fills[cell] != v:
                    conflicted.add(cell)
                else:
                    fills[cell] = v
    for cell in conflicted:
        fills.pop(cell, None)
    return fills


def dead_units(
    board: Sequence[int], candidates: Mapping[int, Sequence[int]]
) -> list[tuple[int, int]]:
    """返回 ``[(unit_index, digit), ...]``：该数字在该单位里已无处可放。"""
    dead: list[tuple[int, int]] = []
    for i, unit in enumerate(UNITS):
        present = {board[c] for c in unit}
        for v in ALL_DIGITS:
            if v in present:
                continue
            if not any(board[c] == 0 and v in candidates.get(c, ()) for c in unit):
                dead.append((i, v))
    return dead


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------


class SearchLimitExceeded(RuntimeError):
    """超过 ``max_nodes`` 的节点预算。"""


@dataclass
class SolveStats:
    nodes: int = 0
    branches: int = 0
    propagation_fills: int = 0
    backtracks: int = 0
    scorer_calls: int = 0
    scorer_questions: int = 0
    scorer_batches: int = 0  # 求解器发起的批量打分请求次数（batch 策略下通常 1~2）
    scorer_errors: int = 0
    max_depth: int = 0
    elapsed_s: float = 0.0
    passes: int = 0
    prune_fallback: bool = False  # 第一遍剪枝无解，靠关闭剪枝的第二遍找回

    def summary(self) -> str:
        extra = f" passes={self.passes}" + ("(剪枝后退回完整搜索)" if self.prune_fallback else "")
        return (
            f"nodes={self.nodes} branches={self.branches} "
            f"propagation_fills={self.propagation_fills} backtracks={self.backtracks} "
            f"scorer_calls={self.scorer_calls} scorer_questions={self.scorer_questions} "
            f"scorer_batches={self.scorer_batches} "
            f"scorer_errors={self.scorer_errors} depth={self.max_depth}{extra} "
            f"elapsed={self.elapsed_s:.2f}s"
        )


# ---------------------------------------------------------------------------
# 核心求解
# ---------------------------------------------------------------------------


class SudokuSolver:
    """约束传播 + 外部打分器排序/剪枝 + 回溯搜索。

    Args:
        scorer: 形如 ``WinnowSudokuScorer`` 的对象。实现
            ``score_one_cell(board, cell, values) -> {value: p}`` 即可；
            额外实现 ``score_candidates(board, {cell: values}) -> {(cell, value): p}``
            就能让 batch 策略一次请求打包多格。``None`` 表示不使用模型，
            退化成纯约束求解器。
        threshold: 剪枝阈值，在 **scorer 返回的原始尺度** 上比较。``None`` 表示
            不做任何剪枝（只排序）。
        prune: 是否按阈值剪枝。只有在**至少一个候选 ≥ threshold** 时才真正
            剪掉低分候选；若没有任何候选达到阈值，则全部按分数顺序尝试
            （保证不会因为模型整体不自信而把一个有解盘面判死）。
        fallback_complete: 剪枝那一遍如果没解出来，自动**关闭剪枝再跑一遍**
            （完整搜索）。第二遍对已经打过分格子的直接命中缓存，所以通常不会
            多花模型调用；这是“用模型加速但最终仍保证有解必得”的关键。
        score_strategy: ``"batch"``（默认）第一次需要打分时用一次批量请求把
            当前候选数最少的一批格子一起问掉，之后整棵搜索树复用这批分数；
            ``"node"`` 恢复旧的“每个分支点只问当前一格”的行为。
        batch_slack: batch 策略的批量范围：包含候选数不超过
            ``当前格子候选数 + batch_slack`` 的所有未打合格子。调大=多问一点、
            少发请求；调小=每次问得少、但可能要补发多次请求。
        parallel: 是否用线程并行探索同一分支点的候选。默认关闭：真正省时间
            的并行在服务端请求内部（一次 system_one 装很多问题，按
            ``--decision-parallel`` 分波），而不是并发多个请求（服务端一次只
            处理一个 decision 请求）。保留此开关是为了兼容旧行为。
        max_workers: 并行分支的最大线程数。
        max_nodes: 节点预算（含两遍搜索），0 表示不限；超过抛
            :class:`SearchLimitExceeded`。
        log: 日志回调（默认丢弃）。传 ``print`` 即可看到分支过程。
    """

    def __init__(
        self,
        scorer: object | None = None,
        *,
        threshold: float | None = 0.6,
        prune: bool = True,
        fallback_complete: bool = True,
        score_strategy: str = "batch",
        batch_slack: int = DEFAULT_BATCH_SLACK,
        parallel: bool = False,
        max_workers: int | None = None,
        max_nodes: int = 0,
        log: Callable[[str], None] | None = None,
    ) -> None:
        if score_strategy not in ("batch", "node"):
            raise ValueError(f"score_strategy must be 'batch' or 'node', got {score_strategy!r}")
        if batch_slack < 0:
            raise ValueError(f"batch_slack must be >= 0, got {batch_slack}")
        self.scorer = scorer
        self.threshold = threshold
        self.prune = prune and threshold is not None
        self.fallback_complete = fallback_complete
        self.score_strategy = score_strategy
        self.batch_slack = batch_slack
        self.parallel = parallel
        self.max_workers = max_workers
        self.max_nodes = max_nodes
        self._log_fn = log
        self.stats = SolveStats()
        # batch 策略：cell -> {value: p}，整棵搜索树共享（候选只会越传播越少）。
        self._cell_probs: dict[int, dict[int, float]] = {}
        # parallel=True 时多个分支线程可能同时遇到没打分的格子；串行化批量请求。
        self._score_lock = threading.Lock()

    # -- public API ---------------------------------------------------------

    def solve(self, board: Sequence[int]) -> list[int] | None:
        """返回解出的棋盘，或 None（无解）。

        Raises:
            ValueError: 输入不是 81 格。
            SearchLimitExceeded: 超过 ``max_nodes``。
        """
        board = list(board)
        if len(board) != 81:
            raise ValueError(f"board must have 81 cells, got {len(board)}")

        self.stats = SolveStats()
        self._cell_probs = {}
        start = time.perf_counter()
        result: list[int] | None = None
        try:
            if is_consistent(board) and all(v in range(10) for v in board):
                self.stats.passes = 1
                result = self._search(board, depth=0)
                if result is None and self.prune and self.fallback_complete:
                    # 剪枝可能因为“模型自信且错”而丢掉正确分支；关掉剪枝再来一遍。
                    # 这一遍通常不产生新的模型调用（打分结果已在缓存里）。
                    self.stats.passes = 2
                    self.stats.prune_fallback = True
                    self._log("剪枝后无解：关闭剪枝重新搜索（完整搜索，缓存命中打分）")
                    saved, self.prune = self.prune, False
                    try:
                        result = self._search(board, depth=0)
                    finally:
                        self.prune = saved
        finally:
            # 即使搜索被 max_nodes 预算打断，也要把真实的模型调用数带出去。
            self.stats.elapsed_s = time.perf_counter() - start
            scorer_stats = getattr(self.scorer, "stats", None)
            self.stats.scorer_calls = getattr(scorer_stats, "calls", 0)
            self.stats.scorer_questions = getattr(scorer_stats, "questions", 0)
            # 注意：不覆盖 scorer_errors —— 那是 _score() 在异常现场自己数的，
            # 打分器的 stats 未必有 errors 字段（打桩 scorer 就没有）。

        if result is not None and not validate_solution(result):
            raise AssertionError("internal error: solver returned an invalid grid")
        return result

    # -- 搜索 ---------------------------------------------------------------

    def _search(self, board: Sequence[int], depth: int) -> list[int] | None:
        self.stats.nodes += 1
        self.stats.max_depth = max(self.stats.max_depth, depth)
        if self.max_nodes and self.stats.nodes > self.max_nodes:
            raise SearchLimitExceeded(f"node budget exhausted ({self.max_nodes})")

        board = self._propagate(list(board))
        if board is None:
            return None

        candidates = compute_candidates(board)
        if not candidates:
            return board  # 已解出

        cell = min(candidates, key=lambda c: (len(candidates[c]), c))
        values = candidates[cell]
        if len(values) == 1:  # 理论上 _propagate 已经处理，保险
            board[cell] = values[0]
            return self._search(board, depth)

        self.stats.branches += 1
        ordered = self._order(board, cell, values, depth, candidates)
        if self.parallel and len(ordered) > 1:
            return self._branch_parallel(board, cell, ordered, depth)

        indent = "  " * depth
        for v in ordered:
            if not is_legal(board, cell, v):  # 候选来自 compute_candidates，正常不会发生
                continue
            self._log(f"{indent}分支 R{cell // 9 + 1}C{cell % 9 + 1} = {v}")
            child = list(board)
            child[cell] = v
            result = self._search(child, depth + 1)
            if result is not None:
                return result
            self.stats.backtracks += 1
            self._log(f"{indent}死分支 R{cell // 9 + 1}C{cell % 9 + 1} = {v}")
        return None

    def _branch_parallel(
        self, board: Sequence[int], cell: int, ordered: Sequence[int], depth: int
    ) -> list[int] | None:
        workers = self.max_workers or len(ordered)
        executor = ThreadPoolExecutor(
            max_workers=max(1, min(len(ordered), workers)), thread_name_prefix="sudoku"
        )
        try:
            futures = {}
            for v in ordered:
                child = list(board)
                child[cell] = v
                futures[executor.submit(self._search, child, depth + 1)] = v
            pending = set(futures)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    if result is not None:
                        self._log(f"{'  ' * depth}并行分支 R{cell // 9 + 1}C{cell % 9 + 1} 命中")
                        return result
            return None
        finally:
            # 命中的分支已经拿到解，剩下的分支不再等待（正在跑的那几个会自然结束）
            executor.shutdown(wait=False, cancel_futures=True)

    def _order(
        self,
        board: Sequence[int],
        cell: int,
        values: Sequence[int],
        depth: int,
        candidates: Mapping[int, Sequence[int]] | None = None,
    ) -> list[int]:
        """用打分器给候选排序（必要时剪枝）。"""
        probs = self._score(board, cell, values, depth, candidates)
        ordered = sorted(values, key=lambda v: (-probs[v], v))

        indent = "  " * depth
        pretty = " ".join(f"{v}:{probs[v]:.2f}" for v in ordered)
        self._log(
            f"{indent}分支点 R{cell // 9 + 1}C{cell % 9 + 1} 候选 {sorted(values)} -> {pretty}"
        )

        if not self.prune or self.threshold is None:
            return ordered

        confident = [v for v in ordered if probs[v] >= self.threshold]
        if not confident:
            # 关键修复：没有候选达到阈值时**不能**把候选全剪掉，否则有解盘面会被判死
            self._log(f"{indent}没有任何候选 ≥ {self.threshold}，按分数顺序全部尝试")
            return ordered
        pruned = [v for v in ordered if probs[v] < self.threshold]
        if pruned:
            self._log(
                f"{indent}剪枝 (< {self.threshold}): "
                + ", ".join(f"{v}({probs[v]:.2f})" for v in pruned)
            )
        return confident

    def _score(
        self,
        board: Sequence[int],
        cell: int,
        values: Sequence[int],
        depth: int,
        candidates: Mapping[int, Sequence[int]] | None = None,
    ) -> dict[int, float]:
        uniform = {v: 1.0 / len(values) for v in values}
        if self.scorer is None or len(values) <= 1:
            return {v: 1.0 for v in values} if len(values) == 1 else uniform
        if self.score_strategy == "batch":
            return self._score_batched(board, cell, values, candidates, depth)
        try:
            raw = self.scorer.score_one_cell(board, cell, values)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - 模型/网络问题不应该让求解崩溃
            self.stats.scorer_errors += 1
            self._log(f"{'  ' * depth}打分失败（改用均匀分布）：{exc}")
            return uniform
        probs = {v: float(raw.get(v, 0.0)) for v in values}
        if sum(probs.values()) <= 0.0:
            return uniform
        # 不在这里重新归一化：量纲由 scorer 决定（choice 是分布，noul 是逐个
        # 独立问题的原始 P(yes)，两者都不该被求解器偷偷改成别的含义）。
        # 阈值比较因此是在 scorer 给出的尺度上进行的，见 _order。
        return probs

    def _score_batched(
        self,
        board: Sequence[int],
        cell: int,
        values: Sequence[int],
        candidates: Mapping[int, Sequence[int]] | None,
        depth: int,
    ) -> dict[int, float]:
        """批量策略：一次请求问掉当前一批格子，之后整棵搜索树复用。

        第一批只覆盖候选数不超过 ``当前候选数 + batch_slack`` 的格子；搜索
        后面如果遇到更宽、还没打分的格子，再补一批。候选集合在传播中只会
        变小，所以旧分数对同一格始终可用（取剩余候选里分数最高的）。
        """
        with self._score_lock:
            if cell not in self._cell_probs:
                pool = candidates if candidates is not None else {cell: values}
                limit = len(values) + self.batch_slack
                pending = {
                    c: list(vals)
                    for c, vals in pool.items()
                    if len(vals) > 1 and c not in self._cell_probs and len(vals) <= limit
                }
                pending.setdefault(cell, list(values))
                self.stats.scorer_batches += 1
                try:
                    batch = self._ask_batch(board, pending)
                except Exception as exc:  # noqa: BLE001 - 模型/网络问题不应该让求解崩溃
                    self.stats.scorer_errors += 1
                    self._log(
                        f"{'  ' * depth}批量打分失败（{len(pending)} 格改用均匀分布）：{exc}"
                    )
                    batch = {}
                self._log(
                    f"{'  ' * depth}批量打分：一次请求 {len(pending)} 格"
                    f"（候选数 ≤ {limit}）"
                )
                for c in pending:
                    # 没拿到分数的格子也记成空 dict，避免对同一格反复重试。
                    self._cell_probs.setdefault(c, dict(batch.get(c, {})))

        raw = self._cell_probs.get(cell, {})
        probs = {v: float(raw.get(v, 0.0)) for v in values}
        if sum(probs.values()) <= 0.0:
            return {v: 1.0 / len(values) for v in values}
        return probs

    def _ask_batch(
        self,
        board: Sequence[int],
        pending: Mapping[int, Sequence[int]],
    ) -> dict[int, dict[int, float]]:
        """把 ``{cell: values}`` 交给打分器；优先用批量接口，否则逐格调用。"""
        score_batch = getattr(self.scorer, "score_candidates", None)
        if callable(score_batch):
            raw = score_batch(board, pending)
            return {
                c: {v: float(raw.get((c, v), 0.0)) for v in vals} for c, vals in pending.items()
            }
        score_one = getattr(self.scorer, "score_one_cell", None)
        if not callable(score_one):
            raise TypeError("scorer must provide score_one_cell or score_candidates")
        return {c: dict(score_one(board, c, list(vals))) for c, vals in pending.items()}

    def _propagate(self, board: list[int]) -> list[int] | None:
        """约束传播到不动点；返回 None 表示死分支。"""
        while True:
            candidates = compute_candidates(board)
            if not candidates:
                return board  # 解出
            if any(not vals for vals in candidates.values()):
                return None  # 有格子没有候选

            fills: dict[int, int] = {}
            for cell, vals in candidates.items():
                if len(vals) == 1:
                    fills[cell] = vals[0]

            # 隐性唯一 + “数字无处可放 → 矛盾”
            for unit in UNITS:
                present = {board[c] for c in unit}
                for v in ALL_DIGITS:
                    if v in present:
                        continue
                    spots = [c for c in unit if board[c] == 0 and v in candidates[c]]
                    if not spots:
                        return None
                    if len(spots) == 1:
                        cell = spots[0]
                        if fills.get(cell, v) != v:
                            return None  # 同一格被两个数字独占
                        fills[cell] = v

            if not fills:
                return board

            for cell, v in fills.items():
                if board[cell] == v:
                    continue
                if not is_legal(board, cell, v):
                    return None
                board[cell] = v
                self.stats.propagation_fills += 1

    # -- 日志 ---------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self._log_fn is not None:
            self._log_fn(message)
