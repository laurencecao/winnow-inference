"""用 Winnow 的 typed decisions 批量给数独候选打分（``examples/ask.py`` 风格入口）。

和 ``examples/ask.py`` 一样，这里只做三件事：

1. 用 ``TypeSafeClient`` 连本地 Winnow 服务；
2. 构造 ``state``（棋盘文本）和一组带类型的问题：``choice`` 是“某格该填哪个
   数字”，``noul`` 是“某格是不是某个数字”；
3. 调 ``system_one``，从 ``r.choices[...] / r.nouls[...]`` 里读概率。

区别在于问题不是写死的，而且是**批量**的：求解器第一次需要打分时，会把当前
盘面里一批格子（候选数最少的那批）的问题一次性打包进 ``system_one``，之后
整棵搜索树复用这批分数；搜索后面遇到没打过分的格子再补一批。

为什么是批量而不是并发多个请求：服务端一次只处理一个 decision 请求，多个
HTTP 请求只会排队；真正的并行粒度在请求内部——服务端把一次请求里的问题按
``--decision-parallel`` 个分支分波并行评估。所以“一次问很多个问题”才能吃满
并行度，``SudokuSolver(score_strategy="batch")`` 就是干这个的。

约束传播和合法性校验都在本地完成，模型永远不是最终裁判：剪枝剪错了会自动
关掉剪枝重搜，见 ``sudoku_solver.py``。

用法（在仓库根目录）::

    python3 examples/sudoku/ask.py                    # medium 盘面，默认批量打分
    python3 examples/sudoku/ask.py hard               # 一次批量请求，整棵树复用
    python3 examples/sudoku/ask.py hard --strategy node   # 旧行为：每个分支点问一次
    python3 examples/sudoku/ask.py hard --mode noul   # 每个候选取一次 P(yes)
    python3 examples/sudoku/ask.py evil --quiet       # 不打印分支过程
    python3 examples/sudoku/ask.py medium --explain   # 打印第一轮批量请求
    python3 examples/sudoku/ask.py --puzzle-file grid.txt
    python3 examples/sudoku/ask.py --help

也可以 ``cd examples/sudoku && python3 ask.py``，或者 ``python3 -m
examples.sudoku.ask``。

先启动服务（默认 127.0.0.1:8091）::

    ./run.sh
    # 或见仓库 README 的 Run 一节

地址、密钥、模型、超时可用环境变量覆盖：``WINNOW_API``、``WINNOW_API_KEY``、
``WINNOW_MODEL``、``WINNOW_TIMEOUT``。本地服务不校验密钥，``aaa`` 只是让 SDK
开心。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from typesafe_sdk import RetryPolicy, TypeSafeClient

try:  # 包内运行：python3 -m examples.sudoku.ask
    from .sudoku_solver import (
        DEFAULT_BATCH_SLACK,
        SearchLimitExceeded,
        SudokuSolver,
        is_consistent,
        validate_solution,
    )
except ImportError:  # 脚本运行：python3 examples/sudoku/ask.py
    from sudoku_solver import (  # type: ignore[no-redef]
        DEFAULT_BATCH_SLACK,
        SearchLimitExceeded,
        SudokuSolver,
        is_consistent,
        validate_solution,
    )

MYAPI = os.environ.get("WINNOW_API", "http://localhost:8091")
MYKEY = os.environ.get("WINNOW_API_KEY", "aaa")
MODEL = os.environ.get("WINNOW_MODEL", "Winnow-12B")
TIMEOUT = float(os.environ.get("WINNOW_TIMEOUT", "900"))
os.environ["TYPESAFE_BASE_URL"] = MYAPI

# /v1/systemone 每次请求 1–256 个问题、每个问题 2–64 个选项；本示例每格最多
# 9 个候选，所以只需要按问题数分片。
MAX_QUESTIONS_PER_CALL = 256

# 所有问题共用的题面前缀。放在 instructions 最前面，服务端会把同一请求里
# 所有问题的公共前缀抽出来只预处理一次（见 native/planner.h 的 make_plan），
# 这样批量请求里每个问题只多付“格子 + 候选”那一小段 token。
SUDOKU_RUBRIC = (
    "Sudoku puzzle. The grid is in the state: 9 lines for rows R1-R9 top to bottom, "
    "each line 9 cells for columns C1-C9 left to right, '.' marks an empty cell. "
    "Every row, column and 3x3 box must contain the digits 1-9 exactly once. "
    "Each question names one empty cell and lists its candidate digits (the digits that "
    "conflict with no filled cell in its row, column or box); choose the candidate that "
    "can be extended to a complete valid grid."
)

# 0 表示空格，用来快速试内置盘面。
PUZZLES: dict[str, str] = {
    "easy": ("003020600900305001001806400008102900700000008006708200002609500800203009005010300"),
    "medium": ("530070000600195000098000060800060003400803001700020006060000280000419005000080079"),
    "hard": ("400000805030000000000700000020000060000080400000010000000603070500200000104000000"),
    "evil": ("100007090030020008009600500005300900010080002600004000300000010040000007007000300"),
}


# ---------------------------------------------------------------------------
# 棋盘 -> state / questions
# ---------------------------------------------------------------------------


def format_board(board: Sequence[int]) -> str:
    """把棋盘渲染成 9 行紧凑文本，作为 ``system_one`` 的 state。

    行号 R1-R9 从上到下、列号 C1-C9 从左到右，问题文本里的 row/column 与之一一
    对应。空格用 '.' 表示。这里刻意不用带表头/空格的多行格式：state 每次请求
    只预处理一次，但紧凑格式能让 ``--strategy node`` 这类逐点打分的模式也少花
    不少 token。
    """
    return "\n".join(
        "".join(str(v) if v else "." for v in board[r * 9 : (r + 1) * 9]) for r in range(9)
    )


def parse_board(text: str) -> list[int]:
    """把 81 个数字/'.'/'0' 的文本（可有空白或换行）解析成棋盘。"""
    board: list[int] = []
    for ch in text:
        if ch in ".0":
            board.append(0)
        elif ch.isdigit():
            v = int(ch)
            if not 1 <= v <= 9:
                raise ValueError(f"digit out of range: {ch}")
            board.append(v)
    if len(board) != 81:
        raise ValueError(f"expected 81 cells, got {len(board)}")
    return board


def render_board_chars(board: Sequence[int]) -> str:
    """紧凑的 81 字符表示。"""
    return "".join(str(v) if v else "." for v in board)


def cell_name(cell: int) -> str:
    """问题 id / 日志里的格子名，例如 R3C5。"""
    r, c = divmod(cell, 9)
    return f"R{r + 1}C{c + 1}"


def question_name(cell: int, value: int) -> str:
    """noul 模式的问题 id，例如 R3C5_7。"""
    return f"{cell_name(cell)}_{value}"


def _fmt_digits(values: Sequence[int]) -> str:
    return ", ".join(str(v) for v in values) if values else "(none)"


def build_choice_question(board: Sequence[int], cell: int, values: Sequence[int]) -> dict:
    """一个格子一个问题：候选数字是 criteria，返回模型在候选上的分布。

    ``SUDOKU_RUBRIC`` 必须放在最前面：同一请求里所有问题都以它开头，服务端
    只预处理一次公共前缀，批量请求因此便宜得多。格子和候选写在后面，保持
    “问题 id 不含模型指令、指令里点名格子”的约定。
    """
    r, c = divmod(cell, 9)
    box = (r // 3) * 3 + (c // 3) + 1
    instructions = (
        f"{SUDOKU_RUBRIC}\n"
        f"Cell {cell_name(cell)} (row {r + 1}, column {c + 1}, 3x3 box {box}). "
        f"Candidates: {_fmt_digits(values)}. Which digit is the value of that cell?"
    )
    criteria = {str(v): None for v in values}
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def build_noul_question(board: Sequence[int], cell: int, value: int) -> dict:
    """一个 (格子, 候选值) 一个问题，返回 P(yes)。

    注意 noul 的 criteria 键只能是 ``true`` / ``false``（服务端协议如此）。
    """
    r, c = divmod(cell, 9)
    box = (r // 3) * 3 + (c // 3) + 1
    instructions = (
        f"{SUDOKU_RUBRIC}\n"
        f"Cell {cell_name(cell)} (row {r + 1}, column {c + 1}, 3x3 box {box}): "
        f"is its value the digit {value}?"
    )
    criteria = {"true": None, "false": None}
    return {"type": "noul", "instructions": instructions, "criteria": criteria}


# ---------------------------------------------------------------------------
# 打分器：把 {cell: [候选值]} 打包成 system_one 请求
# ---------------------------------------------------------------------------


def _normalize(
    values: Sequence[int], probs: Mapping[int, float], force: bool = False
) -> dict[int, float]:
    """原样返回（默认）或归一化（``force=True``）；只有全 0 才兜底成均匀。

    归一化只是“把独立 yes/no 答案读成本格占比”的约定，不是对模型输出的修正：
    它不改变排序（除以同一个正数），只改变与固定阈值比较的含义。
    """
    vals = list(values)
    if not vals:
        return {}
    clean = {v: max(0.0, float(probs.get(v, 0.0))) for v in vals}
    total = sum(clean.values())
    if total <= 0.0:
        return {v: 1.0 / len(vals) for v in vals}
    if not force:
        return clean
    return {v: p / total for v, p in clean.items()}


@dataclass
class ScorerStats:
    calls: int = 0  # system_one 请求数（问题数超过 256 时会分片）
    questions: int = 0  # 提交的问题总数
    cache_hits: int = 0
    errors: int = 0
    input_tokens: int = 0
    error_messages: list[str] = field(default_factory=list)


class WinnowSudokuScorer:
    """把 ``{cell: [候选值]}`` 打包成 typed questions，交给 Winnow 打分。

    一次 ``score_candidates`` 调用会把传入的所有格子打包进尽量少的
    ``system_one`` 请求（服务端上限 256 问/请求，超出自动分片）；求解器的默认
    batch 策略正是靠它把整批格子一次问完，让服务端 ``--decision-parallel``
    的请求内并行有活干。

    mode 的含义
    -----------
    * ``"choice"``：一个格子一个问题，criteria 是该格的候选数字，返回
      ``P(该格的值 = v | 棋盘)``，同一格内和为 1（模型原始输出）。默认模式。
    * ``"noul"``：一个 (格子, 候选值) 一个问题，原样返回 ``P(yes)``。这些是
      彼此独立的问题，不是分布（``noul_normalize=True`` 时才归一化成占比）；
      问题数约为 choice 模式的「候选数总和 / 格子数」倍。

    单候选的格子不问模型，直接给 1.0；相同 ``(棋盘, 格子, 候选)`` 的结果会
    缓存，所以“剪枝失败后关剪枝重搜”那一遍通常不产生新的模型调用。
    """

    def __init__(
        self,
        client: TypeSafeClient,
        *,
        model: str = MODEL,
        mode: str = "choice",
        noul_normalize: bool = False,
        cache: bool = True,
        max_cache_entries: int = 512,
        explain: bool = False,
    ) -> None:
        if mode not in ("choice", "noul"):
            raise ValueError(f"mode must be 'choice' or 'noul', got {mode!r}")
        self.client = client
        self.model = model
        self.mode = mode
        self.noul_normalize = noul_normalize and mode == "noul"
        self.explain = explain
        self.stats = ScorerStats()
        self._cache: dict[tuple, dict[int, float]] | None = {} if cache else None
        self._max_cache_entries = max_cache_entries
        # 服务端一次只处理一个 decision 请求；这把锁把并发分支线程发起的调用
        # 串起来，顺便让 stats/缓存更新不打架。
        self._lock = threading.Lock()

    # -- public API ---------------------------------------------------------

    def score_candidates(
        self, board: Sequence[int], candidates: Mapping[int, Sequence[int]]
    ) -> dict[tuple[int, int], float]:
        """board: 长度 81，0 表示空格；candidates: ``{cell: [合法候选值]}``。

        返回 ``{(cell, value): 概率}``。问题数超过服务端上限时自动分片成多次
        ``system_one`` 调用。打分失败会抛异常（求解器决定是否退化为均匀分布）。
        """
        clean: dict[int, list[int]] = {}
        for cell, values in candidates.items():
            vals = sorted({int(v) for v in values})
            if vals:
                clean[int(cell)] = vals
        if not clean:
            return {}

        scores: dict[tuple[int, int], float] = {}
        pending: dict[int, list[int]] = {}
        for cell, vals in clean.items():
            if len(vals) == 1:
                scores[(cell, vals[0])] = 1.0
            else:
                pending[cell] = vals
        if not pending:
            return scores

        state = format_board(board)
        todo: dict[int, list[int]] = {}
        for cell, vals in pending.items():
            cached = self._cached(state, cell, vals)
            if cached is None:
                todo[cell] = vals
            else:
                self.stats.cache_hits += 1
                for v, p in cached.items():
                    scores[(cell, v)] = p

        if todo:
            per_cell = self._ask(state, board, todo)
            for cell, vals in todo.items():
                probs = _normalize(vals, per_cell.get(cell, {}), force=self.noul_normalize)
                for v, p in probs.items():
                    scores[(cell, v)] = p
                self._store(state, cell, vals, probs)
        return scores

    def score_one_cell(
        self, board: Sequence[int], cell: int, values: Sequence[int]
    ) -> dict[int, float]:
        """便捷入口：只问一个格子，返回 ``{value: 概率}``。"""
        out = self.score_candidates(board, {cell: values})
        return {v: p for (c, v), p in out.items() if c == cell}

    def close(self) -> None:
        self.client.close()

    # -- internals ----------------------------------------------------------

    def _cached(self, state: str, cell: int, values: Sequence[int]) -> dict[int, float] | None:
        if self._cache is None:
            return None
        return self._cache.get(self._cache_key(state, cell, values))

    def _store(self, state: str, cell: int, values: Sequence[int], probs: dict[int, float]) -> None:
        if self._cache is None:
            return
        if len(self._cache) >= self._max_cache_entries:
            self._cache.clear()
        self._cache[self._cache_key(state, cell, values)] = dict(probs)

    def _cache_key(self, state: str, cell: int, values: Sequence[int]) -> tuple:
        return (state, cell, tuple(values), self.mode, self.noul_normalize)

    def _system_one(self, state: str, questions: dict[str, dict]):
        """一次 ``system_one`` 调用；失败会抛 SDK 异常并计入 errors。"""
        first = False
        try:
            with self._lock:
                answers = self.client.system_one(
                    model=self.model, state=state, questions=questions
                )
                self.stats.calls += 1
                self.stats.questions += len(questions)
                usage = getattr(answers, "usage", None)
                self.stats.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                first = self.stats.calls == 1
        except Exception as exc:  # noqa: BLE001 - 交给求解器决定是否退化
            self.stats.errors += 1
            if len(self.stats.error_messages) < 3:
                self.stats.error_messages.append(f"{type(exc).__name__}: {exc}")
            raise
        if self.explain and first:
            self._print_explanation(state, questions, answers)
        return answers

    def _ask(
        self, state: str, board: Sequence[int], pending: Mapping[int, list[int]]
    ) -> dict[int, dict[int, float]]:
        """构造问题并发请求，返回 ``{cell: {value: 概率}}``。"""
        work: list[tuple[int, int | None, str]] = []
        for cell, values in pending.items():
            if self.mode == "choice":
                work.append((cell, None, cell_name(cell)))
            else:
                work.extend((cell, v, question_name(cell, v)) for v in values)

        per_cell: dict[int, dict[int, float]] = {}
        for start in range(0, len(work), MAX_QUESTIONS_PER_CALL):
            chunk = work[start : start + MAX_QUESTIONS_PER_CALL]
            questions: dict[str, dict] = {}
            plan: dict[str, tuple[int, int | None]] = {}
            for cell, value, qid in chunk:
                questions[qid] = (
                    build_choice_question(board, cell, pending[cell])
                    if value is None
                    else build_noul_question(board, cell, value)
                )
                plan[qid] = (cell, value)
            answers = self._system_one(state, questions)
            self._collect(answers, plan, pending, per_cell)
        return per_cell

    def _collect(self, answers, plan, pending, per_cell) -> None:
        if self.mode == "choice":
            for qid, (cell, _) in plan.items():
                ans = answers.choices.get(qid)
                if ans is None:
                    raise RuntimeError(f"missing choice answer for {qid!r}")
                probs = ans.probabilities or {}
                per_cell[cell] = {v: float(probs.get(str(v), 0.0)) for v in pending[cell]}
        else:
            for qid, (cell, value) in plan.items():
                ans = answers.nouls.get(qid)
                if ans is None:
                    raise RuntimeError(f"missing noul answer for {qid!r}")
                per_cell.setdefault(cell, {})[int(value)] = float(ans.noul)

    def _print_explanation(self, state: str, questions: dict[str, dict], answers) -> None:
        """``--explain``：把第一轮（批量）请求和回答按 ask.py 的风格打印出来。"""
        shown = 3
        head = dict(list(questions.items())[:shown])
        print("\n[explain] 第一次打分请求")
        print("[explain] state:")
        print(state)
        print(f"\n[explain] questions = ({len(questions)} 个问题，只展示前 {min(shown, len(questions))} 个)")
        print(json.dumps(head, ensure_ascii=False, indent=2))
        if len(questions) > shown:
            print(f"[explain] ... 其余 {len(questions) - shown} 个问题结构相同，只是格子和候选不同")
        print("[explain] answers:")
        for i, (qid, ans) in enumerate(answers.choices.items()):
            if i >= shown:
                print(f"  ... 其余 {len(answers.choices) - shown} 个 choice 答案省略")
                break
            probs = " ".join(f"{k}={p:.4f}" for k, p in ans.probabilities.items())
            print(f"  {qid}: choice={ans.choice!r} confidence={ans.confidence:.4f} [{probs}]")
        for i, (qid, ans) in enumerate(answers.nouls.items()):
            if i >= shown:
                print(f"  ... 其余 {len(answers.nouls) - shown} 个 noul 答案省略")
                break
            print(f"  {qid}: noul={ans.noul:.4f}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def print_board(board: Sequence[int], title: str | None = None) -> None:
    if title:
        print(title)
    print("    +-------+-------+-------+")
    for r in range(9):
        row = board[r * 9 : (r + 1) * 9]
        cells = " ".join(str(v) if v else "." for v in row)
        print(f" {r + 1}  | {cells[0:5]} | {cells[6:11]} | {cells[12:17]} |")
        if r % 3 == 2:
            print("    +-------+-------+-------+")
    print("      1 2 3   4 5 6   7 8 9")


def load_board(args: argparse.Namespace) -> list[int]:
    if args.puzzle_file:
        return parse_board(Path(args.puzzle_file).read_text(encoding="utf-8"))
    return parse_board(PUZZLES[args.puzzle])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="数独求解器：本地约束传播 + Winnow 给候选打分（typed decisions API）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "puzzle", nargs="?", default="medium", choices=sorted(PUZZLES), help="内置盘面"
    )
    parser.add_argument("--puzzle-file", metavar="FILE", help="从文件读盘面（81 个数字或 '.'）")
    parser.add_argument(
        "--mode",
        choices=["choice", "noul"],
        default="choice",
        help="choice=每格一个问题；noul=每个候选值一个问题（问题数更多）",
    )
    parser.add_argument(
        "--strategy",
        choices=["batch", "node"],
        default="batch",
        help="batch=一次请求打包一批格子，整棵树复用（默认）；node=旧行为，每个分支点问一次",
    )
    parser.add_argument(
        "--batch-slack",
        type=int,
        default=DEFAULT_BATCH_SLACK,
        help="batch 策略一次包含的候选数档位：候选数 <= 当前最小候选数 + slack",
    )
    parser.add_argument(
        "--noul-normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="noul 模式下把本格各候选的 P(yes) 归一化成占比（默认原样使用）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="剪枝阈值；只有存在达到它的候选时才会剪掉低分候选",
    )
    parser.add_argument(
        "--no-prune", action="store_true", help="只用模型排序，不剪枝（保证搜索完整）"
    )
    parser.add_argument(
        "--no-fallback", action="store_true", help="剪枝无解时不要自动关闭剪枝重搜（可能丢解）"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="线程并行探索分支（服务端一次只处理一个 decision 请求，真正的并行在批量问题内部）",
    )
    parser.add_argument("--workers", type=int, default=None, help="并行线程数")
    parser.add_argument("--max-nodes", type=int, default=0, help="节点预算，0 表示不限")
    parser.add_argument("--server", default=MYAPI, help="Winnow 服务地址")
    parser.add_argument("--api-key", default=MYKEY, help="API key（本地服务不校验）")
    parser.add_argument("--model", default=MODEL, help="模型名")
    parser.add_argument("--timeout", type=float, default=TIMEOUT, help="单次请求超时（秒）")
    parser.add_argument(
        "--retries", type=int, default=0, help="SDK 重试次数（默认 0，避免重复计费）"
    )
    parser.add_argument(
        "--explain", action="store_true", help="打印第一次打分请求的 state/questions/answers"
    )
    parser.add_argument("--quiet", action="store_true", help="不打印分支过程")
    args = parser.parse_args(argv)

    board = load_board(args)
    if not is_consistent(board):
        print("输入盘面自相矛盾（行/列/宫有重复数字）", file=sys.stderr)
        return 2

    print_board(board, title="初始盘面:")

    client = TypeSafeClient(
        api_key=args.api_key,
        base_url=args.server,
        timeout=args.timeout,
        retry=RetryPolicy(max_retries=args.retries),
    )
    scorer = WinnowSudokuScorer(
        client,
        model=args.model,
        mode=args.mode,
        noul_normalize=args.noul_normalize,
        explain=args.explain,
    )
    solver = SudokuSolver(
        scorer,
        threshold=args.threshold,
        prune=not args.no_prune,
        fallback_complete=not args.no_fallback,
        score_strategy=args.strategy,
        batch_slack=args.batch_slack,
        parallel=args.parallel,
        max_workers=args.workers,
        max_nodes=args.max_nodes,
        log=None if args.quiet else print,
    )

    print(
        f"\nWinnow: {args.server}  model={args.model}  mode={args.mode}  "
        f"strategy={args.strategy}"
        + (f"(slack={args.batch_slack})" if args.strategy == "batch" else "")
        + f"  threshold={args.threshold}  prune={not args.no_prune}"
        + ("" if (args.no_prune or args.no_fallback) else "  (剪枝无解会自动关剪枝重搜)")
    )

    print(f"\n求解中（空格 {sum(1 for v in board if v == 0)} 个）...")
    t0 = time.perf_counter()
    try:
        solution = solver.solve(board)
    except SearchLimitExceeded as exc:
        print(f"\n搜索中断：{exc}")
        print(solver.stats.summary())
        return 3
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    finally:
        scorer.close()
    wall = time.perf_counter() - t0

    if solution is None:
        print("\n无解（或模型打分导致所有分支死亡，且完整搜索也没找到）")
        print(solver.stats.summary())
        return 1

    print("\n解出:")
    print_board(solution)
    print(f"\n校验: validate_solution={validate_solution(solution)}")
    print(f"解（81 字符）: {render_board_chars(solution)}")
    print(f"\n{solver.stats.summary()}")

    s = scorer.stats
    print(
        f"Winnow: {s.calls} 次调用 / {s.questions} 个问题 / 缓存命中 {s.cache_hits} / "
        f"失败 {s.errors} / 输入 {s.input_tokens} tokens"
    )
    if args.strategy == "batch" and solver.stats.scorer_batches:
        print(
            f"批量打分: {solver.stats.scorer_batches} 批（求解器视角）覆盖 {s.questions} 个问题；"
            "服务端把同一请求内的问题按 --decision-parallel 分波并行"
        )
    if not s.questions and not s.errors:
        print(
            "提示：这个盘面靠约束传播就解出来了，Winnow 一次都没被问到；"
            "换 hard/evil 才会需要模型打分。"
        )
    if s.errors and not s.calls:
        print(
            f"提示：所有打分请求都失败了（{s.error_messages[0] if s.error_messages else '未知错误'}）。"
            f"请确认 Winnow 服务已启动并在 {args.server} 监听。",
            file=sys.stderr,
        )
    print(f"墙钟时间: {wall:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
