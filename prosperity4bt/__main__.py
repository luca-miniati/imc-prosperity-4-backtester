import sys
import json
from collections import defaultdict
from datetime import datetime
from functools import reduce
from importlib import import_module, metadata, reload
from itertools import product
from pathlib import Path
from typing import Annotated, Any, Optional

import yaml
from tqdm import tqdm
import typer
from typer import Argument, Option, Typer

from prosperity4bt.data import has_day_data
from prosperity4bt.file_reader import FileReader, FileSystemReader, PackageResourcesReader
from prosperity4bt.models import BacktestResult, TradeMatchingMode
from prosperity4bt.open import open_visualizer
from prosperity4bt.runner import _run_backtest_single_day, run_backtest


def parse_algorithm(algorithm: Path) -> Any:
    sys.path.append(str(algorithm.parent))
    return import_module(algorithm.stem)


def parse_data(data_root: Optional[Path]) -> FileReader:
    if data_root is not None:
        return FileSystemReader(data_root)
    else:
        return PackageResourcesReader()


def get_days_for_round(file_reader: FileReader, round_num: int) -> list[tuple[int, int]]:
    """Return all (round_num, day_num) pairs that have data for the given round."""
    days = []
    for day_num in range(-5, 100):
        if has_day_data(file_reader, round_num, day_num):
            days.append((round_num, day_num))
    return days


def parse_out(out: Optional[Path], no_out: bool) -> Optional[Path]:
    if out is not None:
        return out

    if no_out:
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path.cwd() / "backtests" / f"{timestamp}.log"


def print_day_summary(result: BacktestResult) -> None:
    last_timestamp = result.activity_logs[-1].timestamp

    product_lines = []
    total_profit = 0

    for row in reversed(result.activity_logs):
        if row.timestamp != last_timestamp:
            break

        product = row.columns[2]
        profit = row.columns[-1]

        product_lines.append(f"{product}: {profit:,.0f}")
        total_profit += profit

    print(*reversed(product_lines), sep="\n")
    print(f"Total profit: {total_profit:,.0f}")


def merge_results(
    a: BacktestResult, b: BacktestResult, merge_profit_loss: bool, merge_timestamps: bool
) -> BacktestResult:
    sandbox_logs = a.sandbox_logs[:]
    activity_logs = a.activity_logs[:]
    trades = a.trades[:]

    if merge_timestamps:
        a_last_timestamp = a.activity_logs[-1].timestamp
        timestamp_offset = a_last_timestamp + 100
    else:
        timestamp_offset = 0

    sandbox_logs.extend([row.with_offset(timestamp_offset) for row in b.sandbox_logs])
    trades.extend([row.with_offset(timestamp_offset) for row in b.trades])

    if merge_profit_loss:
        profit_loss_offsets = defaultdict(float)
        for row in reversed(a.activity_logs):
            if row.timestamp != a_last_timestamp:
                break

            profit_loss_offsets[row.columns[2]] = row.columns[-1]

        activity_logs.extend(
            [row.with_offset(timestamp_offset, profit_loss_offsets[row.columns[2]]) for row in b.activity_logs]
        )
    else:
        activity_logs.extend([row.with_offset(timestamp_offset, 0) for row in b.activity_logs])

    return BacktestResult(a.round_num, a.day_num, sandbox_logs, activity_logs, trades)


def write_output(output_file: Path, merged_results: BacktestResult) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logs = [
        {
            "timestamp": log.timestamp,
            "lambdaLog": log.lambda_log,
            "sandboxLog": log.sandbox_log,
        }
        for log in merged_results.sandbox_logs
    ]
    activity_logs = "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss\n"
    activity_logs += "\n".join(map(str, merged_results.activity_logs))
    trades = [trade.trade.__dict__ for trade in merged_results.trades]

    # breakpoint()
    json_data = {
        "logs": logs,
        "activitiesLog": activity_logs,
        "tradeHistory": trades
    }

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(json_data, file, ensure_ascii=False)

    # with output_file.open("w+", encoding="utf-8") as file:
    #     file.write("Sandbox logs:\n")
    #     for row in merged_results.sandbox_logs:
    #         file.write(str(row))
    #
    #     file.write("\n\n\nActivities log:\n")
    #     file.write(
    #         "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss\n"
    #     )
    #     file.write("\n".join(map(str, merged_results.activity_logs)))
    #
    #     file.write("\n\n\n\n\nTrade History:\n")
    #     file.write("[\n")
    #     file.write(",\n".join(map(str, merged_results.trades)))
    #     file.write("]")


def print_overall_summary(results: list[BacktestResult]) -> None:
    print("Profit summary:")

    total_profit = 0
    for result in results:
        last_timestamp = result.activity_logs[-1].timestamp

        profit = 0
        for row in reversed(result.activity_logs):
            if row.timestamp != last_timestamp:
                break

            profit += row.columns[-1]

        print(f"Round {result.round_num} day {result.day_num}: {profit:,.0f}")
        total_profit += profit

    print(f"Total profit: {total_profit:,.0f}")


def format_path(path: Path) -> str:
    cwd = Path.cwd()
    if path.is_relative_to(cwd):
        return str(path.relative_to(cwd))
    else:
        return str(path)


def version_callback(value: bool) -> None:
    if value:
        print(f"prosperity4bt {metadata.version(__package__)}")
        sys.exit(0)


app = Typer(context_settings={"help_option_names": ["--help", "-h"]})

KNOWN_COMMANDS = {"grid-search"}


def _run_backtest_cli(
    algorithm: Path,
    round_num: int,
    merge_pnl: bool,
    vis: bool,
    out: Optional[Path],
    no_out: bool,
    data: Optional[Path],
    print_output: bool,
    match_trades: TradeMatchingMode,
    no_progress: bool,
    original_timestamps: bool,
) -> None:
    """Run the backtest CLI logic."""
    if out is not None and no_out:
        print("Error: --out and --no-out are mutually exclusive")
        sys.exit(1)

    try:
        trader_module = parse_algorithm(algorithm)
    except ModuleNotFoundError as e:
        print(f"{algorithm} is not a valid algorithm file: {e}")
        sys.exit(1)

    if not hasattr(trader_module, "Trader"):
        print(f"{algorithm} does not expose a Trader class")
        sys.exit(1)

    file_reader = parse_data(data)
    parsed_days = get_days_for_round(file_reader, round_num)
    if not parsed_days:
        print(f"Error: no data found for round {round_num}")
        sys.exit(1)
    output_file = parse_out(out, no_out)

    show_progress_bars = not no_progress and not print_output

    results = []
    for rnd, day_num in parsed_days:
        print(f"Backtesting {algorithm} on round {rnd} day {day_num}")

        reload(trader_module)

        result = _run_backtest_single_day(
            trader_module.Trader(),
            file_reader,
            rnd,
            day_num,
            print_output,
            match_trades,
            True,
            show_progress_bars,
        )

        print_day_summary(result)
        if len(parsed_days) > 1:
            print()

        results.append(result)

    if len(parsed_days) > 1:
        print_overall_summary(results)

    if output_file is not None:
        merged_results = reduce(lambda a, b: merge_results(a, b, merge_pnl, not original_timestamps), results)
        write_output(output_file, merged_results)
        print(f"\nSuccessfully saved backtest results to {format_path(output_file)}")

    if vis and output_file is not None:
        open_visualizer(output_file)


@app.command("run")
def run(
    algorithm: Annotated[Path, Argument(help="Path to the Python file containing the algorithm to backtest.", show_default=False, exists=True, file_okay=True, dir_okay=False, resolve_path=True)],
    round_num: Annotated[int, Argument(help="Round number to backtest (0–6). Backtests all days in the round.")],
    merge_pnl: Annotated[bool, Option("--merge-pnl", help="Merge profit and loss across days.")] = False,
    vis: Annotated[bool, Option("--vis", help="Open backtest results in https://jmerle.github.io/imc-prosperity-3-visualizer/ when done.")] = False,
    out: Annotated[Optional[Path], Option(help="File to save output log to (defaults to backtests/<timestamp>.log).", show_default=False, dir_okay=False, resolve_path=True)] = None,
    no_out: Annotated[bool, Option("--no-out", help="Skip saving output log.")] = False,
    data: Annotated[Optional[Path], Option(help="Path to data directory.", show_default=False, exists=True, file_okay=False, dir_okay=True, resolve_path=True)] = None,
    print_output: Annotated[bool, Option("--print", help="Print the trader's output to stdout while it's running.")] = False,
    match_trades: Annotated[TradeMatchingMode, Option(help="How to match orders against market trades.")] = TradeMatchingMode.all,
    no_progress: Annotated[bool, Option("--no-progress", help="Don't show progress bars.")] = False,
    original_timestamps: Annotated[bool, Option("--original-timestamps", help="Preserve original timestamps in output log.")] = False,
) -> None:  # fmt: skip
    """Backtest a Prosperity algorithm on all days in a round."""
    _run_backtest_cli(
        algorithm, round_num, merge_pnl, vis, out, no_out, data,
        print_output, match_trades, no_progress, original_timestamps,
    )


def _config_to_trader_params(symbol_params: dict[str, Any], param_order: list[str]) -> list:
    """Convert {quantity: 60, inv_reset: 70, penny: 0} to [60, 70, 0] using param_order."""
    return [symbol_params[key] for key in param_order if key in symbol_params]


def _grid_config_to_combos(config: dict[str, Any]) -> tuple[list[dict[str, list]], dict[str, list[str]], list[str]]:
    """
    Convert YAML grid config to list of full configs.
    Config may include a top-level 'param_order' key specifying the order params map to Trader.
    If omitted, each symbol's param order is taken from the YAML key order.
    Returns (combos, symbol_param_orders, symbols).
    """
    param_order = config.pop("param_order", None)
    symbols = [k for k in config.keys() if isinstance(config[k], dict)]

    symbol_combos = []
    for symbol in symbols:
        params = config[symbol]
        param_names = [k for k in params.keys() if isinstance(params[k], (list, int, float))]
        param_values = [
            params[name] if isinstance(params[name], list) else [params[name]]
            for name in param_names
        ]
        combos = []
        for values in product(*param_values):
            combo = dict(zip(param_names, values))
            combos.append(combo)
        symbol_combos.append((symbol, param_names, combos))

    order = param_order if param_order is not None else None
    symbol_param_orders = {sym: pnames for sym, pnames, _ in symbol_combos}
    results = []
    for combo_tuple in product(*[c[2] for c in symbol_combos]):
        full_config = {}
        for (symbol, param_names, _), combo in zip(symbol_combos, combo_tuple):
            use_order = order if order is not None else param_names
            full_config[symbol] = _config_to_trader_params(combo, use_order)
        results.append(full_config)
    return results, symbol_param_orders, symbols


@app.command("grid-search")
def grid_search(
    algorithm: Annotated[Path, Argument(help="Path to the Python file containing the algorithm.", exists=True, file_okay=True, dir_okay=False, resolve_path=True)],
    round_num: Annotated[int, Argument(help="Round number to backtest.")],
    config: Annotated[Path, Option("--config", "-c", help="Path to YAML config file defining the param grid.", exists=True, file_okay=True, dir_okay=False, resolve_path=True)],
    merge_pnl: Annotated[bool, Option("--merge-pnl", help="Merge profit and loss across days.")] = False,
    data: Annotated[Optional[Path], Option(help="Path to data directory.", exists=True, file_okay=False, dir_okay=True, resolve_path=True)] = None,
    out_csv: Annotated[Optional[Path], Option("--out-csv", help="Save all results to CSV file.")] = None,
    no_progress: Annotated[bool, Option("--no-progress", help="Don't show progress bar.")] = False,
) -> None:
    """Grid search over Trader parameters to find the best configuration."""
    try:
        trader_module = parse_algorithm(algorithm)
    except ModuleNotFoundError as e:
        print(f"{algorithm} is not a valid algorithm file: {e}")
        sys.exit(1)

    if not hasattr(trader_module, "Trader"):
        print(f"{algorithm} does not expose a Trader class")
        sys.exit(1)

    with config.open() as f:
        grid_config = yaml.safe_load(f)

    combos, symbol_param_orders, symbols = _grid_config_to_combos(grid_config)
    print(f"Testing {len(combos)} parameter combinations...")

    best_profit = float("-inf")
    best_config = None
    results_list = []

    # iterator = tqdm(combos, ascii=True) if not no_progress else combos

    for i, full_config in enumerate(combos):
        reload(trader_module)
        trader = trader_module.Trader(config=full_config)

        result = run_backtest(
            trader,
            round_num,
            data_path=data,
            merge_pnl=merge_pnl,
            print_output=False,
            show_progress=False,
        )

        profit = result.profit
        print(f"Config {(i + 1)} / {len(combos)}: {full_config}")
        print(f"Profit {(i + 1)} / {len(combos)}: {profit:,.0f}")
        results_list.append((full_config, profit))

        if profit > best_profit:
            best_profit = profit
            best_config = full_config

    print(f"\nBest config: {best_config}")
    print(f"Best profit: {best_profit:,.0f}")

    if out_csv is not None:
        import csv

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            header = []
            for symbol in symbols:
                for param in symbol_param_orders.get(symbol, []):
                    header.append(f"{symbol}_{param}")
            header.append("profit")
            writer.writerow(header)

            for full_config, profit in results_list:
                row = []
                for symbol in symbols:
                    params = full_config.get(symbol, [])
                    param_names = symbol_param_orders.get(symbol, [])
                    for i in range(len(param_names)):
                        row.append(params[i] if i < len(params) else "")
                row.append(profit)
                writer.writerow(row)
        print(f"Results saved to {format_path(out_csv)}")


def main() -> None:
    # If first non-option arg is not a known command, treat as default "run"
    args = [a for a in sys.argv[1:] if not a.startswith("-") or a in ("--help", "-h", "--version", "-v")]
    if args and args[0] not in KNOWN_COMMANDS:
        sys.argv.insert(1, "run")
    app()


if __name__ == "__main__":
    main()
