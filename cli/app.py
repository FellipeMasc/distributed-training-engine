import argparse

from engine.cli.commands.dryrun import run_dryrun_command
from engine.cli.commands.inspect import run_inspect_command
from engine.cli.commands.train import run_train_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run DP training")
    train_parser.add_argument("--config", required=True, help="Path to YAML config")

    dryrun_parser = subparsers.add_parser("dryrun", help="Validate run setup")
    dryrun_parser.add_argument("--config", required=True, help="Path to YAML config")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect resolved plan")
    inspect_parser.add_argument("--config", required=True, help="Path to YAML config")
    inspect_parser.add_argument("--json", action="store_true", help="Emit JSON output")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        return run_train_command(args.config)
    if args.command == "dryrun":
        return run_dryrun_command(args.config)
    if args.command == "inspect":
        return run_inspect_command(args.config, args.json)

    parser.error(f"Unknown command: {args.command}")
    return 2
