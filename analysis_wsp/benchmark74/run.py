from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bootstrap import instrument_ils, validate_environment
from .execute import execute, status
from .prepare import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description="Campanha final WSP: seis algoritmos nas 74 instancias e trajetorias IBS nas 19 suplementares.")
    parser.add_argument("--profile", default="benchmark74_v2_4")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--license", type=Path, required=True)
    doctor.add_argument("--instrument-ils", action="store_true")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--license", type=Path, required=True)
    prepare_parser.add_argument("--shard-count", type=int, default=2)
    prepare_parser.add_argument("--official", action="store_true")

    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--license", type=Path, required=True)
    pilot.add_argument("--shard-index", type=int, default=0)
    pilot.add_argument("--workers", type=int, default=4)
    pilot.add_argument("--solver-workers", type=int, default=1)
    pilot.add_argument("--seconds", type=int, default=10)
    pilot.add_argument("--resume", action="store_true")

    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--license", type=Path, required=True)
    execute_parser.add_argument("--shard-index", type=int, required=True)
    execute_parser.add_argument("--workers", type=int, required=True)
    execute_parser.add_argument("--solver-workers", type=int, required=True)
    execute_parser.add_argument("--resume", action="store_true")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--shard-index", type=int, required=True)

    args = parser.parse_args()
    if args.command == "doctor":
        value = {"instrumentation": instrument_ils() if args.instrument_ils else "not_requested"}
        value["environment"] = validate_environment(args.license)
    elif args.command == "prepare":
        value = prepare(profile=args.profile, shard_count=args.shard_count, license_path=args.license, official=args.official)
    elif args.command == "pilot":
        value = execute(
            profile=args.profile,
            shard_index=args.shard_index,
            workers=args.workers,
            solver_workers=args.solver_workers,
            license_path=args.license,
            resume=args.resume,
            pilot_limit=args.seconds,
        )
    elif args.command == "execute":
        value = execute(
            profile=args.profile,
            shard_index=args.shard_index,
            workers=args.workers,
            solver_workers=args.solver_workers,
            license_path=args.license,
            resume=args.resume,
            pilot_limit=None,
        )
    else:
        value = status(args.profile, args.shard_index)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
