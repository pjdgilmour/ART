#!/usr/bin/env python3
"""Profile real ART CLI exports without writing to the inputs or their sidecars."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="new directory for TIFFs, logs, isolated settings and results")
    parser.add_argument("--repeat", type=positive, default=3)
    parser.add_argument("--threads", type=positive, help="OpenMP thread count; default: runtime default")
    parser.add_argument("--profile", type=Path, help="override profile for all inputs; default: each input's .arp")
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    cli = args.cli.resolve(strict=True)
    inputs = [p.resolve(strict=True) for p in args.inputs]
    profiles = [args.profile.resolve(strict=True) if args.profile else Path(str(p) + ".arp").resolve(strict=True)
                for p in inputs]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(ART_PROFILE="1", ART_SETTINGS=str(output / "settings"), ART_CACHE=str(output / "cache"))
    if args.threads:
        env["OMP_NUM_THREADS"] = str(args.threads)
    results = []
    # Rotate image order between rounds to reduce systematic order effects.
    for repeat in range(args.repeat):
        for offset in range(len(inputs)):
            index = (offset + repeat) % len(inputs)
            source, profile = inputs[index], profiles[index]
            stem = f"{index:02d}-{source.stem}-{repeat:02d}"
            dest = output / (stem + ".tif")
            command = [str(cli), "-q", "-a", "-o", str(dest), "-t", "-b16", "-p", str(profile), "-c", str(source)]
            start = time.perf_counter()
            with (output / (stem + ".stdout")).open("w") as stdout, (output / (stem + ".stderr")).open("w") as stderr:
                proc = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
            elapsed = time.perf_counter() - start
            events = []
            for line in (output / (stem + ".stderr")).read_text(errors="replace").splitlines():
                if line.startswith("ART_PROFILE "):
                    events.append(json.loads(line[len("ART_PROFILE "):]))
            record = dict(input=str(source), profile=str(profile), repeat=repeat, command=command,
                          seconds=elapsed, returncode=proc.returncode, events=events)
            results.append(record)
            (output / "results.json").write_text(json.dumps(dict(threads=env.get("OMP_NUM_THREADS"),
                                                                runs=results), indent=2) + "\n")
            if proc.returncode or not dest.is_file() or not events:
                raise SystemExit(f"Export/profiling failed: inspect {output / (stem + '.stderr')}")
            print(f"{source.name} run={repeat + 1} total={elapsed:.3f}s", flush=True)
    for source in inputs:
        runs = [r for r in results if r["input"] == str(source)]
        totals = [r["seconds"] for r in runs]
        print(f"\n{source.name}: median={statistics.median(totals):.3f}s min={min(totals):.3f}s max={max(totals):.3f}s")
        by_name = defaultdict(list)
        for run in runs:
            sums = defaultdict(int)
            for event in run["events"]:
                sums[event["name"]] += event["us"]
            for name, us in sums.items():
                by_name[name].append(us / 1000)
        for name, values in sorted(by_name.items(), key=lambda item: statistics.median(item[1]), reverse=True):
            if statistics.median(values) >= 1:
                print(f"  {name}: {statistics.median(values):.2f} ms (inclusive)")


if __name__ == "__main__":
    main()
