#!/usr/bin/env python3
"""Sequential real exports, isolated settings, per-process CPU/RAM and GPU samples.

Matrix JSON: [{"name":"cpu28","env":{"OMP_NUM_THREADS":"28","ART_CUDA":"0"}}].
GPU utilization is device-wide; compute memory is filtered to the ART PID.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def sample(pid, gpu):
    result = {"time": time.monotonic()}
    try:
        status = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines())
        result.update(rss_kib=int(status["VmRSS"].split()[0]),
                      hwm_kib=int(status["VmHWM"].split()[0]), threads=int(status["Threads"]))
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        result["cpu_seconds"] = (int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, KeyError, ValueError):
        pass
    if gpu:
        try:
            info = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                   "--format=csv,noheader,nounits", "-i", "0"],
                                  capture_output=True, text=True, timeout=3, check=True).stdout.strip().split(",")
            result.update(gpu_percent=float(info[0]), device_vram_mib=float(info[1]))
            rows = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                   "--format=csv,noheader,nounits", "-i", "0"],
                                  capture_output=True, text=True, timeout=3, check=True).stdout.splitlines()
            result["process_vram_mib"] = sum(float(row.split(",")[1]) for row in rows
                                             if row.split(",")[0].strip() == str(pid))
        except (OSError, ValueError, subprocess.SubprocessError):
            result["gpu_unavailable"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--gpu-samples", action="store_true")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--fast", action="store_true", help="ART fast-export pipeline (resize before processing)")
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("repeat must be positive")
    cli = args.cli.resolve(strict=True)
    inputs = [p.resolve(strict=True) for p in args.inputs]
    profiles = [args.profile.resolve(strict=True) if args.profile else Path(str(p)+".arp").resolve(strict=True) for p in inputs]
    matrix = json.loads(args.matrix.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runs = []
    for repeat in range(args.repeat):
        for index in range(len(inputs)):
            case = (index + repeat) % len(inputs)
            for offset in range(len(matrix)):
                variant = matrix[(offset+repeat+index) % len(matrix)]
                name = variant["name"]
                if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
                    raise ValueError("invalid variant name")
                stem = f"{repeat:02}-{case:02}-{name}"
                dest = output / (stem+".tif")
                env = dict(os.environ, ART_PROFILE="1", ART_SETTINGS=str(output / (stem+"-settings")),
                           ART_CACHE=str(output / (stem+"-cache")))
                for key, value in variant.get('env', {}).items():
                    if value is None:
                        env.pop(key, None)
                    else:
                        env[key] = value
                command = [str(cli), "-q", "-a", "-o", str(dest), "-t", "-b16", "-p", str(profiles[case]), "-c", str(inputs[case])]
                if args.fast:
                    command.insert(1, "-f")
                measurements = []
                start = time.monotonic()
                with (output/(stem+".stdout")).open("w") as stdout, (output/(stem+".stderr")).open("w") as stderr:
                    process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
                    try:
                        while process.poll() is None:
                            measurements.append(sample(process.pid, args.gpu_samples))
                            try:
                                process.wait(timeout=.2)
                            except subprocess.TimeoutExpired:
                                pass
                        code = process.wait()
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            process.wait(timeout=10)
                seconds = time.monotonic() - start
                events = [json.loads(line[len("ART_PROFILE "):]) for line in (output/(stem+".stderr")).read_text().splitlines() if line.startswith("ART_PROFILE ")]
                record = dict(name=name, repeat=repeat, input=str(inputs[case]), profile=str(profiles[case]),
                              command=command, env=variant.get("env", {}), seconds=seconds, returncode=code,
                              events=events, samples=measurements, output=str(dest))
                runs.append(record)
                (output/"results.json").write_text(json.dumps(dict(matrix=matrix, gpu_sampling=args.gpu_samples, runs=runs), indent=2)+"\n")
                print(f"{name} {inputs[case].name} repeat={repeat} seconds={seconds:.3f} code={code}", flush=True)
                if code or not dest.is_file() or not events:
                    raise RuntimeError(f"Export failed: {output/(stem+'.stderr')}")


if __name__ == "__main__":
    main()
