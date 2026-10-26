#!/usr/bin/env python3
"""Measure isolated ART editor startup and main-loop intervals under Xvfb.

Requires Xvfb and ImageMagick import. Copies the input and sidecar so GUI
preferences or autosave cannot modify the source photo. This is not an
interactive slider/zoom benchmark; the harness terminates ART after capture.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', required=True, type=Path, help='graphical ART binary')
    parser.add_argument('--matrix', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seconds', type=int, default=25)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--browser', action='store_true')
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error('--seconds must be positive')
    binary = args.cli.resolve(strict=True)
    source = args.input.resolve(strict=True)
    sidecar = Path(str(source)+'.arp').resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    photo_dir = output/'input'
    photo_dir.mkdir()
    photo = photo_dir/source.name
    shutil.copy2(source, photo)
    shutil.copy2(sidecar, Path(str(photo)+'.arp'))
    results = []
    matrix = json.loads(args.matrix.read_text())
    variants = [dict(matrix[(i+r)%len(matrix)], repeat=r) for r in range(args.repeat) for i in range(len(matrix))]
    for index, variant in enumerate(variants):
        shutil.copy2(sidecar, Path(str(photo)+'.arp'))
        prefix = output/f'{index:02}'
        server = subprocess.Popen(['Xvfb', '-displayfd', '1', '-screen', '0', '1600x1000x24', '-nolisten', 'tcp'],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        process = None
        try:
            display = ':'+server.stdout.readline().strip()
            if display == ':':
                raise RuntimeError('Xvfb did not provide a display')
            env = dict(os.environ, DISPLAY=display, ART_SETTINGS=str(prefix)+'-settings',
                       ART_CACHE=str(prefix)+'-cache', ART_PROFILE='1', ART_UI_PROFILE='1', ART_OPEN_TRACE='1')
            for key, value in variant.get('env', {}).items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = value
            with Path(str(prefix)+'.stdout').open('w') as out, Path(str(prefix)+'.stderr').open('w') as err:
                process = subprocess.Popen([str(binary), '-N' if args.browser else '-s', str(photo)], env=env, stdout=out, stderr=err)
                deadline = time.monotonic()+args.seconds
                while process.poll() is None and time.monotonic()<deadline:
                    time.sleep(.1)
                alive = process.poll() is None
                if alive:
                    subprocess.run(['import', '-display', display, '-window', 'root', str(prefix)+'.png'], check=True)
                stop(process)
            lines = Path(str(prefix)+'.stderr').read_text(errors='replace').splitlines()
            ticks = [json.loads(s[len('ART_UI_PROFILE '):])['interval_ms'] for s in lines if s.startswith('ART_UI_PROFILE ')]
            events = [json.loads(s[len('ART_PROFILE '):]) for s in lines if s.startswith('ART_PROFILE ')]
            trace = [json.loads(s[len('ART_OPEN_TRACE '):]) for s in lines if s.startswith('ART_OPEN_TRACE ')]
            values = sorted(ticks)
            result = dict(name=variant['name'], repeat=variant['repeat'], trace=trace, env=variant.get('env', {}), alive_at_capture=alive,
                          returncode=process.returncode, intervals_ms=ticks, events=events,
                          median_ms=statistics.median(values) if values else None,
                          p95_ms=values[min(len(values)-1,int(len(values)*.95))] if values else None,
                          max_ms=max(values) if values else None, screenshot=str(prefix)+'.png')
            results.append(result)
            (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
            print(variant['name'], {k:result[k] for k in ['alive_at_capture','median_ms','p95_ms','max_ms']},flush=True)
            if not alive or not ticks or not any(e['name']=='preview_update' for e in events):
                raise RuntimeError('Editor did not finish a preview; inspect logs and screenshot')
        finally:
            if process is not None:
                stop(process)
            stop(server)


if __name__ == '__main__':
    main()
