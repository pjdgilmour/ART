#!/usr/bin/env python3
"""Reproduce browser clicks on a private Xvfb display (never the desktop).

Run under dbus-run-session. Inputs/sidecars are copied; no original is modified.
Actions JSON: [{"at":8,"shot":"browser"},{"at":9,"click":[x,y],"count":2},
{"at":9.6,"shot":"early"},...]. Coordinates must be inspected in screenshots.
Results include monotonic event timestamps, heartbeat intervals and ART RSS.
Xvfb, libXtst and ImageMagick import are required. ART is terminated at the end.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from probe_art_gui import stop


class Input:
    def __init__(self, display):
        self.x = ctypes.CDLL('libX11.so.6')
        self.xt = ctypes.CDLL('libXtst.so.6')
        self.x.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self.x.XOpenDisplay.restype = ctypes.c_void_p
        self.x.XFlush.argtypes = [ctypes.c_void_p]
        self.x.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self.xt.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
        self.xt.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
        self.display = self.x.XOpenDisplay(display.encode())
        if not self.display:
            raise RuntimeError('Cannot open private display')

    def click(self, position, count):
        self.xt.XTestFakeMotionEvent(self.display, -1, *position, 0)
        for _ in range(count):
            self.xt.XTestFakeButtonEvent(self.display, 1, 1, 0)
            self.xt.XTestFakeButtonEvent(self.display, 1, 0, 0)
            self.x.XFlush(self.display)
            time.sleep(.06)

    def close_window(self):
        # Select the largest mapped child on this private Xvfb server, then
        # send its standard WM_DELETE_WINDOW protocol (there is no WM here).
        class Data(ctypes.Union):
            _fields_ = [('l', ctypes.c_long*5)]
        class Message(ctypes.Structure):
            _fields_ = [('type',ctypes.c_int),('serial',ctypes.c_ulong),
                        ('send_event',ctypes.c_int),('display',ctypes.c_void_p),
                        ('window',ctypes.c_ulong),('message_type',ctypes.c_ulong),
                        ('format',ctypes.c_int),('data',Data)]
        class Event(ctypes.Union):
            _fields_ = [('message',Message),('pad',ctypes.c_long*24)]
        self.x.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        self.x.XDefaultRootWindow.restype = ctypes.c_ulong
        self.x.XQueryTree.argtypes = [ctypes.c_void_p,ctypes.c_ulong,ctypes.POINTER(ctypes.c_ulong),ctypes.POINTER(ctypes.c_ulong),ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),ctypes.POINTER(ctypes.c_uint)]
        self.x.XGetGeometry.argtypes = [ctypes.c_void_p,ctypes.c_ulong,ctypes.POINTER(ctypes.c_ulong),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_uint),ctypes.POINTER(ctypes.c_uint),ctypes.POINTER(ctypes.c_uint),ctypes.POINTER(ctypes.c_uint)]
        self.x.XFree.argtypes = [ctypes.c_void_p]
        self.x.XInternAtom.argtypes = [ctypes.c_void_p,ctypes.c_char_p,ctypes.c_int]
        self.x.XInternAtom.restype = ctypes.c_ulong
        self.x.XSendEvent.argtypes = [ctypes.c_void_p,ctypes.c_ulong,ctypes.c_int,ctypes.c_long,ctypes.POINTER(Event)]
        root = self.x.XDefaultRootWindow(self.display)
        parent, returned_root = ctypes.c_ulong(), ctypes.c_ulong()
        children = ctypes.POINTER(ctypes.c_ulong)()
        count = ctypes.c_uint()
        if not self.x.XQueryTree(self.display,root,ctypes.byref(returned_root),ctypes.byref(parent),ctypes.byref(children),ctypes.byref(count)):
            raise RuntimeError('Cannot query private display')
        windows = []
        for i in range(count.value):
            x,y = ctypes.c_int(),ctypes.c_int()
            w,h,border,depth = (ctypes.c_uint() for _ in range(4))
            if self.x.XGetGeometry(self.display,children[i],ctypes.byref(returned_root),ctypes.byref(x),ctypes.byref(y),ctypes.byref(w),ctypes.byref(h),ctypes.byref(border),ctypes.byref(depth)):
                windows.append((w.value*h.value,children[i]))
        self.x.XFree(children)
        window = max(windows)[1]
        event = Event()
        event.message.type = 33
        event.message.display = self.display
        event.message.window = window
        event.message.message_type = self.x.XInternAtom(self.display,b'WM_PROTOCOLS',0)
        event.message.format = 32
        event.message.data.l[0] = self.x.XInternAtom(self.display,b'WM_DELETE_WINDOW',0)
        self.x.XSendEvent(self.display,window,0,0,ctypes.byref(event))
        self.x.XFlush(self.display)

    def close(self):
        self.x.XCloseDisplay(self.display)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cli', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--actions', type=Path, required=True)
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--close-at-end', action='store_true', help='verify normal WM_DELETE_WINDOW shutdown')
    p.add_argument('inputs', type=Path, nargs='+')
    a = p.parse_args()
    binary = a.cli.resolve(strict=True)
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    photos = root/'input'
    photos.mkdir()
    for source in a.inputs:
        shutil.copy2(source, photos/source.name)
        sidecar = Path(str(source)+'.arp')
        if sidecar.exists():
            shutil.copy2(sidecar, photos/sidecar.name)
    actions = json.loads(a.actions.read_text())
    matrix = json.loads(a.matrix.read_text())
    results = []
    for repeat in range(a.repeat):
        for index in range(len(matrix)):
            variant = matrix[(index+repeat) % len(matrix)]
            # ART autosaves computed WB/noise/histogram settings on photo close.
            # Restore sidecars before EVERY variant, not just once per matrix.
            # RAW contents remain shared read-only between these sessions.
            for source in a.inputs:
                original = Path(str(source)+'.arp')
                target = photos/(source.name+'.arp')
                if original.exists():
                    shutil.copy2(original, target)
                elif target.exists():
                    target.unlink()
            sidecar_hashes = {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in photos.glob('*.arp')}
            prefix = root/f'{len(results):02}'
            server = subprocess.Popen(['Xvfb', '-displayfd', '1', '-screen', '0', '1600x1000x24', '-nolisten', 'tcp'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            process = None
            controller = None
            try:
                display = ':'+server.stdout.readline().strip()
                if display == ':':
                    raise RuntimeError('Xvfb startup failed')
                env = dict(os.environ, DISPLAY=display, ART_SETTINGS=str(prefix)+'-settings', ART_CACHE=str(prefix)+'-cache', ART_PROFILE='1', ART_UI_PROFILE='1', ART_OPEN_TRACE='1')
                for key, value in variant.get('env', {}).items():
                    if value is None:
                        env.pop(key, None)
                    else:
                        env[key] = value
                with Path(str(prefix)+'.stdout').open('w') as out, Path(str(prefix)+'.stderr').open('w') as err:
                    start = time.monotonic()
                    process = subprocess.Popen([str(binary), str(photos)], env=env, stdout=out, stderr=err)
                    controller = Input(display)
                    recorded, samples = [], []
                    for action in actions:
                        deadline = start+action['at']
                        while time.monotonic()<deadline and process.poll() is None:
                            try:
                                rss = int(Path(f'/proc/{process.pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
                                samples.append(dict(us=int(time.monotonic()*1e6), rss_bytes=rss))
                            except (OSError, IndexError):
                                pass
                            time.sleep(.05)
                        if process.poll() is not None:
                            raise RuntimeError('ART exited before the last action')
                        record = dict(action, us=int(time.monotonic()*1e6))
                        if 'click' in action:
                            controller.click(action['click'], action.get('count', 1))
                        if action.get('close'):
                            controller.close_window()
                        if 'shot' in action:
                            subprocess.run(['import', '-display', display, '-window', 'root', str(prefix)+'-'+action['shot']+'.png'], check=True)
                        recorded.append(record)
                    if a.close_at_end:
                        controller.close_window()
                        if process.wait(timeout=15) != 0:
                            raise RuntimeError('ART failed normal shutdown')
                    else:
                        stop(process)
                lines = Path(str(prefix)+'.stderr').read_text(errors='replace').splitlines()
                result = dict(name=variant['name'], repeat=repeat, env=variant.get('env', {}), actions=recorded, samples=samples, returncode=process.returncode, normal_close=a.close_at_end, sidecar_sha256_before=sidecar_hashes)
                for key, marker in [('trace','ART_OPEN_TRACE '),('ticks','ART_UI_PROFILE '),('events','ART_PROFILE ')]:
                    result[key] = [json.loads(line[len(marker):]) for line in lines if line.startswith(marker)]
                results.append(result)
                (root/'results.json').write_text(json.dumps(results, indent=2)+'\n')
                print(variant['name'], repeat, 'opens', sum(e['event']=='open_requested' for e in result['trace']), flush=True)
            finally:
                if controller:
                    controller.close()
                if process:
                    stop(process)
                stop(server)


if __name__ == '__main__':
    main()
