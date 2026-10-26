#!/usr/bin/env python3
"""Package a Release bundle build as art-optimized, without installing it.

See doc/optimized-deb.pt-BR.md for the matching CMake configuration.
Only the staging tree and output directory are written. System libraries remain
Debian dependencies; optional --deps-root supplies extracted build dependencies.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def main():
    os.umask(0o022)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('dist'))
    parser.add_argument('--version', required=True)
    parser.add_argument('--deps-root', type=Path)
    parser.add_argument('--cuda-license', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    build = args.build_dir.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run('dpkg', '--validate-version', args.version)
    cache = {}
    for line in (build/'CMakeCache.txt').read_text().splitlines():
        if line and not line.startswith(('#', '//')) and '=' in line:
            key, value = line.split('=', 1)
            cache[key.split(':')[0]] = value
    required = {'BUILD_BUNDLE': 'ON', 'BUNDLE_BASE_INSTALL_DIR': '/usr/lib/art-optimized',
                'DATADIR': '.', 'ART_ENABLE_CUDA': 'ON', 'ART_CUDA_STATIC_RUNTIME': 'ON',
                'CMAKE_INSTALL_RPATH': '', 'CMAKE_INSTALL_RPATH_USE_LINK_PATH': 'OFF'}
    for key, value in required.items():
        if cache.get(key) != value:
            raise RuntimeError(f'{key} must be {value!r}; see packaging documentation')
    architecture = run('dpkg', '--print-architecture', capture_output=True).stdout.strip()
    if architecture != 'amd64':
        raise RuntimeError('This package configuration has only been validated on amd64')
    filename = f'art-optimized_{args.version}_{architecture}.deb'
    destination = output/filename
    if destination.exists():
        raise FileExistsError(destination)

    with tempfile.TemporaryDirectory(prefix='art-deb-') as temporary:
        work = Path(temporary)
        stage = work/'root'
        env = dict(os.environ, DESTDIR=str(stage))
        with (work/'install.log').open('w') as log:
            run('cmake', '--install', str(build), '--strip', env=env, stdout=log)
        app = stage/'usr/lib/art-optimized'
        documentation = stage/'usr/share/doc/art-optimized'
        documentation.mkdir(parents=True)
        shutil.copy2(source/'LICENSE.txt', documentation/'copyright')
        shutil.copy2(args.cuda_license, documentation/'CUDA-EULA.txt')
        for name in ['optimized-deb.pt-BR.md', 'editor-performance.pt-BR.md',
                     'advanced-performance.pt-BR.md', 'gpu-feasibility.pt-BR.md']:
            shutil.copy2(source/'doc'/name, documentation/name)
        shutil.copy2(source/'tools/run_art_optimized.sh', app/'run_art_optimized.sh')
        (app/'run_art_optimized.sh').chmod(0o755)
        binaries = stage/'usr/bin'
        binaries.mkdir(parents=True)
        for name, executable in [('art-optimized', 'ART'), ('art-optimized-cli', 'ART-cli')]:
            launcher = binaries/name
            launcher.write_text('#!/bin/sh\nset -eu\n'
                'prefix=$(CDPATH= cd -- "$(dirname -- "$(readlink -f -- "$0")")/../lib/art-optimized" && pwd)\n'
                f'exec "$prefix/run_art_optimized.sh" "$prefix/{executable}" "$@"\n')
            launcher.chmod(0o755)

        # Give the package its own desktop entry/icons; coexist with upstream ART.
        desktop = (app/'share/applications/ART.desktop').read_text()
        desktop = desktop.replace('Name=ART\n', 'Name=ART Optimized\nName[pt_BR]=ART Otimizado\n')
        desktop = desktop.replace('Icon=ART\n', 'Icon=art-optimized\n')
        desktop = desktop.replace('Exec=ART %f\n', 'Exec=art-optimized -N %f\n')
        desktop = desktop.replace('StartupWMClass=ART\n', 'StartupWMClass=ART\nTryExec=art-optimized\n')
        applications = stage/'usr/share/applications'
        applications.mkdir(parents=True)
        desktop_path = applications/'art-optimized.desktop'
        desktop_path.write_text(desktop)
        run('desktop-file-validate', str(desktop_path))
        for icon in (app/'share/icons').glob('hicolor/*/apps/ART.png'):
            target = stage/'usr/share/icons'/icon.relative_to(app/'share/icons')
            target = target.with_name('art-optimized.png')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(icon, target)
        shutil.rmtree(app/'share/applications')
        shutil.rmtree(app/'share/icons')
        # Upstream AppStream ID describes the upstream desktop entry, not this fork.
        shutil.rmtree(app/'metainfo')

        # Resolve real ABI dependencies, including locally extracted .deb metadata.
        debian = work/'debian'
        debian.mkdir()
        (debian/'control').write_text('Source: art-optimized\nSection: graphics\nPriority: optional\n'
            'Maintainer: Local ART build <art-optimized@localhost>\n\n'
            'Package: art-optimized\nArchitecture: amd64\nDescription: Optimized ART local build\n')
        shlibs = []
        search = []
        if args.deps_root:
            deps = args.deps_root.resolve(strict=True)
            search = ['-l'+str(deps/'usr/lib/x86_64-linux-gnu')]
            for index, deb in enumerate(sorted((deps/'debs').glob('*.deb'))):
                metadata = work/f'metadata-{index}'
                run('dpkg-deb', '-e', str(deb), str(metadata))
                if (metadata/'shlibs').exists():
                    shlibs.append((metadata/'shlibs').read_text())
        local_shlibs = work/'shlibs.local'
        local_shlibs.write_text('\n'.join(shlibs))
        dependency_output = run('dpkg-shlibdeps', '-O', '-L'+str(local_shlibs), *search,
            '-e'+str(app/'ART'), '-e'+str(app/'ART-cli'), cwd=work, capture_output=True)
        dependencies = next(line.split('=', 1)[1] for line in dependency_output.stdout.splitlines()
                            if line.startswith('shlibs:Depends='))
        for executable in ['ART', 'ART-cli']:
            dynamic = run('readelf', '-d', str(app/executable), capture_output=True).stdout
            if 'RPATH' in dynamic or 'RUNPATH' in dynamic or 'libcudart.so' in dynamic:
                raise RuntimeError(f'Unexpected runtime search path or shared CUDA runtime: {executable}')

        manifest = {'package': 'art-optimized', 'version': args.version, 'architecture': architecture,
                    'dependencies': dependencies, 'cuda_runtime': 'static',
                    'cuda_architectures': cache['CMAKE_CUDA_ARCHITECTURES'],
                    'cpu_flags': cache.get('PROC_TARGET_2_FLAGS'),
                    'git_commit': run('git', 'rev-parse', 'HEAD', cwd=source, capture_output=True).stdout.strip(),
                    'working_tree_modified': True,
                    'binary_sha256': {name: hashlib.sha256((app/name).read_bytes()).hexdigest()
                                      for name in ['ART', 'ART-cli']},
                    'build_information': (app/'AboutThisBuild.txt').read_text()}
        (documentation/'build-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        control = stage/'DEBIAN'
        control.mkdir()
        size = sum(p.stat().st_size for p in stage.rglob('*') if p.is_file())
        (control/'control').write_text(f'Package: art-optimized\nVersion: {args.version}\n'
            f'Architecture: {architecture}\nSection: graphics\nPriority: optional\n'
            'Maintainer: Local ART build <art-optimized@localhost>\n'
            f'Installed-Size: {(size+1023)//1024}\nDepends: bash, {dependencies}\n'
            'Description: ART with selective CUDA and multicore optimizations\n'
            ' Local Debian 13 build tuned for the build machine CPU and NVIDIA SM 8.6.\n'
            ' Includes GUI, CLI, processing profiles and the optimized launch preset.\n'
            ' CUDA is optional at runtime; unavailable GPU operations fall back to CPU.\n')
        (control/'md5sums').write_text(''.join(
            f'{hashlib.md5(p.read_bytes()).hexdigest()}  {p.relative_to(stage)}\n'
            for p in sorted(stage.rglob('*')) if p.is_file() and control not in p.parents))
        for path in stage.rglob('*'):
            if not path.is_symlink():
                path.chmod(0o755 if path.is_dir() or path.stat().st_mode & 0o111 else 0o644)
        run('dpkg-deb', '--root-owner-group', '-Zxz', '-z6', '--build', str(stage), str(destination))
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        (output/(filename+'.sha256')).write_text(f'{digest}  {filename}\n')
        (output/(filename+'.manifest.json')).write_text(json.dumps(manifest, indent=2)+'\n')
        print(destination)


if __name__ == '__main__':
    main()
