"""Build, archive, and run a standalone desktop bundle on its native OS."""
import argparse
import hashlib
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile


ROOT = Path(__file__).resolve().parents[1]
APP = 'SEM_Stitcher'
DEPENDENCIES = ('customtkinter', 'numpy', 'opencv-contrib-python-headless', 'pillow',
                'pyinstaller', 'darkdetect', 'packaging')


def write_notices(directory):
    directory.mkdir()
    for name in DEPENDENCIES:
        package = distribution(name)
        for entry in package.files or ():
            if (entry.is_absolute() or '..' in entry.parts or
                    not entry.name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE'))):
                continue
            path = package.locate_file(entry)
            if path.is_file():
                target = directory / name / entry
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    python_license = Path(sysconfig.get_path('stdlib')) / 'LICENSE.txt'
    if python_license.is_file():
        shutil.copyfile(python_license, directory / 'Python-LICENSE.txt')


def build(target, output):
    mac = sys.platform == 'darwin'
    machine = platform.machine().lower()
    expected = {('darwin', 'arm64'): 'macos-arm64', ('darwin', 'x86_64'): 'macos-intel',
                ('win32', 'amd64'): 'windows-x64'}.get((sys.platform, machine))
    if target != expected:
        raise RuntimeError(f'{target} requires a native runner, got {sys.platform}/{machine}')
    output.mkdir(parents=True, exist_ok=True)
    name = f'SEM-Stitcher-{target}'
    with tempfile.TemporaryDirectory(prefix='sem-build-') as directory:
        work = Path(directory)
        env = dict(os.environ, PYINSTALLER_CONFIG_DIR=str(work / 'cache'))
        command = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir',
                   '--windowed', '--name', APP, '--collect-data', 'customtkinter', '--paths', str(ROOT),
                   '--distpath', str(work / 'dist'), '--workpath', str(work / 'build'), '--specpath', str(work)]
        if mac:
            command += ['--target-arch', machine, '--osx-bundle-identifier', 'io.github.haow1026.sem-stitcher']
        subprocess.run(command + [str(ROOT / 'tools/desktop_entry.py')], cwd=ROOT, env=env, check=True)

        stage = work / name
        stage.mkdir()
        bundle = work / 'dist' / (APP + '.app' if mac else APP)
        if mac:
            subprocess.run(['ditto', str(bundle), str(stage / bundle.name)], check=True)
        else:
            shutil.copytree(bundle, stage / bundle.name)
        shutil.copyfile(ROOT / 'docs/desktop-packages.md', stage / 'START-HERE.md')
        write_notices(stage / 'licenses')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        info = dict(commit=commit, target=target, python=platform.python_version(),
                    dependencies={dep: version(dep) for dep in DEPENDENCIES})
        (stage / 'build.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')

        # Run the archive's extracted app, rather than the build tree, to catch
        # missing resources, broken bundle links and non-portable dependencies.
        archive = work / (name + '.zip')
        unpacked = work / 'verify'
        if mac:
            subprocess.run(['ditto', '-c', '-k', '--sequesterRsrc', '--keepParent', str(stage), str(archive)], check=True)
            subprocess.run(['ditto', '-x', '-k', str(archive), str(unpacked)], check=True)
            app_bundle = unpacked / name / bundle.name
            executable = app_bundle / 'Contents/MacOS' / APP
            subprocess.run(['codesign', '--verify', '--deep', '--strict', str(app_bundle)], check=True)
            arches = subprocess.check_output(['lipo', '-archs', str(executable)], text=True).split()
            if arches != [machine]:
                raise RuntimeError(f'Unexpected app architecture: {arches}')
        else:
            shutil.make_archive(str(archive.with_suffix('')), 'zip', work, name)
            shutil.unpack_archive(archive, unpacked)
            executable = unpacked / name / bundle.name / (APP + '.exe')

        smoke = work / 'smoke.json'
        smoke_env = {key: value for key, value in os.environ.items()
                     if key.upper() not in ('PYTHONPATH', 'PYTHONHOME')}
        run = subprocess.run([str(executable), '--smoke-test', str(smoke)], cwd=unpacked,
                             env=smoke_env, timeout=90)
        if not smoke.is_file():
            raise RuntimeError(f'Packaged application produced no smoke report (exit {run.returncode})')
        report = json.loads(smoke.read_text(encoding='utf-8'))
        if run.returncode or not report.get('ok') or not report.get('frozen') or report.get('bits') != 64:
            raise RuntimeError(json.dumps(report, ensure_ascii=False))
        if report['machine'].lower() != machine:
            raise RuntimeError(f'Packaged process architecture mismatch: {report["machine"]}')

        destination = output / archive.name
        shutil.copyfile(archive, destination)
        with destination.open('rb') as file:
            digest = hashlib.file_digest(file, 'sha256').hexdigest()
        (output / (archive.name + '.sha256')).write_text(f'{digest}  {archive.name}\n', encoding='ascii')
        report.update(info, sha256=digest)
        (output / (name + '-smoke.json')).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(f'{destination.name}: {destination.stat().st_size / (1024*1024):.1f} MiB, SHA-256 {digest}')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True, choices=('macos-arm64', 'macos-intel', 'windows-x64'))
    parser.add_argument('--output', type=Path, default=ROOT / 'packages')
    args = parser.parse_args()
    build(args.target, args.output.resolve())
