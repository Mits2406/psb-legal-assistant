"""Build the complete final deliverable locally, without upload or credentials."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    'README.md', 'Dockerfile', 'compose.yaml', '.dockerignore', '.gitignore',
    '.env.example', 'requirements.txt', '.gitattributes',
}
DIRECTORIES = {'static', 'tests', 'scripts', 'docs', 'data', 'evaluation', 'artifacts', '.github'}
REQUIRED = {
    *ROOT_FILES, 'app.py', 'access_control.py', 'scoped_assistant.py', 'solution.py', 'precomputed_index.py',
    'docs/presentation.pdf', 'docs/architecture.md', 'docs/evaluation.md',
    'docs/review-checklist.md', 'artifacts/submission-v20.zip',
    'data/knowledge-base.zip', 'data/demo_questions.csv', 'static/index.html',
}


def collect() -> dict[str, bytes]:
    files = {}
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        parts = relative.parts
        if parts[0] not in DIRECTORIES and not (
            len(parts) == 1 and (path.name in ROOT_FILES or path.suffix == '.py')
        ):
            continue
        if any(p in {'__pycache__', '.DS_Store', '.venv', '.git', 'tmp', 'state'} for p in parts):
            continue
        if path.is_symlink():
            raise ValueError(f'Symlinks must not enter the final package: {relative}')
        if not path.is_file() or path.suffix in {'.pyc', '.pyo'}:
            continue
        if path.name.startswith('.env') and str(relative) != '.env.example':
            raise ValueError(f'Environment file must not enter the package: {relative}')
        if (path.suffix in {'.key', '.pem', '.sqlite', '.sqlite3', '.db'}
                or 'credentials' in path.name.lower() or path.name.lower().startswith('token')
                or '.sqlite-' in path.name.lower()):
            raise ValueError(f'Potential credential file: {relative}')
        files[relative.as_posix()] = path.read_bytes()
    missing = REQUIRED - files.keys()
    if missing:
        raise ValueError(f'Missing final materials: {sorted(missing)}')
    if not files['docs/presentation.pdf'].startswith(b'%PDF-'):
        raise ValueError('Presentation is not a PDF')
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/psb-final-v20.zip')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT) and output.relative_to(ROOT).parts[0] in DIRECTORIES:
        parser.error('Write the final archive to dist/ or outside the source tree.')
    files = collect()
    manifest = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n' for name, data in files.items())
    files['MANIFEST.sha256'] = manifest.encode('utf-8')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w') as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo('psb-final-v20/' + name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC validation failed')
    print(f'Final package: {output}\nFiles: {len(files)}\nBytes: {output.stat().st_size}')
    print('SHA256: ' + hashlib.sha256(output.read_bytes()).hexdigest())
    print('Nothing was uploaded or submitted.')


if __name__ == '__main__':
    main()
