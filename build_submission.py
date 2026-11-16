"""Build a local archive. This script has no networking or submission code."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import zipfile

REQUIRED_FILES = ("solution.py", "answer.py", "answer_style.py", "abstention.py", "normalization.py", "clarification.py", "corpus.py", "document.py", "chunk.py", "registry.py", "query.py", "retrieval.py", "evidence.py", "generation.py", "vector_index.py", "precomputed_index.py", "requirements.txt")
OPTIONAL_FILES = ("llm.py", "policy.py")


def build_archive(project: Path, output: Path) -> list[str]:
    project = project.resolve()
    names = list(REQUIRED_FILES)
    names.extend(name for name in OPTIONAL_FILES if (project / name).is_file())
    contents: dict[str, bytes] = {}
    for name in names:
        path = project / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Required regular file is missing: {name}")
        data = path.read_bytes()
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8"), filename=name)
        contents[name] = data
    # Empty/comment-only requirements is intentional: all modules use stdlib.
    active_requirements = [
        line.strip() for line in contents["requirements.txt"].decode("utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if active_requirements:
        raise ValueError("This stdlib MVP must not add third-party requirements.")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(contents):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, contents[name])
    if output.stat().st_size > 128 * 1024 * 1024:
        output.unlink()
        raise ValueError("Submission archive exceeds 128 MiB.")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(contents):
            raise ValueError("Archive validation failed.")
    return sorted(contents)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/submission.zip"))
    args = parser.parse_args(argv)
    try:
        names = build_archive(Path(__file__).resolve().parent, args.output)
    except (OSError, ValueError, SyntaxError, UnicodeError) as exc:
        parser.error(str(exc))
    print(f"Local archive: {args.output.resolve()}")
    print("Files: " + ", ".join(names))
    print("Nothing was uploaded or submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
