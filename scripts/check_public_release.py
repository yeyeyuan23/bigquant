"""Check the source-only public snapshot without importing research modules.

This is a regression guard, not a guarantee that arbitrary content is safe.
It reads tracked files in a Git checkout, or all files in an exported tree.
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
from pathlib import Path
import re
import subprocess

BINARY_SUFFIXES = {
    '.parquet', '.pt', '.pth', '.ckpt', '.npz', '.npy', '.pkl', '.pickle',
    '.joblib', '.feather', '.arrow', '.gz', '.zip', '.tar', '.zst', '.7z',
}
ROW_KEYS = {'date', 'trade_date', 'signal_date', 'return_date', 'instrument'}
PRIVATE_PATTERNS = (
    re.compile(r'/' + r'Users/[A-Za-z0-9_.-]+/'),
    re.compile(r'/' + r'home/[A-Za-z0-9_.-]+/'),
    re.compile(r'/' + r'root/'),
    re.compile(r'root' + r'@[A-Za-z0-9]'),
    re.compile(r'\bautodl-container-[a-f0-9]'),
    re.compile(r'https://github\.com/yeyeyuan23/bigquant/(?:releases/(?:download|tag)/|(?:blob|tree|commit)/[0-9a-f]{7,40})'),
)
PAYLOAD_NAME = re.compile(r'(?:CHECKPOINT|MODEL|WEIGHT).*?(?:B85|B64|BASE64|BASE85)|(?:B85|B64).*?(?:CHECKPOINT|MODEL|WEIGHT)', re.I)


def _paths(root: Path) -> list[Path]:
    if (root / '.git').exists():
        result = subprocess.run(
            ['git', '-C', str(root), 'ls-files', '-z'],
            check=True, stdout=subprocess.PIPE,
        )
        return [root / p.decode() for p in result.stdout.split(b'\0') if p]
    return sorted(p for p in root.rglob('*') if (p.is_file() or p.is_symlink())
                  and not {'.git', '__pycache__', '.ruff_cache', '.pytest_cache'}
                  .intersection(p.relative_to(root).parts))


def _has_weights(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == 'state_dict' and child:
                return True
            if _has_weights(child):
                return True
    elif isinstance(value, list):
        return any(_has_weights(child) for child in value)
    return False


def _private_json_string(value: object) -> bool:
    if isinstance(value, str):
        return any(pattern.search(value) for pattern in PRIVATE_PATTERNS)
    if isinstance(value, dict):
        return any(_private_json_string(key) or _private_json_string(child)
                   for key, child in value.items())
    if isinstance(value, list):
        return any(_private_json_string(child) for child in value)
    return False


def _python_payload(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if any(PAYLOAD_NAME.search(name) for name in names):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (str, bytes)) and len(node.value.value) > 1000:
                    return True
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            if len(node.value) > 100_000:
                # Generated Candidate454 modules contain readable algorithm source.
                # Parse it recursively; never execute it or exempt nested payloads.
                try:
                    embedded = ast.parse(node.value)
                except (SyntaxError, ValueError, UnicodeDecodeError):
                    return True
                if not any(isinstance(item, (ast.FunctionDef, ast.ClassDef, ast.Import,
                                             ast.ImportFrom)) for item in embedded.body):
                    return True
                if _python_payload(node.value):
                    return True
    return False


def scan_tree(root: Path) -> list[str]:
    root = root.resolve()
    problems = []
    paths = _paths(root)
    published = {root}
    for path in paths:
        published.add(path)
        published.update(parent for parent in path.parents if parent.is_relative_to(root))
    for path in paths:
        name = str(path.relative_to(root))
        if path.is_symlink():
            try:
                target = path.readlink()
                resolved = path.resolve(strict=True)
                cursor = path.parent
                public_target = not target.is_absolute()
                for index, part in enumerate(target.parts):
                    cursor = cursor.parent if part == '..' else cursor / part
                    # Intermediate links can change the meaning of later '..'.
                    if cursor not in published or (cursor.is_symlink() and index < len(target.parts) - 1):
                        public_target = False
                        break
                if not public_target or not resolved.is_relative_to(root) or resolved not in published:
                    problems.append(f'{name}: symlink target is outside the published source tree')
            except (OSError, RuntimeError):
                problems.append(f'{name}: dangling or cyclic symlink')
            continue
        if not path.exists():
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            problems.append(f'{name}: data/model/archive binary is not public source')
            continue
        try:
            source = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            problems.append(f'{name}: unreviewed binary file')
            continue
        if any(pattern.search(source) for pattern in PRIVATE_PATTERNS):
            problems.append(f'{name}: private machine path/login or historical artifact URL')
        if path.suffix == '.csv':
            header = next(csv.reader(io.StringIO(source)), [])
            if ROW_KEYS.intersection(header) or path.name.endswith('_weights.csv'):
                problems.append(f'{name}: market rows/daily series or trained coefficients')
        if path.suffix == '.txt' and source.startswith('tree\nversion='):
            problems.append(f'{name}: trained tree model')
        if path.suffix == '.py':
            try:
                if _python_payload(source):
                    problems.append(f'{name}: embedded model-sized payload')
            except SyntaxError as exc:
                problems.append(f'{name}: invalid Python at line {exc.lineno}')
        if path.suffix in {'.json', '.ipynb'}:
            try:
                document = json.loads(source)
                if _private_json_string(document) and not any(
                    pattern.search(source) for pattern in PRIVATE_PATTERNS
                ):
                    problems.append(f'{name}: escaped private path/login or historical artifact URL')
                if _has_weights(document):
                    problems.append(f'{name}: trained JSON state_dict')
                if path.suffix == '.ipynb':
                    for cell in document.get('cells', []):
                        if cell.get('outputs') or cell.get('execution_count') is not None:
                            problems.append(f'{name}: saved notebook execution output')
                            break
                        if cell.get('cell_type') == 'code':
                            code = ''.join(cell.get('source', []))
                            try:
                                if _python_payload(code):
                                    problems.append(f'{name}: embedded notebook payload')
                            except SyntaxError:
                                # IPython shell/magic syntax is legitimate notebook source.
                                pass
            except json.JSONDecodeError as exc:
                problems.append(f'{name}: invalid JSON at line {exc.lineno}')
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    problems = scan_tree(args.root.resolve())
    for problem in problems:
        print(problem)
    if problems:
        print(f'FAILED: {len(problems)} publication-policy findings')
        return 1
    print('PASS: no prohibited artifact, private-path or notebook-output pattern found')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
