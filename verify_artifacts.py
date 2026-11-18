"""Verify the exported snapshot's SHA-256 inventory (no model execution)."""
import hashlib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    failures = []
    entries = (root / 'MANIFEST.sha256').read_text(encoding='utf-8').splitlines()
    for entry in entries:
        expected, name = entry.split('  ', 1)
        path = root / name
        if not path.is_file():
            failures.append(name + ': missing')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name + ': checksum mismatch')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'Verified {len(entries)} artifacts.')


if __name__ == '__main__':
    main()
