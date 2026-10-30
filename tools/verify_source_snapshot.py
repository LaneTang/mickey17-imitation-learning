"""Verify the included NUC source snapshot without importing robot code."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "source_provenance.json").read_text())
    errors = []
    for relative_path, expected_hash in manifest["nuc_files"].items():
        path = root / relative_path
        if not path.is_file():
            errors.append(f"Missing: {relative_path}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            errors.append(f"Changed: {relative_path}")
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"Verified {len(manifest['nuc_files'])} NUC source/configuration files.")


if __name__ == "__main__":
    main()
