"""Verify the included NUC source snapshot without importing robot code."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "source_provenance.json").read_text())
    extensions = json.loads((root / "local_extensions.json").read_text())["files"]
    relocations = json.loads((root / "source_relocations.json").read_text())["files"]
    errors = []
    unchanged = extended = 0
    for relative_path, expected_hash in manifest["nuc_files"].items():
        path = root / relocations.get(relative_path, {}).get("path", relative_path)
        if not path.is_file():
            errors.append(f"Missing: {relative_path}")
        else:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual == expected_hash:
                unchanged += 1
            elif relative_path in extensions:
                extension = extensions[relative_path]
                if (extension["original_nuc_sha256"] == expected_hash and
                        extension["local_sha256"] == actual):
                    extended += 1
                    print(f"Verified recorded local extension (not NUC bytes): {relative_path}")
                else:
                    errors.append(f"Unrecorded local change: {relative_path}")
            else:
                errors.append(f"Changed: {relative_path}")
    for path in extensions:
        if path not in manifest["nuc_files"]:
            errors.append(f"Extension has no original NUC entry: {path}")
    for original, relocation in relocations.items():
        path = root / relocation["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != relocation["sha256"]:
            errors.append(f"Relocated source changed or missing: {original}")
        if (root / original).exists():
            errors.append(f"Old source path unexpectedly exists: {original}")
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"Verified {unchanged} unchanged NUC files and {extended} recorded local extensions.")


if __name__ == "__main__":
    main()
