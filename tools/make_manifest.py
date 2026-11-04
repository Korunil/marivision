"""Write MANIFEST.sha256 for scripts/ and docs/, and refuse dataset-derived files.  Run from the repo root."""
from pathlib import Path
import hashlib, sys

FORBIDDEN = {".zip", ".png", ".jpg", ".jpeg", ".gz", ".pth", ".pt", ".ckpt", ".safetensors"}
MAX_BYTES = 2 * 2 ** 20          # a 2 MB text file in a code repo is suspicious: check it by hand

def main():
    root = Path(".").resolve()
    bad, lines = [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or ".git" in p.parts:
            continue
        rel = p.relative_to(root).as_posix()
        if p.suffix.lower() in FORBIDDEN or p.name.startswith("seaclips-") or p.stat().st_size > MAX_BYTES:
            bad.append(rel)
        if rel.startswith(("scripts/", "docs/")):
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {rel}")
    if bad:
        sys.exit("REFUSED -- remove before committing:\n  " + "\n  ".join(bad))
    Path("MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"MANIFEST.sha256: {len(lines)} files; no dataset-derived files found")

if __name__ == "__main__":
    main()
