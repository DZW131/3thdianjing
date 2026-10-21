"""Download a published weight, or securely copy an authorized private artifact."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "docs/weights_inventory.json")
    p.add_argument("--path", help="Exact relative path from the inventory")
    p.add_argument("--list", action="store_true")
    p.add_argument("--output-root", type=Path, default=Path("."))
    p.add_argument("--ssh", help="Authorized user@host, if no public URL exists")
    p.add_argument("--remote-root", help="Remote project directory for SCP")
    p.add_argument("--port", type=int, default=22)
    args = p.parse_args()
    rows = json.loads(args.manifest.read_text())
    if args.list:
        for r in rows:
            print(r["path"], r["size_bytes"], "published" if r.get("url") else "private/SCP required")
        return
    found = [r for r in rows if r["path"] == args.path]
    if len(found) != 1:
        p.error("Use --list and choose an exact --path")
    row = found[0]
    dest = (args.output_root / row["path"]).resolve()
    if not dest.is_relative_to(args.output_root.resolve()):
        p.error("Unsafe inventory path")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        with dest.open("rb") as f:
            h = hashlib.sha256()
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() == row["sha256"]:
            print("Already verified:", dest)
            return
        raise RuntimeError("Existing file differs; move it aside explicitly before downloading")
    temp = dest.with_name(dest.name + ".part")
    if row.get("url"):
        urllib.request.urlretrieve(row["url"], temp)
    elif args.ssh and args.remote_root:
        subprocess.run(["scp", "-P", str(args.port), "--",
                        args.ssh + ":" + args.remote_root.rstrip("/") + "/" + row["path"], str(temp)], check=True)
    else:
        p.error("Not published: ask the data/model custodian for access, then supply --ssh and --remote-root")
    h = hashlib.sha256()
    with temp.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    if temp.stat().st_size != row["size_bytes"] or h.hexdigest() != row["sha256"]:
        raise RuntimeError("Downloaded artifact failed size/SHA-256 verification; .part retained")
    temp.rename(dest)
    print("Verified:", dest)


if __name__ == "__main__":
    main()
