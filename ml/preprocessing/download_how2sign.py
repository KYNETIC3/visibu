"""Download the How2Sign files Visibu uses, from the links in the official
download script (https://github.com/how2sign/how2sign.github.io/blob/main/download_how2sign.sh).

How2Sign is CC BY-NC 4.0: research / non-commercial use only. You must pass
--accept-licence to confirm you have read https://how2sign.github.io and accept it.
Downloaded data stays in ml/datasets/how2sign (git-ignored). Never commit or
redistribute it.

  python -m preprocessing.download_how2sign --accept-licence annotations
  python -m preprocessing.download_how2sign --accept-licence clips --split val
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
ROOT = ML_ROOT / "datasets" / "how2sign"

# Google Drive ids copied from the official download_how2sign.sh.
ANNOTATIONS = {
    "train": "1dUHSoefk9OxKJnHrHPX--I4tpm9QD0ok",
    "val": "1Vpag7VPfdTCCJSao8Pz14rlPfekRMggI",
    "test": "1AgwBZW26kFHS4CWNMQTCMPGkBPkH3qCu",
}
FRONT_CLIPS = {  # sizes as reported by Google Drive, Nov 2026
    "train": ("1VX7n0jjW0pW3GEdgOks3z8nqE6iI6EnW", "31G"),
    "val": ("1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO", "1.7G"),
    "test": ("1qTIXFsu8M55HrCiaGv7vZ7GkdB3ubjaG", "2.2G"),
}


def fetch(drive_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.usercontent.google.com/download?id={drive_id}&export=download&confirm=t"
    print(f"Downloading {dest.name} ...", flush=True)
    subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(dest), url], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["annotations", "clips"])
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    ap.add_argument("--accept-licence", action="store_true")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()
    if not args.accept_licence:
        sys.exit("How2Sign is CC BY-NC 4.0 (non-commercial). Re-run with --accept-licence once you have read it.")

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for s in splits:
        if args.what == "annotations":
            fetch(ANNOTATIONS[s], ROOT / "annotations" / f"how2sign_realigned_{s}.csv")
            continue
        drive_id, size = FRONT_CLIPS[s]
        z = ROOT / "raw" / f"{s}_rgb_front_clips.zip"
        print(f"{s} front clips are about {size}.")
        if not z.exists() or not zipfile.is_zipfile(z):
            fetch(drive_id, z)
        if not zipfile.is_zipfile(z):
            head = z.read_bytes()[:4000].decode("utf-8", "replace")
            z.unlink()
            reason = "Google Drive download quota exceeded" if "Quota exceeded" in head else "not a zip file"
            sys.exit(f"{s} clips: {reason}. Open https://drive.google.com/file/d/{drive_id} in a signed-in browser, "
                     f"download it (or 'Add shortcut to Drive' and download from your Drive), save it as {z}, "
                     "then run this command again. Retrying after 24 hours can also work.")
        out = ROOT / "clips" / s
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            members = [m for m in zf.namelist() if m.endswith(".mp4")]
            print(f"Unzipping {len(members)} clips to {out} ...", flush=True)
            for m in members:
                target = out / Path(m).name  # flatten; never trust archive paths
                if not target.exists():
                    with zf.open(m) as src, open(target, "wb") as dst:
                        while chunk := src.read(1 << 20):
                            dst.write(chunk)
        if not args.keep_zip:
            z.unlink()


if __name__ == "__main__":
    main()
