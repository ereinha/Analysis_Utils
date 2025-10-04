import argparse
import re
import numpy as np
import ROOT
from pathlib import Path

ROOT.gSystem.Load("libFWCoreFWLite")
ROOT.FWLiteEnabler.enable()
from DataFormats.FWLite import Events, Handle


def compute_pass_rates(fname: str) -> list[tuple[str, float, int, int]]:
    """
    Compute pass rate for each HLT_ path in the file.
    Returns a list of tuples: (path, rate, passed, total)
    """
    evs      = Events(fname)
    trg_h    = Handle("edm::TriggerResults")
    trg_tag  = ("TriggerResults", "", "HLT")

    paths = []
    for ev in evs:
        ev.getByLabel(trg_tag, trg_h)
        if trg_h.isValid():
            names = ev.object().triggerNames(trg_h.product())
            paths = [names.triggerName(i) for i in range(names.size())
                     if names.triggerName(i).startswith('HLT_')]
            break

    stats = []
    for path in paths:
        passed = 0
        total  = 0
        for ev in Events(fname):
            ev.getByLabel(trg_tag, trg_h)
            trg   = trg_h.product()
            names = ev.object().triggerNames(trg)
            idx   = names.triggerIndex(path)
            if idx < trg.size():
                total += 1
                if trg.accept(idx):
                    passed += 1
        rate = passed / total if total > 0 else 0.0
        stats.append((path, rate, passed, total))

    return sorted(stats, key=lambda x: x[1], reverse=True)


def save_mask(fname: str, hlt_path: str, outname: str) -> None:
    """Build & save a boolean mask for one trigger path in one file."""
    evs     = Events(fname)
    trg_h   = Handle("edm::TriggerResults")
    trg_tag = ("TriggerResults", "", "HLT")
    mask    = np.zeros(evs.size(), dtype=bool)

    for ievt, ev in enumerate(evs):
        ev.getByLabel(trg_tag, trg_h)
        trg   = trg_h.product()
        names = ev.object().triggerNames(trg)
        idx   = names.triggerIndex(hlt_path)
        if idx < trg.size():
            mask[ievt] = trg.accept(idx)

    np.save(outname, mask)
    print(f"[+] {outname} written: {mask.sum()}/{mask.size} passed '{hlt_path}'")


def _natural_key(s: str):
    """Natural sort key: splits digits so file_2 < file_10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def process_dir(indir: Path, hlt_path: str, outdir: Path, glob: str) -> None:
    """Make one mask per file. Output names start with the input stem to align lexicographic order."""
    files = sorted([p for p in indir.glob(glob) if p.is_file()], key=lambda p: _natural_key(p.name))
    if not files:
        print(f"{indir}: no files matching '{glob}'")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    hlt_stem = Path(hlt_path).stem  # keep the path name visible in outputs

    print(f"{indir}: {len(files)} files found. Writing masks to {outdir}")
    for f in files:
        out = outdir / f"{f.stem}__{hlt_stem}_mask.npy"
        save_mask(str(f), hlt_path, str(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FWLite helper: list HLT_ paths by pass rate or dump trigger masks "
                    "for a file or an entire directory (sorted).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true",
                       help="List all HLT_ paths sorted by pass rate and exit (uses a single file)")
    group.add_argument("-a", "--all", action="store_true",
                       help="Dump masks for all HLT_ triggers into separate .npy files (single file input)")
    group.add_argument("-t", "--hlt_path", nargs="?",
                       help="Single HLT_ path name (e.g. HLT_IsoMu24_v). If -i is a directory, this is applied to every file in it.")
    
    parser.add_argument("-i", "--input", required=True,
                        help="input EDM/ROOT file OR directory")
    parser.add_argument("-o", "--output", nargs="?",
                        help="single-file: output .npy for the mask; directory: output folder for masks (default: <indir>/masks_<hlt>)")
    parser.add_argument("--glob", default="*.root",
                        help="when -i is a directory, glob to pick files (default: *.root)")
    
    args = parser.parse_args()
    inp = Path(args.input)
    
    if args.list:
        # If a directory is provided, use the first matching file for discovery.
        if inp.is_dir():
            files = sorted([p for p in inp.glob(args.glob) if p.is_file()], key=lambda p: _natural_key(p.name))
            if not files:
                print(f"{inp}: no files matching '{args.glob}'")
            else:
                stats = compute_pass_rates(str(files[0]))
                if stats:
                    print(f"{files[0]}: HLT_ paths sorted by pass rate:\n")
                    for path, rate, passed, total in stats:
                        print(f"{path}: {passed}/{total} = {rate:.2%}")
                else:
                    print(f"{files[0]}: no HLT_ TriggerResults found")
        else:
            stats = compute_pass_rates(str(inp))
            if stats:
                print(f"{inp}: HLT_ paths sorted by pass rate:\n")
                for path, rate, passed, total in stats:
                    print(f"{path}: {passed}/{total} = {rate:.2%}")
            else:
                print(f"{inp}: no HLT_ TriggerResults found")
    
    elif args.all:
        if inp.is_dir():
            print("Directory + --all is not supported. Use -t/--hlt_path for per-file masks in a directory.")
        else:
            stats = compute_pass_rates(str(inp))
            if not stats:
                print(f"{inp}: no HLT_ TriggerResults found; nothing to do")
            else:
                for path, rate, passed, total in stats:
                    stem = Path(path).stem
                    out  = f"{stem}_mask.npy"
                    save_mask(str(inp), path, out)
    
    else:
        # single-path mask mode
        if not args.hlt_path:
            raise SystemExit("Missing -t/--hlt_path")
        if inp.is_dir():
            hlt_stem = Path(args.hlt_path).stem
            outdir = Path(args.output) if args.output else inp / f"masks_{hlt_stem}"
            process_dir(inp, args.hlt_path, outdir, args.glob)
        else:
            stem = Path(args.hlt_path).stem
            out  = args.output or f"{stem}_mask.npy"
            save_mask(str(inp), args.hlt_path, out)
