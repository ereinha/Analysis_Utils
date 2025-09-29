import argparse
import numpy as np
import ROOT
from pathlib import Path

# ----------------------------------------------------------------------
# FWLite bootstrap (only once, no matter which mode)
# ----------------------------------------------------------------------
ROOT.gSystem.Load("libFWCoreFWLite")
ROOT.FWLiteEnabler.enable()
from DataFormats.FWLite import Events, Handle


def compute_pass_rates(fname: str) -> list[tuple[str, float, int, int]]:
    """
    Compute pass rate for each HLT_ path in the file.
    Returns a list of tuples: (path, rate, passed, total)
    """
    # First, get list of HLT_ paths
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

    # sort descending by pass rate
    return sorted(stats, key=lambda x: x[1], reverse=True)
def save_mask(fname: str, hlt_path: str, outname: str) -> None:
    """Build & save a boolean mask for one trigger path."""
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


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="FWLite helper: list HLT_ paths by pass rate or dump trigger masks")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--list", action="store_true",
                   help="List all HLT_ paths sorted by pass rate and exit")
group.add_argument("-a", "--all", action="store_true",
                   help="Dump masks for all HLT_ triggers into separate .npy files, ordered by pass rate")
group.add_argument("-t", "--hlt_path", nargs="?",
                   help="Single HLT_ path name (e.g. HLT_IsoMu24_v)")

parser.add_argument("-i", "--input",  required=True,
                    help="input EDM/ROOT file (NanoAOD, MiniAOD, etc.)")
parser.add_argument("-o", "--output", nargs="?",
                    help="output .npy file for the mask (only for single-path mode)")

args = parser.parse_args()
if args.list:
    stats = compute_pass_rates(args.input)
    if stats:
        print(f"{args.input}: HLT_ paths sorted by pass rate:\n")
        for path, rate, passed, total in stats:
            print(f"{path}: {passed}/{total} = {rate:.2%}")
    else:
        print(f"{args.input}: no HLT_ TriggerResults found")
elif args.all:
    stats = compute_pass_rates(args.input)
    if not stats:
        print(f"{args.input}: no HLT_ TriggerResults found; nothing to do")
    else:
        for path, rate, passed, total in stats:
            stem = Path(path).stem
            out  = f"{stem}_mask.npy"
            save_mask(args.input, path, out)
else:
    # single-path mask mode
    stem = Path(args.hlt_path).stem
    out  = args.output or f"{stem}_mask.npy"
    save_mask(args.input, args.hlt_path, out)
