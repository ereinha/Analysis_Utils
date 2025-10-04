import argparse
import sys
import time
from pathlib import Path
import re

import h5py
import numpy as np
import uproot
from uproot.source.futures import TrivialExecutor


def compress_channel(values, coords, out, *, k=8192, thresh=1e-10):
    mask = values > thresh
    if not mask.any():
        out[:] = -np.inf
        return False
    vals = values[mask]
    coord = coords[mask]
    if vals.size > k:
        top_idx = np.argpartition(-vals, k - 1)[:k]
        vals, coord = vals[top_idx], coord[top_idx]
        overflow = True
    else:
        overflow = False
    n = vals.size
    out[:n, 0] = vals.astype(np.float32, copy=False)
    out[:n, 1:] = coord.astype(np.float32, copy=False)
    if n < k:
        out[n:, :] = -np.inf
    return overflow


def compress_highres(triplets, out, *, k=8192, thresh=1e-10, pad_val=-999.0):
    values = triplets[:, 0]
    mask = (values > thresh) & (values > pad_val + 1.0)
    if not mask.any():
        out[:] = -np.inf
        return False
    vals = values[mask]
    if vals.size > k:
        top_idx = np.argpartition(-vals, k - 1)[:k]
        vals = vals[top_idx]
        overflow = True
    else:
        overflow = False
    n = vals.size
    out[:n] = vals.astype(np.float32, copy=False)
    if n < k:
        out[n:] = -np.inf
    out[:k] = np.sort(out[:k])[::-1]
    return overflow


def _natural_key(p: Path):
    # Sort like 1,2,10 instead of 1,10,2
    parts = re.split(r'(\d+)', p.name)
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def _sorted_files_in_dir(d: Path, suffix: str):
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix == suffix], key=_natural_key)


def main(args):
    chunk_size = args.chunk_size
    base_idx = args.idx

    # geometry constants
    HCAL_eta = np.linspace(-3, 3, 56, dtype=np.float32).reshape(56, 1)
    HCAL_phi = np.linspace(-np.pi, np.pi, 72, dtype=np.float32).reshape(1, 72)
    HCAL_etaphi = np.stack(
        (np.broadcast_to(HCAL_eta, (56, 72)), np.broadcast_to(HCAL_phi, (56, 72))),
        axis=-1,
    ).reshape(-1, 2)

    ECAL_eta = np.linspace(-3, 3, 280, dtype=np.float32).reshape(280, 1)
    ECAL_phi = np.linspace(-np.pi, np.pi, 360, dtype=np.float32).reshape(1, 360)
    ECAL_etaphi = np.stack(
        (np.broadcast_to(ECAL_eta, (280, 360)), np.broadcast_to(ECAL_phi, (280, 360))),
        axis=-1,
    ).reshape(-1, 2)

    # branch lists
    scalar_branches = [
        # diphoton - gen
        "A_diphoton_gen_m0",
        "A_diphoton_gen_dR",
        "A_diphoton_gen_E",
        "A_diphoton_gen_pT",
        "A_diphoton_gen_eta",
        "A_diphoton_gen_phi",
        # diphoton - reco
        "A_diphoton_reco_M",
        "A_diphoton_reco_dR",
        "A_diphoton_reco_E",
        "A_diphoton_reco_pT",
        "A_diphoton_reco_eta",
        "A_diphoton_reco_phi",
        # ditau - gen
        "A_ditau_gen_m0",
        "A_ditau_gen_dR",
        "A_ditau_gen_E",
        "A_ditau_gen_pT",
        "A_ditau_gen_eta",
        "A_ditau_gen_phi",
    ]

    HIGHRES_BRANCHES = [
        "ECAL_tracksPt_triplet_atECALfixIP",
        "BPIX_layer1_triplets_atPV",
        "BPIX_layer2_triplets_atPV",
        "BPIX_layer3_triplets_atPV",
        "BPIX_layer4_triplets_atPV",
        "FPIX_layer1_triplets_atPV",
        "FPIX_layer2_triplets_atPV",
        "FPIX_layer3_triplets_atPV",
        "TIB_layer1_triplets_atPV",
        "TIB_layer2_triplets_atPV",
        "TIB_layer3_triplets_atPV",
        "TIB_layer4_triplets_atPV",
        "TID_layer1_triplets_atPV",
        "TID_layer2_triplets_atPV",
        "TID_layer3_triplets_atPV",
        "TOB_layer1_triplets_atPV",
        "TOB_layer2_triplets_atPV",
        "TOB_layer3_triplets_atPV",
        "TOB_layer4_triplets_atPV",
        "TOB_layer5_triplets_atPV",
        "TOB_layer6_triplets_atPV",
        "TEC_layer1_triplets_atPV",
        "TEC_layer2_triplets_atPV",
        "TEC_layer3_triplets_atPV",
        "TEC_layer4_triplets_atPV",
        "TEC_layer5_triplets_atPV",
        "TEC_layer6_triplets_atPV",
        "TEC_layer7_triplets_atPV",
        "TEC_layer8_triplets_atPV",
        "TEC_layer9_triplets_atPV",
    ]

    extra_branches = ["ECAL_energy", "HBHE_energy"]
    branches_needed = scalar_branches + HIGHRES_BRANCHES + extra_branches

    in_path = Path(args.infile)
    if not in_path.exists():
        sys.exit(f"[error] Could not open {args.infile}")

    # Build file lists
    if in_path.is_dir():
        root_files = _sorted_files_in_dir(in_path, ".root")
        if not root_files:
            sys.exit(f"[error] No .root files in {in_path}")
    else:
        root_files = [in_path]

    # Build mask list
    mask_list = [None] * len(root_files)
    if args.mask_path is not None:
        mpath = Path(args.mask_path)
        if mpath.is_dir():
            masks = _sorted_files_in_dir(mpath, ".npy")
            if len(masks) != len(root_files):
                sys.exit(f"[error] mask count ({len(masks)}) != file count ({len(root_files)})")
            mask_list = masks
        else:
            if len(root_files) != 1:
                sys.exit("[error] Single mask file provided but multiple ROOT files detected")
            mask_list = [mpath]

    print(f" >> Inputs: {len(root_files)} file(s)")
    for i, p in enumerate(root_files):
        print(f"    [{i}] {p}")

    # Pre-count pass to size HDF5
    total_good = 0
    per_file_good = []
    per_file_nev = []

    for fi, rf in enumerate(root_files):
        with uproot.open(rf, basketcache="0 B") as f:
            tree = f["fevt/RHTree"]
            nEvts = tree.num_entries
            per_file_nev.append(nEvts)

            # optional mask
            msk = None
            if mask_list[fi] is not None:
                msk = np.load(mask_list[fi]).astype(bool)
                if msk.shape[0] != nEvts:
                    sys.exit(f"[error] mask length {msk.shape[0]} != nEvts {nEvts} for {mask_list[fi]}")

            n_good = 0
            offset = 0
            for arrays in tree.iterate(
                ["A_diphoton_gen_m0"],
                step_size=args.chunk_size,
                library="np",
                decompression_executor=TrivialExecutor(),
            ):
                m0 = arrays["A_diphoton_gen_m0"]
                cl = len(m0)
                good = np.fromiter((len(x) > 0 for x in m0), dtype=bool)
                if msk is not None:
                    good &= msk[offset:offset + cl]
                n_good += int(good.sum())
                offset += cl
                arrays.clear()

            per_file_good.append(n_good)
            total_good += n_good

        print(f" >> {rf.name}: nEvts={nEvts}  nGood={n_good} ({0 if nEvts==0 else 100*n_good/nEvts:.1f} %)")

    if total_good == 0:
        sys.exit("[error] No events survived pre-selection across all inputs.")

    # shapes
    HIT_SHAPE = (8192, 32, 3)
    SCALAR_SHAPE = (1,)

    # Output name
    if in_path.is_dir():
        out_path = f"{args.outdir}/{args.decay}_merged.h5"
    else:
        out_path = f"{args.outdir}/{args.decay}_{args.idx}.h5"

    print(f" >> Output file: {out_path}")

    overflow_counter = np.zeros(32, dtype=np.int64)
    hit_matrix = np.empty(HIT_SHAPE, dtype=np.float32)
    BUFFER_ROWS = chunk_size
    hit_buffer = np.empty((BUFFER_ROWS, *HIT_SHAPE), dtype=np.float32)
    scalar_buffers = {n: np.empty((BUFFER_ROWS, 1), dtype=np.float32) for n in scalar_branches}
    file_index_buffer = np.empty((BUFFER_ROWS, 1), dtype=np.int32)

    start_time = time.time()

    with h5py.File(out_path, "w") as proper_data:
        dsets = {}

        # Create datasets
        dsets["zero_suppressed_hit_collection"] = proper_data.create_dataset(
            "zero_suppressed_hit_collection",
            shape=(total_good, *HIT_SHAPE),
            dtype="float32",
            compression="lzf",
            chunks=(chunk_size, *HIT_SHAPE),
        )
        for name in scalar_branches:
            dsets[name] = proper_data.create_dataset(
                name,
                shape=(total_good, *SCALAR_SHAPE),
                dtype="float32",
                compression="lzf",
                chunks=(chunk_size, *SCALAR_SHAPE),
            )
        # per‑event file index
        dsets["file_index"] = proper_data.create_dataset(
            "file_index",
            shape=(total_good, *SCALAR_SHAPE),
            dtype="int32",
            compression="lzf",
            chunks=(chunk_size, *SCALAR_SHAPE),
        )

        write_head = 0
        buf_fill = 0

        # Second pass: fill
        for fi, rf in enumerate(root_files):
            file_idx_value = base_idx + fi
            with uproot.open(rf, basketcache="0 B") as f:
                tree = f["fevt/RHTree"]

                # file mask
                msk = None
                if mask_list[fi] is not None:
                    msk = np.load(mask_list[fi]).astype(bool)

                offset = 0
                for arrays in tree.iterate(
                    branches_needed,
                    step_size=BUFFER_ROWS,
                    library="np",
                    decompression_executor=TrivialExecutor(),
                ):
                    m0 = arrays["A_diphoton_gen_m0"]
                    cl = len(m0)

                    good_mask = np.fromiter((len(x) > 0 for x in m0), dtype=bool)
                    if msk is not None:
                        good_mask &= msk[offset:offset + cl]

                    offset += cl
                    if not good_mask.any():
                        arrays.clear()
                        continue

                    # narrow arrays to selected events only
                    for k in arrays:
                        arrays[k] = arrays[k][good_mask]

                    for li in range(len(arrays["ECAL_energy"])):
                        if (write_head + buf_fill) % 100 == 0:
                            print(f" .. processing {write_head + buf_fill}/{total_good}")

                        hit_matrix.fill(-np.inf)

                        # ECAL energy
                        ecal_energy = arrays["ECAL_energy"][li]
                        if compress_channel(
                            ecal_energy, ECAL_etaphi, hit_matrix[:, 30, :], k=8192, thresh=1e-10
                        ):
                            overflow_counter[30] += 1

                        # HBHE energy
                        hbhe_energy = arrays["HBHE_energy"][li]
                        if compress_channel(
                            hbhe_energy, HCAL_etaphi, hit_matrix[:, 31, :], k=8192, thresh=1e-10
                        ):
                            overflow_counter[31] += 1

                        # high-res hits
                        for ch, branch in enumerate(HIGHRES_BRANCHES):
                            triplet_np = arrays[branch][li]
                            triplet_reshaped = triplet_np.reshape(-1, 3)
                            if compress_highres(triplet_reshaped, hit_matrix[:, ch, 0], thresh=1e-10):
                                overflow_counter[ch] += 1

                        # scalars
                        for name in scalar_branches:
                            scalar_buffers[name][buf_fill, 0] = arrays[name][li][0]

                        # file index
                        file_index_buffer[buf_fill, 0] = file_idx_value

                        hit_buffer[buf_fill] = hit_matrix
                        buf_fill += 1

                        if buf_fill == BUFFER_ROWS:
                            sl = slice(write_head, write_head + buf_fill)
                            dsets["zero_suppressed_hit_collection"][sl] = hit_buffer[:buf_fill]
                            for n in scalar_branches:
                                dsets[n][sl] = scalar_buffers[n][:buf_fill]
                            dsets["file_index"][sl] = file_index_buffer[:buf_fill]
                            write_head += buf_fill
                            buf_fill = 0

                    arrays.clear()

        # trailing rows
        if buf_fill:
            sl = slice(write_head, write_head + buf_fill)
            dsets["zero_suppressed_hit_collection"][sl] = hit_buffer[:buf_fill]
            for n in scalar_branches:
                dsets[n][sl] = scalar_buffers[n][:buf_fill]
            dsets["file_index"][sl] = file_index_buffer[:buf_fill]
            write_head += buf_fill

        assert write_head == total_good

    elapsed = (time.time() - start_time) / 60.0
    print(f" >> Wall time: {elapsed:.1f} minutes")

    print("Overflow summary (events with >8192 non-zero hits)")
    channel_names = [
        "TrkPt","BPIX_L1","BPIX_L2","BPIX_L3","BPIX_L4",
        "FPIX_L1","FPIX_L2","FPIX_L3",
        "TIB_L1","TIB_L2","TIB_L3","TIB_L4",
        "TID_L1","TID_L2","TID_L3",
        "TOB_L1","TOB_L2","TOB_L3","TOB_L4","TOB_L5","TOB_L6",
        "TEC_L1","TEC_L2","TEC_L3","TEC_L4","TEC_L5","TEC_L6","TEC_L7","TEC_L8","TEC_L9",
        "ECAL_E","HBHE_E",
    ]
    for ch, cnt in enumerate(overflow_counter):
        print(f"{channel_names[ch]:>8s}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROOT -> HDF5 converter (uproot), single file or directory")
    parser.add_argument("-i", "--infile", default="output_qqgg.root",
                        help="ROOT file or a directory containing .root files")
    parser.add_argument("-o", "--outdir", default=".")
    parser.add_argument("-d", "--decay", default="test")
    parser.add_argument("-n", "--idx", type=int, default=0,
                        help="Base index added to per-file file_index (base_idx + file_number)")
    parser.add_argument("-c", "--chunk_size", type=int, default=32)
    parser.add_argument("-m", "--mask_path", type=str, default=None,
                        help="Optional .npy mask file or a directory of .npy masks (sorted to match inputs)")
    main(parser.parse_args())
