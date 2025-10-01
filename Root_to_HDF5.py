import argparse
import sys
import time
from pathlib import Path

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


def main(args):
    chunk_size = args.chunk_size
    
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

    file_path = Path(args.infile)
    if not file_path.exists():
        sys.exit(f"[error] Could not open {args.infile}")

    print(f" >> Input file:  {args.infile}")

    with uproot.open(args.infile, basketcache="0 B") as f:
        tree = f["fevt/RHTree"]

        # pre-selection
        diphoton_m0 = tree["A_diphoton_gen_m0"].array(library="np")
        valid_mask = np.fromiter((len(x) > 0 for x in diphoton_m0), dtype=bool)
        if args.mask_path is not None:
            mask = np.load(args.mask_path).astype(bool)
            valid_mask = np.logical_and(valid_mask, mask)
        valid_indices = np.nonzero(valid_mask)[0]
        nGood = valid_indices.size
        nEvts = tree.num_entries
        if nGood == 0:
            sys.exit("[error] No events survived pre-selection.")
        print(
            f" >> nEvts: {nEvts}\n >> {nGood}/{nEvts} events "
            f"({100 * nGood / nEvts:.1f} %) after pre-selection"
        )

        # shapes
        HIT_SHAPE = (8192, 32, 3)
        SCALAR_SHAPE = (1,)

        out_path = f"{args.outdir}/{args.decay}_{args.idx}.h5"
        print(f" >> Output file: {out_path}")

        overflow_counter = np.zeros(32, dtype=np.int64)

        hit_matrix = np.empty(HIT_SHAPE, dtype=np.float32)
        BUFFER_ROWS = chunk_size
        hit_buffer = np.empty((BUFFER_ROWS, *HIT_SHAPE), dtype=np.float32)
        scalar_buffers = {
            n: np.empty((BUFFER_ROWS, 1), dtype=np.float32) for n in scalar_branches
        }

        start_time = time.time()

        with h5py.File(out_path, "w") as proper_data:
            dsets = {}
            for name in scalar_branches + ["zero_suppressed_hit_collection"]:
                shape = (
                    (nGood, *HIT_SHAPE)
                    if name == "zero_suppressed_hit_collection"
                    else (nGood, *SCALAR_SHAPE)
                )
                chunks = (
                    (chunk_size, *HIT_SHAPE)
                    if name == "zero_suppressed_hit_collection"
                    else (chunk_size, *SCALAR_SHAPE)
                )
                dsets[name] = proper_data.create_dataset(
                    name,
                    shape=shape,
                    dtype="float32",
                    compression="lzf",
                    chunks=chunks,
                )

            write_head = 0
            buf_fill = 0

            for arrays in tree.iterate(
                    branches_needed,
                    step_size=BUFFER_ROWS,               # 32 events per chunk
                    library="np",
                    decompression_executor=TrivialExecutor(),  # single-threaded decompression
                ):
                m0 = arrays["A_diphoton_gen_m0"]
                good_mask = np.fromiter((len(x) > 0 for x in m0), dtype=bool)
                if not good_mask.any():
                    continue
                # narrow arrays to selected events only
                for k in arrays:
                    arrays[k] = arrays[k][good_mask]

                for li in range(len(arrays["ECAL_energy"])):
                    if (write_head + buf_fill) % 100 == 0:
                        print(
                            f" .. processing {write_head + buf_fill}/{nGood} "
                        )

                    hit_matrix.fill(-np.inf)

                    # ECAL energy
                    ecal_energy = arrays["ECAL_energy"][li]
                    if compress_channel(
                        ecal_energy,
                        ECAL_etaphi,
                        hit_matrix[:, 30, :],
                        k=8192,
                        thresh=1e-10,
                    ):
                        overflow_counter[30] += 1

                    # HBHE energy
                    hbhe_energy = arrays["HBHE_energy"][li]
                    if compress_channel(
                        hbhe_energy,
                        HCAL_etaphi,
                        hit_matrix[:, 31, :],
                        k=8192,
                        thresh=1e-10,
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

                    hit_buffer[buf_fill] = hit_matrix
                    buf_fill += 1

                    if buf_fill == BUFFER_ROWS:
                        slice_ = slice(write_head, write_head + buf_fill)
                        dsets["zero_suppressed_hit_collection"][slice_] = hit_buffer[:buf_fill]
                        for n in scalar_branches:
                            dsets[n][slice_] = scalar_buffers[n][:buf_fill]
                        write_head += buf_fill
                        buf_fill = 0
                arrays.clear()

            # trailing rows
            if buf_fill:
                slice_obj = slice(write_head, write_head + buf_fill)
                dsets["zero_suppressed_hit_collection"][slice_obj] = hit_buffer[:buf_fill]
                for n in scalar_branches:
                    dsets[n][slice_obj] = scalar_buffers[n][:buf_fill]
                write_head += buf_fill
            assert write_head == nGood

        elapsed = (time.time() - start_time) / 60.0
        print(f" >> Wall time: {elapsed:.1f} minutes")

    print("========================================================")
    print("=== Overflow summary (events with >8192 non-zero hits) ===")
    channel_names = [
        "TrkPt",
        "BPIX_L1",
        "BPIX_L2",
        "BPIX_L3",
        "BPIX_L4",
        "FPIX_L1",
        "FPIX_L2",
        "FPIX_L3",
        "TIB_L1",
        "TIB_L2",
        "TIB_L3",
        "TIB_L4",
        "TID_L1",
        "TID_L2",
        "TID_L3",
        "TOB_L1",
        "TOB_L2",
        "TOB_L3",
        "TOB_L4",
        "TOB_L5",
        "TOB_L6",
        "TEC_L1",
        "TEC_L2",
        "TEC_L3",
        "TEC_L4",
        "TEC_L5",
        "TEC_L6",
        "TEC_L7",
        "TEC_L8",
        "TEC_L9",
        "ECAL_E",
        "HBHE_E",
    ]
    for ch, cnt in enumerate(overflow_counter):
        print(f"{channel_names[ch]:>8s}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROOT -> HDF5 converter (uproot)")
    parser.add_argument("-i", "--infile", default="output_qqgg.root")
    parser.add_argument("-o", "--outdir", default=".")
    parser.add_argument("-d", "--decay", default="test")
    parser.add_argument("-n", "--idx", type=int, default=0)
    parser.add_argument("-c", "--chunk_size", type=int, default=32)
    parser.add_argument("-m", "--mask_path", type=str, default=None)
    main(parser.parse_args())
