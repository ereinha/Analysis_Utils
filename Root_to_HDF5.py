import argparse
import h5py
import sys

import ROOT
import numpy as np


def compress_channel(
    values: np.ndarray,
    coords: np.ndarray,
    out: np.ndarray,
    *,
    k: int = 1500,
    thresh: float = 1e-4,
) -> bool:
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


def compress_highres(
    triplets: np.ndarray,
    out: np.ndarray,
    *,
    k: int = 1500,
    thresh: float = 1e-4,
    pad_val: float = -999.0,
) -> bool:
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

    # write and pad
    n = vals.size
    out[:n] = vals.astype(np.float32, copy=False)
    if n < k:
        out[n:] = -np.inf

    # sort descending for determinism
    out[:k] = np.sort(out[:k])[::-1]
    return overflow


# Main
def main(args):
    chunk_size = args.chunk_size

    # Geometry constants
    HCAL_eta = np.linspace(-3, 3, 56, dtype=np.float32).reshape(56, 1)
    HCAL_phi = np.linspace(-np.pi, np.pi, 72, dtype=np.float32).reshape(1, 72)
    HCAL_etaphi = np.stack((np.broadcast_to(HCAL_eta, (56, 72)),
                            np.broadcast_to(HCAL_phi, (56, 72))), axis=-1).reshape(-1, 2)

    ECAL_eta = np.linspace(-3, 3, 280, dtype=np.float32).reshape(280, 1)
    ECAL_phi = np.linspace(-np.pi, np.pi, 360, dtype=np.float32).reshape(1, 360)
    ECAL_etaphi = np.stack((np.broadcast_to(ECAL_eta, (280, 360)),
                            np.broadcast_to(ECAL_phi, (280, 360))), axis=-1).reshape(-1, 2)

    # Set up ROOT TChain and pre-selection
    rhTree = ROOT.TChain("fevt/RHTree")
    if rhTree.Add(args.infile) == 0:
        sys.exit(f"[error] Could not open {args.infile}")

    nEvts = rhTree.GetEntries()
    print(f" >> Input file:  {args.infile}\n >> nEvts:       {nEvts}")

    def get_valid_event_indices(chain: ROOT.TChain, key: str = "A_diphoton_gen_m0") -> list[int]:
        good: list[int] = []
        n = chain.GetEntries()
        print(f" >> Pre-scan: looping over {n} events to check '{key}'...")
        for i in range(n):
            chain.GetEntry(i)
            if len(getattr(chain, key)):
                good.append(i)
            if i and i % max(1, n // 20) == 0:
                print(f"    ... {100 * i / n:4.0f}%")
        return good

    valid_indices = get_valid_event_indices(rhTree)
    nGood = len(valid_indices)
    if nGood == 0:
        sys.exit("[error] No events survived pre-selection.")

    print(f" >> {nGood}/{nEvts} events ({100 * nGood / nEvts:.1f} %) after pre-selection")

    # Dataset definitions
    HIT_SHAPE = (1500, 32, 3)
    SCALAR_SHAPE = (1,)

    dataset_names = [
        # diphoton - gen
        "A_diphoton_gen_m0",  "A_diphoton_gen_dR",  "A_diphoton_gen_E",
        "A_diphoton_gen_pT",  "A_diphoton_gen_eta", "A_diphoton_gen_phi",
        # diphoton - reco
        "A_diphoton_reco_M",  "A_diphoton_reco_dR", "A_diphoton_reco_E",
        "A_diphoton_reco_pT", "A_diphoton_reco_eta","A_diphoton_reco_phi",
        # ditau - gen
        "A_ditau_gen_m0",     "A_ditau_gen_dR",     "A_ditau_gen_E",
        "A_ditau_gen_pT",     "A_ditau_gen_eta",    "A_ditau_gen_phi",
        # hits
        "zero_suppressed_hit_collection",
    ]

    scalar_branches = [n for n in dataset_names if n != "zero_suppressed_hit_collection"]

    out_path = f"{args.outdir}/{args.decay}_{args.idx}.h5"
    print(f" >> Output file: {out_path}")

    # overflow bookkeeping
    overflow_counter = np.zeros(32, dtype=np.int64)

    # High-res tracker branch list (order defines channel index 0-29)
    HIGHRES_BRANCHES = [
        "ECAL_tracksPt_triplet_atECALfixIP",
        "BPIX_layer1_triplets_atPV", "BPIX_layer2_triplets_atPV", "BPIX_layer3_triplets_atPV", "BPIX_layer4_triplets_atPV",
        "FPIX_layer1_triplets_atPV", "FPIX_layer2_triplets_atPV", "FPIX_layer3_triplets_atPV",
        "TIB_layer1_triplets_atPV", "TIB_layer2_triplets_atPV", "TIB_layer3_triplets_atPV", "TIB_layer4_triplets_atPV",
        "TID_layer1_triplets_atPV", "TID_layer2_triplets_atPV", "TID_layer3_triplets_atPV",
        "TOB_layer1_triplets_atPV", "TOB_layer2_triplets_atPV", "TOB_layer3_triplets_atPV", "TOB_layer4_triplets_atPV",
        "TOB_layer5_triplets_atPV", "TOB_layer6_triplets_atPV",
        "TEC_layer1_triplets_atPV", "TEC_layer2_triplets_atPV", "TEC_layer3_triplets_atPV", "TEC_layer4_triplets_atPV",
        "TEC_layer5_triplets_atPV", "TEC_layer6_triplets_atPV", "TEC_layer7_triplets_atPV", "TEC_layer8_triplets_atPV",
        "TEC_layer9_triplets_atPV",
    ]

    # Pre-allocate working buffers (reused every event)
    hit_matrix = np.empty(HIT_SHAPE, dtype=np.float32)        # event-local, reused
    triplet_buffer = np.empty((5000, 3), dtype=np.float32)    # view for high-res branch

    # HDF5 in-RAM staging buffers - sized to chunk boundaries
    BUFFER_ROWS = chunk_size * 8
    hit_buffer = np.empty((BUFFER_ROWS, *HIT_SHAPE), dtype=np.float32)
    scalar_buffers = {n: np.empty((BUFFER_ROWS, 1), dtype=np.float32) for n in scalar_branches}

    # Start conversion
    sw = ROOT.TStopwatch()
    sw.Start()

    with h5py.File(out_path, "w") as proper_data:
        dsets = {}
        for name in dataset_names:
            shape = (nGood, *HIT_SHAPE) if name == "zero_suppressed_hit_collection" else (nGood, *SCALAR_SHAPE)
            chunks = (chunk_size, *HIT_SHAPE) if name == "zero_suppressed_hit_collection" else (chunk_size, *SCALAR_SHAPE)
            dsets[name] = proper_data.create_dataset(
                name,
                shape=shape,
                dtype="float32",
                compression="lzf",
                chunks=chunks,
            )

        write_head = 0  # next row to write to HDF5
        buf_fill = 0    # how many rows are currently in in-RAM buffer

        for sel_idx, tree_idx in enumerate(valid_indices):
            rhTree.GetEntry(tree_idx)
            if sel_idx % 100 == 0:
                print(f" .. Processing selected entry {sel_idx}/{nGood} (tree index {tree_idx})")

            # reset event buffer
            hit_matrix.fill(-np.inf)

            # ECAL energy
            ECAL_energy = np.asarray(rhTree.ECAL_energy, dtype=np.float32)
            compress_channel(
                ECAL_energy.reshape(-1),
                ECAL_etaphi,
                hit_matrix[:, 30, :],
                k=1500,
                thresh=1e-4,
            ) and overflow_counter.__setitem__(30, overflow_counter[30] + 1)

            # HBHE energy
            HBHE_energy = np.asarray(rhTree.HBHE_energy, dtype=np.float32)
            if compress_channel(HBHE_energy.reshape(-1), HCAL_etaphi, hit_matrix[:, 31, :]):
                overflow_counter[31] += 1

            # High-resolution tracker hits (30 branches)
            for ch, branch in enumerate(HIGHRES_BRANCHES):
                triplet_buffer[:] = np.asarray(getattr(rhTree, branch), dtype=np.float32).reshape(5000, 3)
                if compress_highres(triplet_buffer, hit_matrix[:, ch, 0]):
                    overflow_counter[ch] += 1

            # Scalars
            for name in scalar_branches:
                scalar_buffers[name][buf_fill, 0] = getattr(rhTree, name)[0]

            # stage into RAM buffer
            hit_buffer[buf_fill, :, :, :] = hit_matrix
            buf_fill += 1

            # flush if full
            if buf_fill == BUFFER_ROWS:
                slice_ = slice(write_head, write_head + buf_fill)
                dsets["zero_suppressed_hit_collection"][slice_, :, :, :] = hit_buffer
                for n in scalar_branches:
                    dsets[n][slice_, :] = scalar_buffers[n]
                write_head += buf_fill
                buf_fill = 0

        # flush trailing rows
        if buf_fill:
            slice_ = slice(write_head, write_head + buf_fill)
            dsets["zero_suppressed_hit_collection"][slice_, :, :, :] = hit_buffer[:buf_fill]
            for n in scalar_branches:
                dsets[n][slice_, :] = scalar_buffers[n][:buf_fill]
            write_head += buf_fill

        assert write_head == nGood, f"Logic error - expected {nGood} rows, wrote {write_head}"

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    print(" >> Real time:", sw.RealTime() / 60.0, "minutes")
    print(" >> CPU time: ", sw.CpuTime()  / 60.0, "minutes")

    print("========================================================")
    print("\n=== Overflow summary (events with >1500 non-zero hits) ===")
    channel_names = [
        "TrkPt", "ECAL_E",
        "BPIX_L1", "BPIX_L2", "BPIX_L3", "BPIX_L4",
        "FPIX_L1", "FPIX_L2", "FPIX_L3",
        "TIB_L1", "TIB_L2", "TIB_L3", "TIB_L4",
        "TID_L1", "TID_L2", "TID_L3",
        "TOB_L1", "TOB_L2", "TOB_L3", "TOB_L4", "TOB_L5", "TOB_L6",
        "TEC_L1", "TEC_L2", "TEC_L3", "TEC_L4", "TEC_L5", "TEC_L6", "TEC_L7", "TEC_L8", "TEC_L9",
        "HBHE_E",
    ]
    for ch, cnt in enumerate(overflow_counter):
        print(f"{channel_names[ch]:>8s}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROOT -> HDF5 converter (optimized)")
    parser.add_argument("-i", "--infile", default="output_qqgg.root", help="input ROOT file")
    parser.add_argument("-o", "--outdir", default=".", help="output directory")
    parser.add_argument("-d", "--decay", default="test", help="decay name")
    parser.add_argument("-n", "--idx",   type=int, default=0, help="index tag in output filename")
    parser.add_argument("-c", "--chunk_size", type=int, default=32, help="HDF5 chunk size")
    args = parser.parse_args()
    main(args)
