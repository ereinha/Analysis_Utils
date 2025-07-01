import h5py
import ROOT
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--infile', default='output_qqgg.root', type=str, help='Input root file.')
parser.add_argument('-o', '--outdir', default='.', type=str, help='Output pq file dir.')
parser.add_argument('-d', '--decay', default='test', type=str, help='Decay name.')
parser.add_argument('-n', '--idx', default=0, type=int, help='Input root file index.')
parser.add_argument('-c', '--chunk_size', default=32, type=int, help='chunk size for h5 file')
args = parser.parse_args()
chunk_size = args.chunk_size

def compress_channel(values: np.ndarray,
                     coords:  np.ndarray,
                     k: int = 1500,
                     thresh: float = 1e-4):
    mask         = values > thresh
    if not mask.any():
        # no hits above threshold
        return np.full((k, 3), -np.inf, dtype=np.float32), False

    vals   = values[mask]
    coord  = coords[mask]

    # if too many hits, pick top k hits and remember the overflow
    if vals.shape[0] > k:
        top_idx = np.argpartition(-vals, k - 1)[:k]   # O(N)
        vals, coord = vals[top_idx], coord[top_idx]
        overflow = True
    else:
        overflow = False
        pad = k - vals.shape[0]
        # pad with -inf
        vals   = np.concatenate([vals,   np.full(pad, -np.inf, dtype=np.float32)])
        coord  = np.concatenate([coord,  np.full((pad, 2), -np.inf, dtype=np.float32)])

    hits = np.column_stack([vals, coord])   # (k, 3)
    return hits.astype(np.float32), overflow

overflow_counter = np.zeros(11, dtype=np.int64)

ECAL_eta = np.linspace(-3, 3, 56).reshape(56, 1)
ECAL_phi = np.linspace(-np.pi, np.pi, 72).reshape(1, 72)
ECAL_etaphi = np.ones((56, 72, 2))
ECAL_etaphi[:, :, 0] *= ECAL_eta
ECAL_etaphi[:, :, 1] *= ECAL_phi
ECAL_etaphi = ECAL_etaphi.reshape(56*72,2)

HCAL_eta = np.linspace(-3, 3, 280).reshape(280,1)
HCAL_phi = np.linspace(-np.pi, np.pi, 360).reshape(1, 360)
HCAL_etaphi = np.ones((280, 360, 2))
HCAL_etaphi[:, :, 0] *= HCAL_eta
HCAL_etaphi[:, :, 1] *= HCAL_phi
HCAL_etaphi = HCAL_etaphi.reshape(280*360,2)

rhTreeStr = args.infile
rhTree = ROOT.TChain("fevt/RHTree")
rhTree.Add(rhTreeStr)
nEvts = rhTree.GetEntries()
assert nEvts > 0
print (" >> Input file:",rhTreeStr)
print (" >> nEvts:",nEvts)
outStr = '%s/%s_%s.h5'%(args.outdir, args.decay, args.idx)
print (" >> Output file:",outStr)


iEvtStart = 0
iEvtEnd   = nEvts
assert iEvtEnd <= nEvts
print(" >> Processing entries: [",iEvtStart,"->",iEvtEnd,")")

sw = ROOT.TStopwatch()
sw.Start()
with h5py.File(outStr, "w") as proper_data:

    # geometry-dependent constants
    HIT_SHAPE   = (1500, 11, 3)           # one row ->  zero-suppressed hit matrix
    SCALAR_SHAPE = (1,)                   # one row ->  any single scalar value

    dataset_names = [
        'A_diphoton_gen_m0',  'A_diphoton_gen_dR',  'A_diphoton_gen_E',
        'A_diphoton_gen_pT',  'A_diphoton_gen_eta', 'A_diphoton_gen_phi',
        'A_diphoton_reco_M',  'A_diphoton_reco_dR', 'A_diphoton_reco_E',
        'A_diphoton_reco_pT', 'A_diphoton_reco_eta','A_diphoton_reco_phi',
        'A_ditau_gen_m0',     'A_ditau_gen_dR',     'A_ditau_gen_E',
        'A_ditau_gen_pT',     'A_ditau_gen_eta',    'A_ditau_gen_phi',
        'zero_suppressed_hit_collection'
    ]

    dsets = {}
    for name in dataset_names:
        if name == 'zero_suppressed_hit_collection':
            dsets[name] = proper_data.create_dataset(
                name,
                shape=(nEvts, *HIT_SHAPE),
                dtype='float32',
                compression='lzf',
                chunks=(args.chunk_size, *HIT_SHAPE)   # keep fast chunking on disk
            )
        else:
            dsets[name] = proper_data.create_dataset(
                name,
                shape=(nEvts, *SCALAR_SHAPE),
                dtype='float32',
                compression='lzf',
                chunks=(args.chunk_size, *SCALAR_SHAPE)
            )


    BUFFER_ROWS      = args.chunk_size * 32     # any multiple of chunk size works
    hit_buffer       = np.empty((BUFFER_ROWS, *HIT_SHAPE),   dtype=np.float32)
    scalar_buffers   = {n: np.empty((BUFFER_ROWS, 1), dtype=np.float32)
                        for n in dataset_names if n != 'zero_suppressed_hit_collection'}

    write_head = 0 # next row in HDF5 file to be filled
    buf_fill   = 0 # rows currently stored in the in-memory buffer

    for iEvt in range(iEvtStart, iEvtEnd):

        rhTree.GetEntry(iEvt)
        if iEvt % 100 == 0:
            print(" .. Processing entry", iEvt)

        HighResCollection = np.stack([
            np.array(rhTree.ECAL_tracksPt_atECALfixIP).reshape(280*360),
            np.array(rhTree.ECAL_energy).reshape(280*360),
            np.array(rhTree.BPIX_layer1_ECAL_atPV).reshape(280*360),
            np.array(rhTree.BPIX_layer2_ECAL_atPV).reshape(280*360),
            np.array(rhTree.BPIX_layer3_ECAL_atPV).reshape(280*360),
            np.array(rhTree.BPIX_layer4_ECAL_atPV).reshape(280*360),
            np.array(rhTree.TIB_layer1_ECAL_atPV).reshape(280*360),
            np.array(rhTree.TIB_layer2_ECAL_atPV).reshape(280*360),
            np.array(rhTree.TOB_layer1_ECAL_atPV).reshape(280*360),
            np.array(rhTree.TOB_layer2_ECAL_atPV).reshape(280*360)
        ], axis=1)

        hit_matrix = np.full((1500, 11, 3), -np.inf, dtype=np.float32)

        # 10 high-resolution channels (HCAL grid for coordinates)
        for ch in range(10):
            hit_slice, over = compress_channel(HighResCollection[:, ch], HCAL_etaphi, 1500, 1e-4)
            hit_matrix[:, ch, :] = hit_slice
            if over:
                overflow_counter[ch] += 1

        # HBHE channel (ECAL grid for coordinates)
        HBHE_energy = np.array(rhTree.HBHE_energy).reshape(56*72)
        hit_slice, over = compress_channel(HBHE_energy, ECAL_etaphi, 1500, 1e-4)
        hit_matrix[:, 10, :] = hit_slice
        if over:
            overflow_counter[10] += 1


        # fill the in-RAM buffer
        hit_buffer[buf_fill, :, :, :] = hit_matrix
        scalar_buffers['A_diphoton_gen_m0' ][buf_fill, 0] = rhTree.A_diphoton_gen_m0[0]
        scalar_buffers['A_diphoton_gen_dR' ][buf_fill, 0] = rhTree.A_diphoton_gen_dR[0]
        scalar_buffers['A_diphoton_gen_E'  ][buf_fill, 0] = rhTree.A_diphoton_gen_E[0]
        scalar_buffers['A_diphoton_gen_pT' ][buf_fill, 0] = rhTree.A_diphoton_gen_pT[0]
        scalar_buffers['A_diphoton_gen_eta'][buf_fill, 0] = rhTree.A_diphoton_gen_eta[0]
        scalar_buffers['A_diphoton_gen_phi'][buf_fill, 0] = rhTree.A_diphoton_gen_phi[0]

        scalar_buffers['A_diphoton_reco_M' ][buf_fill, 0] = rhTree.A_diphoton_reco_M[0]
        scalar_buffers['A_diphoton_reco_dR'][buf_fill, 0] = rhTree.A_diphoton_reco_dR[0]
        scalar_buffers['A_diphoton_reco_E' ][buf_fill, 0] = rhTree.A_diphoton_reco_E[0]
        scalar_buffers['A_diphoton_reco_pT'][buf_fill, 0] = rhTree.A_diphoton_reco_pT[0]
        scalar_buffers['A_diphoton_reco_eta'][buf_fill, 0] = rhTree.A_diphoton_reco_eta[0]
        scalar_buffers['A_diphoton_reco_phi'][buf_fill, 0] = rhTree.A_diphoton_reco_phi[0]

        scalar_buffers['A_ditau_gen_m0'    ][buf_fill, 0] = rhTree.A_ditau_gen_m0[0]
        scalar_buffers['A_ditau_gen_dR'    ][buf_fill, 0] = rhTree.A_ditau_gen_dR[0]
        scalar_buffers['A_ditau_gen_E'     ][buf_fill, 0] = rhTree.A_ditau_gen_E[0]
        scalar_buffers['A_ditau_gen_pT'    ][buf_fill, 0] = rhTree.A_ditau_gen_pT[0]
        scalar_buffers['A_ditau_gen_eta'   ][buf_fill, 0] = rhTree.A_ditau_gen_eta[0]
        scalar_buffers['A_ditau_gen_phi'   ][buf_fill, 0] = rhTree.A_ditau_gen_phi[0]

        buf_fill += 1

        if buf_fill == BUFFER_ROWS:
            slice_ = slice(write_head, write_head + buf_fill)
            dsets['zero_suppressed_hit_collection'][slice_, :, :, :] = hit_buffer
            for n, arr in scalar_buffers.items():
                dsets[n][slice_, :] = arr

            write_head += buf_fill
            buf_fill = 0      # reset buffer pointer

    if buf_fill:
        slice_ = slice(write_head, write_head + buf_fill)
        dsets['zero_suppressed_hit_collection'][slice_, :, :, :] = hit_buffer[:buf_fill]
        for n, arr in scalar_buffers.items():
            dsets[n][slice_, :] = arr[:buf_fill]

        write_head += buf_fill   # (optional) sanity check

    assert write_head == nEvts, "Logic error - not all rows written!"

print(" >> Real time:", sw.RealTime() / 60., "minutes")
print(" >> CPU time: ", sw.CpuTime() / 60., "minutes")
print("========================================================")
print("\n=== Overflow summary (events that had >1500 non-zero hits) ===")
channel_names = [
    'TracksPt', 'ECAL_E',
    'BPIX_L1', 'BPIX_L2', 'BPIX_L3', 'BPIX_L4',
    'TIB_L1', 'TIB_L2', 'TOB_L1', 'TOB_L2',
    'HBHE_E'
]
for ch, cnt in enumerate(overflow_counter):
    print(f"{channel_names[ch]:>8s}: {cnt}")
