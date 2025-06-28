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
parser.add_argument('-m', '--mass', default='', type=str, help='signal mass')
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
with h5py.File(f'{outStr}', 'w') as proper_data:
        dataset_names = ['A_diphoton_gen_m0', 'A_diphoton_gen_dR', 'A_diphoton_gen_E', 
                         'A_diphoton_gen_pT', 'A_diphoton_gen_eta', 'A_diphoton_gen_phi',
                         'A_diphoton_reco_M', 'A_diphoton_reco_dR', 'A_diphoton_reco_E',
                         'A_diphoton_reco_pT', 'A_diphoton_reco_eta', 'A_diphoton_reco_phi',
                         'A_ditau_gen_m0', 'A_ditau_gen_dR', 'A_ditau_gen_E',
                         'A_ditau_gen_pT', 'A_ditau_gen_eta', 'A_ditau_gen_phi']
        datasets = {
            name: proper_data.create_dataset(
                name,
                shape= (0, 1500, 11, 3) if 'zero_suppressed_hit_collection' in name else (0,1),
                maxshape=(None, 1500, 11, 3) if 'zero_suppressed_hit_collection' in name else (None, 1),
                dtype='float32',  # Specify an appropriate data type
                compression='lzf',
                chunks=(chunk_size, 1500, 11, 3) if 'zero_suppressed_hit_collection' in name else (chunk_size, 1),
            ) for name in dataset_names
        }
        end_idx = 0
        for iEvt in range(iEvtStart,iEvtEnd):

            # Initialize event
            rhTree.GetEntry(iEvt)

            if iEvt % 100 == 0:
                print(" .. Processing entry",iEvt)

            # Jet attributes
            A_diphoton_gen_m0       = rhTree.A_diphoton_gen_m0
            A_diphoton_gen_dR       = rhTree.A_diphoton_gen_dR
            A_diphoton_gen_E        = rhTree.A_diphoton_gen_E
            A_diphoton_gen_pT       = rhTree.A_diphoton_gen_pT
            A_diphoton_gen_eta      = rhTree.A_diphoton_gen_eta
            A_diphoton_gen_phi      = rhTree.A_diphoton_gen_phi
            A_diphoton_reco_M       = rhTree.A_diphoton_reco_M
            A_diphoton_reco_dR      = rhTree.A_diphoton_reco_dR
            A_diphoton_reco_E       = rhTree.A_diphoton_reco_E
            A_diphoton_reco_pT      = rhTree.A_diphoton_reco_pT
            A_diphoton_reco_eta     = rhTree.A_diphoton_reco_eta
            A_diphoton_reco_phi     = rhTree.A_diphoton_reco_phi

            A_ditau_gen_m0          = rhTree.A_ditau_gen_m0
            A_ditau_gen_dR          = rhTree.A_ditau_gen_dR
            A_ditau_gen_E           = rhTree.A_ditau_gen_E
            A_ditau_gen_pT          = rhTree.A_ditau_gen_pT
            A_ditau_gen_eta         = rhTree.A_ditau_gen_eta
            A_ditau_gen_phi         = rhTree.A_ditau_gen_phi

            ys  = len(A_diphoton_gen_m0)
            if ys < 2: continue
            end_idx = end_idx + ys



            HBHE_energy        = np.array(rhTree.HBHE_energy).reshape(56*72)
            ECAL_energy        = np.array(rhTree.ECAL_energy).reshape(280*360)
            TracksAtECAL_pt    = np.array(rhTree.ECAL_tracksPt_atECALfixIP).reshape(280*360)
            PixAtEcal_1        = np.array(rhTree.BPIX_layer1_ECAL_atPV).reshape(280*360)
            PixAtEcal_2        = np.array(rhTree.BPIX_layer2_ECAL_atPV).reshape(280*360)
            PixAtEcal_3        = np.array(rhTree.BPIX_layer3_ECAL_atPV).reshape(280*360)
            PixAtEcal_4        = np.array(rhTree.BPIX_layer4_ECAL_atPV).reshape(280*360)
            TibAtEcal_1        = np.array(rhTree.TIB_layer1_ECAL_atPV).reshape(280*360)
            TibAtEcal_2        = np.array(rhTree.TIB_layer2_ECAL_atPV).reshape(280*360)
            TobAtEcal_1        = np.array(rhTree.TOB_layer1_ECAL_atPV).reshape(280*360)
            TobAtEcal_2        = np.array(rhTree.TOB_layer2_ECAL_atPV).reshape(280*360)
            HighResCollection            = np.stack([TracksAtECAL_pt, ECAL_energy, PixAtEcal_1, 
                                                     PixAtEcal_2, PixAtEcal_3, PixAtEcal_4, 
                                                     TibAtEcal_1, TibAtEcal_2, TobAtEcal_1, 
                                                     TobAtEcal_2], axis=1)

            for name, dataset in datasets.items():
                dataset.resize((end_idx, 1500, 11, 3) if 'zero_suppressed_hit_collection' in name else (end_idx,1))

            # build hit matrix once per event (will be reused for each jet i)
            hit_matrix = np.full((1500, 11, 3), -np.inf, dtype=np.float32)

            # 10 high-res channels (use HCAL grid for coords)
            for ch in range(10):
                hit_slice, over = compress_channel(
                    HighResCollection[:, ch],
                    HCAL_etaphi, 1500, 1e-4)
                hit_matrix[:, ch, :] = hit_slice
                if over:
                    overflow_counter[ch] += 1

            # HBHE channel (use ECAL grid for coords)
            hit_slice, over = compress_channel(
                HBHE_energy,
                ECAL_etaphi, 1500, 1e-4)
            hit_matrix[:, 10, :] = hit_slice
            if over:
                overflow_counter[10] += 1

            for i in range(ys):
                proper_data['zero_suppressed_hit_collection'][end_idx - ys + i, :, :, :] = hit_matrix

                proper_data['A_diphoton_gen_m0'][end_idx - ys + i, :]   = A_diphoton_gen_m0[i]
                proper_data['A_diphoton_gen_dR'][end_idx - ys + i, :]   = A_diphoton_gen_dR[i]
                proper_data['A_diphoton_gen_E'][end_idx - ys + i, :]    = A_diphoton_gen_E[i]
                proper_data['A_diphoton_gen_pT'][end_idx - ys + i, :]   = A_diphoton_gen_pT[i]
                proper_data['A_diphoton_gen_eta'][end_idx - ys + i, :]  = A_diphoton_gen_eta[i]
                proper_data['A_diphoton_gen_phi'][end_idx - ys + i, :]  = A_diphoton_gen_phi[i]
                proper_data['A_diphoton_reco_M'][end_idx - ys + i, :]   = A_diphoton_reco_M[i]
                proper_data['A_diphoton_reco_dR'][end_idx - ys + i, :]  = A_diphoton_reco_dR[i]
                proper_data['A_diphoton_reco_E'][end_idx - ys + i, :]   = A_diphoton_reco_E[i]
                proper_data['A_diphoton_reco_pT'][end_idx - ys + i, :]  = A_diphoton_reco_pT[i]
                proper_data['A_diphoton_reco_eta'][end_idx - ys + i, :] = A_diphoton_reco_eta[i]
                proper_data['A_diphoton_reco_phi'][end_idx - ys + i, :] = A_diphoton_reco_phi[i]

                proper_data['A_ditau_gen_m0'][end_idx - ys + i, :]      = A_ditau_gen_m0[i]
                proper_data['A_ditau_gen_dR'][end_idx - ys + i, :]      = A_ditau_gen_dR[i]
                proper_data['A_ditau_gen_E'][end_idx - ys + i, :]       = A_ditau_gen_E[i]
                proper_data['A_ditau_gen_pT'][end_idx - ys + i, :]      = A_ditau_gen_pT[i]
                proper_data['A_ditau_gen_eta'][end_idx - ys + i, :]     = A_ditau_gen_eta[i]
                proper_data['A_ditau_gen_phi'][end_idx - ys + i, :]     = A_ditau_gen_phi[i]

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
