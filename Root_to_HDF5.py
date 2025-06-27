import h5py
import ROOT
import numpy as np
import glob, os
from numpy.lib.stride_tricks import as_strided

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

def upsample_array(x, b0, b1):
    r, c = x.shape                                    # number of rows/columns
    rs, cs = x.strides                                # row/column strides
    x = as_strided(x, (r, b0, c, b1), (rs, 0, cs, 0)) # view as a larger 4D array

    return x.reshape(r*b0, c*b1)/(b0*b1)              # create new 2D array with same total occupancy

def crop_jet(imgECAL, iphi, ieta, jet_shape=125):
    # NOTE: jet_shape here should correspond to the one used in RHAnalyzer
    off = jet_shape//2
    iphi = int(iphi*5 + 2) # 5 EB xtals per HB tower
    ieta = int(ieta*5 + 2) # 5 EB xtals per HB tower

    # Wrap-around on left side
    if iphi < off:
        diff = off-iphi
        img_crop = np.concatenate((imgECAL[:,ieta-off:ieta+off+1,-diff:],
                                   imgECAL[:,ieta-off:ieta+off+1,:iphi+off+1]), axis=-1)
    # Wrap-around on right side
    elif 360-iphi < off:
        diff = off - (360-iphi)
        img_crop = np.concatenate((imgECAL[:,ieta-off:ieta+off+1,iphi-off:],
                                   imgECAL[:,ieta-off:ieta+off+1,:diff+1]), axis=-1)
    # Nominal case
    else:
        img_crop = imgECAL[:,ieta-off:ieta+off+1,iphi-off:iphi+off+1]

    return img_crop

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
        dataset_names = ['all_jet', 'am', 'ieta', 'iphi', 'apt', 'a_eta', 'a_phi','jet_mass', 'jet_pt', 'jet_e', 'TaudR', 'nVtx']
        datasets = {
            name: proper_data.create_dataset(
                name,
                shape= (0, 13, 125, 125) if 'all_jet' in name else (0,1),
                maxshape=(None, 13, 125, 125) if 'all_jet' in name else (None, 1),
                dtype='float32',  # Specify an appropriate data type
                compression='lzf',
                chunks=(chunk_size, 13, 125, 125) if 'all_jet' in name else (chunk_size, 1),
            ) for name in dataset_names
        }
        end_idx = 0
        for iEvt in range(iEvtStart,iEvtEnd):

            # Initialize event
            rhTree.GetEntry(iEvt)

            if iEvt % 100 == 0:
                print(" .. Processing entry",iEvt)

            # Jet attributes
            ams    = rhTree.a_m
            apts   = rhTree.a_pt
            aetas   = rhTree.a_eta
            aphis   = rhTree.a_phi
            jet_mass = rhTree.jetM
            jet_Pt = rhTree.jetPt
            jet_E = rhTree.jetE
            iphis  = rhTree.jetSeed_iphi
            ietas  = rhTree.jetSeed_ieta
            taudrs = rhTree.TaudR
            nVtx  = rhTree.nVtx
            ys  = min(len(ietas), len(iphis))
            if ys < 2: continue
            end_idx = end_idx + ys



            ECAL_energy = np.array(rhTree.ECAL_energy).reshape(280,360)
            HBHE_energy = np.array(rhTree.HBHE_energy).reshape(56,72)
            HBHE_energy = upsample_array(HBHE_energy, 5, 5) # (280, 360)
            TracksAtECAL_pt    = np.array(rhTree.ECAL_tracksPt_atECALfixIP).reshape(280,360)
            TracksAtECAL_dZSig = np.array(rhTree.ECAL_tracksDzSig_atECALfixIP).reshape(280,360)
            TracksAtECAL_d0Sig = np.array(rhTree.ECAL_tracksD0Sig_atECALfixIP).reshape(280,360)
            PixAtEcal_1        = np.array(rhTree.BPIX_layer1_ECAL_atPV).reshape(280,360)
            PixAtEcal_2        = np.array(rhTree.BPIX_layer2_ECAL_atPV).reshape(280,360)
            PixAtEcal_3        = np.array(rhTree.BPIX_layer3_ECAL_atPV).reshape(280,360)
            PixAtEcal_4        = np.array(rhTree.BPIX_layer4_ECAL_atPV).reshape(280,360)
            TibAtEcal_1        = np.array(rhTree.TIB_layer1_ECAL_atPV).reshape(280,360)
            TibAtEcal_2        = np.array(rhTree.TIB_layer2_ECAL_atPV).reshape(280,360)
            TobAtEcal_1        = np.array(rhTree.TOB_layer1_ECAL_atPV).reshape(280,360)
            TobAtEcal_2        = np.array(rhTree.TOB_layer2_ECAL_atPV).reshape(280,360)
            X_CMSII            = np.stack([TracksAtECAL_pt, TracksAtECAL_dZSig, TracksAtECAL_d0Sig, ECAL_energy, HBHE_energy, PixAtEcal_1, PixAtEcal_2, PixAtEcal_3, PixAtEcal_4, TibAtEcal_1, TibAtEcal_2, TobAtEcal_1, TobAtEcal_2], axis=0) # (13, 280, 360)




            for name, dataset in datasets.items():
                dataset.resize((end_idx,13, 125, 125) if 'all_jet' in name else (end_idx,1))

            for i in range(ys):
                proper_data['all_jet'][end_idx - ys + i, :, :, :] = crop_jet(X_CMSII, iphis[i], ietas[i], jet_shape=125)

                proper_data['am'][end_idx - ys + i, :] = ams[i]
                proper_data['ieta'][end_idx - ys + i, :] = ietas[i]
                proper_data['iphi'][end_idx - ys + i, :] = iphis[i]
                proper_data['apt'][end_idx - ys + i, :] = apts[i]
                proper_data['a_eta'][end_idx - ys + i, :] = aetas[i]
                proper_data['a_phi'][end_idx - ys + i, :] = aphis[i]
                proper_data['jet_mass'][end_idx - ys + i, :] = jet_mass[i]
                proper_data['jet_pt'][end_idx - ys + i, :] = jet_Pt[i]
                proper_data['jet_e'][end_idx - ys + i, :] = jet_E[i]
                proper_data['TaudR'][end_idx - ys + i, :] = taudrs[i]
                proper_data['nVtx'][end_idx - ys + i, :] = nVtx


print(" >> Real time:",sw.RealTime()/60.,"minutes")
print(" >> CPU time: ",sw.CpuTime() /60.,"minutes")
print("========================================================")
