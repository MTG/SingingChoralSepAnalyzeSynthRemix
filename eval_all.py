import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint


from mir_eval import separation

import importlib.util
import sys


convtasnet_dict = {'use_gpu': 0, 'exp_dir': './SATB/ALL_uc2/', 'n_save_ex': -1, 'sample_rate': 22020, 'train_conf': {'data': {'mode': 'min', 'nondefault_nsrc': None, 'sample_rate': 22020, 'task': 'sep_clean', 'train_dir': '/home/pc2752/share/Darius/Wave-U-Net/satb_dataset_all.hdf5', 'valid_dir': '/home/pc2752/share/Darius/Wave-U-Net/satb_dataset_all.hdf5'}, 'filterbank': {'kernel_size': 16, 'n_filters': 512, 'stride': 8}, 'main_args': {'exp_dir': './SATB/ALL_uc2/', 'help': None}, 'masknet': {'bn_chan': 128, 'hid_chan': 512, 'mask_act': 'relu', 'n_blocks': 8, 'n_repeats': 3, 'n_src': 4, 'skip_chan': 128}, 'optim': {'lr': 0.001, 'optimizer': 'adam', 'weight_decay': 0.0}, 'positional arguments': {}, 'training': {'batch_size': 2, 'early_stop': False, 'epochs': 2000, 'half_lr': True, 'num_workers': 4}}}

cfg = {'model_config':\
 {'estimates_path': './Source_Estimates', \
 'data_path': 'data', \
 'satb_path_train': '../data/satb_dst/train/raw_audio', \
 'satb_path_valid': '../data/satb_dst/valid/raw_audio', \
 'satb_path_test': '../data/satb_dst/test/raw_audio', \
 'satb_hdf5_filepath': './satb_dataset_only_csd.hdf5', \
 'satb_debug': False, \
 'satb_use_case': 2, \
 'model_base_dir': 'checkpoints', \
 'log_dir': 'logs', \
 'batch_size': 4, \
 'init_sup_sep_lr': 0.0001, \
 'epoch_it': 2000, \
 'cache_size': 4000, \
 'num_workers': 4, \
 'num_snippets_per_track': 100, \
 'num_layers': 6, \
 'filter_size': 15, \
 'merge_filter_size': 5, \
 'input_filter_size': 15, \
 'output_filter_size': 1, \
 'num_initial_filters': 16, \
 'num_frames': 33536, \
 'expected_sr': 22050, \
 'mono_downmix': True, \
 'output_type': 'direct', \
 'output_activation': 'tanh', \
 'context': False, \
 'network': 'unet_spectrogram', \
 'upsampling': 'linear', \
 'task': 'satb', \
 'augmentation': True, \
 'raw_audio_loss': True, \
 'worse_epochs': 50, \
 'source_names': ['soprano', 'alto', 'tenor', 'bass'], \
 'num_sources': 4, \
 'num_channels': 1}, \
 'experiment_id': 'unet_baseline_csd_usecase2'}

def convtasnet(conf):

    sys.path.append('./asteroid')
    from asteroid.models import ConvTasNet
    from asteroid.utils import tensors_to_device
    from asteroid.models import save_publishable

    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    model = ConvTasNet.from_pretrained(model_path)
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device

    torch.no_grad().__enter__()

    mix, fs = sf.read('/home/pc2752/share//Darius/Wave-U-Net/test_set_mixes/dcs/DCS_TPQuartetA_mix.wav')

    mix = torch.from_numpy(mix).type(torch.FloatTensor)

    outputs = model.float()(mix)

    return outputs

def unet(cfg):
    sys.path.append('./Wave-U-Net')
    import Evaluate

    model_config = cfg["model_config"]
    model_path = os.path.join('/home/pc2752/share/Darius/Wave-U-Net/Source_Estimates/unet_968405-112000_csd',"968405-112000")
    input_path = os.path.join('/home/pc2752/share//Darius/Wave-U-Net/test_set_mixes/dcs/DCS_TPQuartetA_mix.wav')
    output_path = './'
    outputs = Evaluate.produce_source_estimates(model_config, model_path, input_path, output_path)

    return outputs

def main():

    alto, fs = sf.read('/home/pc2752/share//Darius/data/satb_dst/test_dcs/raw_audio/DCS_TPQuartetA/DCS_TPQuartetA_alto_1.wav')

    bass, fs = sf.read('/home/pc2752/share//Darius/data/satb_dst/test_dcs/raw_audio/DCS_TPQuartetA/DCS_TPQuartetA_bass_1.wav')

    soprano, fs = sf.read('/home/pc2752/share//Darius/data/satb_dst/test_dcs/raw_audio/DCS_TPQuartetA/DCS_TPQuartetA_soprano_1.wav')

    tenor, fs = sf.read('/home/pc2752/share//Darius/data/satb_dst/test_dcs/raw_audio/DCS_TPQuartetA/DCS_TPQuartetA_tenor_1.wav')


    outputs_conv = convtasnet(convtasnet_dict)
    outputs_unet = unet(cfg)

    sdr_conv, sir_conv, sar_conv, _ = separation.bss_eval_sources(np.array([soprano, alto, tenor, bass]), outputs_conv[:, :soprano.shape[0]].detach().numpy())


    sdr_unet, sir_unet, sar_unet, _ = separation.bss_eval_sources(np.array([soprano[:outputs_unet['soprano'][:soprano.shape[0]].shape[0]], alto[:outputs_unet['soprano'][:soprano.shape[0]].shape[0]], tenor[:outputs_unet['soprano'][:soprano.shape[0]].shape[0]], bass[:outputs_unet['soprano'][:soprano.shape[0]].shape[0]]]), np.array([outputs_unet['soprano'][:soprano.shape[0]], outputs_unet['alto'][:soprano.shape[0]], outputs_unet['tenor'][:soprano.shape[0]], outputs_unet['bass'][:soprano.shape[0]]]   ))
    
    print_data = pd.DataFrame({'model':['unet','convtasnet'], 'usecase': ['1','1'], 'sdr_soprano':[sdr_unet[0], sdr_conv[0]], 'sdr_alto':[sdr_unet[1], sdr_conv[1]], 'sdr_tenor':[sdr_unet[2], sdr_conv[2]], 'sdr_bass':[sdr_unet[3], sdr_conv[3]] })

    print(print_data)

main()

