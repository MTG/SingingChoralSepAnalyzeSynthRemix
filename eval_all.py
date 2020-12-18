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

import time

from mir_eval import separation

import importlib.util
import sys


import config


def convtasnet(conf):

    sys.path.append('./asteroid')
    from asteroid.models import ConvTasNet
    from asteroid.utils import tensors_to_device
    from asteroid.models import save_publishable

    model_path = './models/convtasnet_usecase2.pth'
    model = ConvTasNet.from_pretrained(model_path)
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device

    torch.no_grad().__enter__()

    mix, fs = sf.read(conf["input_path"])

    mix = torch.from_numpy(mix).type(torch.FloatTensor)

    outputs = model.float()(mix)

    return outputs

def unet(cfg):
    sys.path.append('./Wave-U-Net')
    import Evaluate

    model_config = cfg["model_config"]
    model_path = cfg["model_path"]
    input_path = cfg['input_path']
    output_path = './'
    outputs = Evaluate.produce_source_estimates(model_config, model_path, input_path, output_path)

    return outputs

def waveunet(cfg):
    sys.path.append('./Wave-U-Net')
    import Evaluate

    model_config = cfg["model_config"]
    model_path = cfg["model_path"]
    input_path = cfg['input_path']
    output_path = './'
    outputs = Evaluate.produce_source_estimates(model_config, model_path, input_path, output_path)

    return outputs


def DPRNN(conf):
    sys.path.append('./asteroid')
    from asteroid.models import DPRNNTasNet
    from asteroid.utils import tensors_to_device
    from asteroid.models import save_publishable

    model_path = "./models/dprnn_usecase1"
    model = DPRNNTasNet.from_pretrained(model_path)
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device

    torch.no_grad().__enter__()

    mix, fs = sf.read(conf["input_path"])

    mix = torch.from_numpy(mix).type(torch.FloatTensor)

    outputs = model.float()(mix)

    return outputs    

def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    #remove % and cast to float
    data = data.replace('%','', regex=True).astype(float)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)

def main():

    alto, fs = sf.read('/home/pc2752/share//Darius/data/satb_dst/test_dcs/raw_audio/DCS_TPQuartetA/DCS_TPQuartetA_alto_1.wav')

    bass, fs = sf.read('/home/pc2752/share//Darius/data/satb_dst/test_dcs/raw_audio/DCS_TPQuartetA/DCS_TPQuartetA_bass_1.wav')

    soprano, fs = sf.read('/home/pc2752/share//Darius/data/satb_dst/test_dcs/raw_audio/DCS_TPQuartetA/DCS_TPQuartetA_soprano_1.wav')

    tenor, fs = sf.read('/home/pc2752/share//Darius/data/satb_dst/test_dcs/raw_audio/DCS_TPQuartetA/DCS_TPQuartetA_tenor_1.wav')

    start_time = time.time()
    outputs_dprnn = DPRNN(config.config_dprnn)
    end_time_dprnn = time.time()- start_time

    start_time = time.time()
    outputs_conv = convtasnet(config.config_convtasnet)
    end_time_conv = time.time()- start_time

    start_time = time.time()
    outputs_unet = unet(config.cnfig_unet_baseline_csd_usecase2)
    end_time_unet = time.time()- start_time

    start_time = time.time()
    outputs_waveunet = waveunet(config.cnfig_waveunet_all_usecase2)
    end_time_waveunet = time.time()- start_time

    sdr_dprnn, sir_dprnn, sar_dprnn, _ = separation.bss_eval_sources(np.array([soprano, alto, tenor, bass]), outputs_dprnn[:, :soprano.shape[0]].detach().numpy())

    sdr_conv, sir_conv, sar_conv, _ = separation.bss_eval_sources(np.array([soprano, alto, tenor, bass]), outputs_conv[:, :soprano.shape[0]].detach().numpy())

    sdr_unet, sir_unet, sar_unet, _ = separation.bss_eval_sources(np.array([soprano[:outputs_unet['soprano'][:soprano.shape[0]].shape[0]], alto[:outputs_unet['soprano'][:soprano.shape[0]].shape[0]], tenor[:outputs_unet['soprano'][:soprano.shape[0]].shape[0]], bass[:outputs_unet['soprano'][:soprano.shape[0]].shape[0]]]), np.array([outputs_unet['soprano'][:soprano.shape[0]], outputs_unet['alto'][:soprano.shape[0]], outputs_unet['tenor'][:soprano.shape[0]], outputs_unet['bass'][:soprano.shape[0]]]   ))
    
    sdr_waveunet, sir_waveunet, sar_waveunet, _ = separation.bss_eval_sources(np.array([soprano[:outputs_waveunet['soprano'][:soprano.shape[0]].shape[0]], alto[:outputs_waveunet['soprano'][:soprano.shape[0]].shape[0]], tenor[:outputs_waveunet['soprano'][:soprano.shape[0]].shape[0]], bass[:outputs_waveunet['soprano'][:soprano.shape[0]].shape[0]]]), np.array([outputs_waveunet['soprano'][:soprano.shape[0]], outputs_waveunet['alto'][:soprano.shape[0]], outputs_waveunet['tenor'][:soprano.shape[0]], outputs_waveunet['bass'][:soprano.shape[0]]]   ))

    print_data = pd.DataFrame({'model':['unet','convtasnet', 'waveunet', 'drpnn'], 'trainusecase': ['2','1','2','1'],'evalusecase': ['1','1','1','1'],\
     'sdr_soprano':[sdr_unet[0], sdr_conv[0], sdr_waveunet[0], sdr_dprnn[0]], 'sdr_alto':[sdr_unet[1], sdr_conv[1], sdr_waveunet[1], sdr_dprnn[1] ],\
     'sdr_tenor':[sdr_unet[2], sdr_conv[2], sdr_waveunet[2], sdr_dprnn[2]], 'sdr_bass':[sdr_unet[3], sdr_conv[3], sdr_waveunet[3], sdr_dprnn[3]],\
      'time': [end_time_unet, end_time_conv, end_time_waveunet, end_time_dprnn] })

    pd.options.display.float_format = "{:,.2f}".format

    # print_data.style.apply(highlight_max)

    print(print_data)

    # from tabulate import tabulate
    # pdtabulate=lambda df:tabulate(df,headers='keys',tablefmt='html')

    # print(pdtabulate(print_data))

main()

