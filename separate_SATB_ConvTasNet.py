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
sys.path.append('./asteroid')
from asteroid.models import ConvTasNet
from asteroid.utils import tensors_to_device
from asteroid.models import save_publishable
# from asteroid.metrics import get_metrics
# from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
# from asteroid.data.satb_dataset import SATBDataset

# ConvTasNet = importlib.util.spec_from_file_location("ConvTasNet", "./asteroid/asteroid/models/__init__.py")



from contextlib import contextmanager
import os

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


compute_metrics = ["si_sdr", "sdr", "sir", "sar", "stoi"]

def main(conf):




    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    model = ConvTasNet.from_pretrained(model_path)
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device

    torch.no_grad().__enter__()

    alto, fs = sf.read('/home/pc2752/share//Darius/data/satb_dst/test_dcs/raw_audio/DCS_TPQuartetA/DCS_TPQuartetA_alto_1.wav')

    bass, fs = sf.read('/home/pc2752/share//Darius/data/satb_dst/test_dcs/raw_audio/DCS_TPQuartetA/DCS_TPQuartetA_bass_1.wav')

    soprano, fs = sf.read('/home/pc2752/share//Darius/data/satb_dst/test_dcs/raw_audio/DCS_TPQuartetA/DCS_TPQuartetA_soprano_1.wav')

    tenor, fs = sf.read('/home/pc2752/share//Darius/data/satb_dst/test_dcs/raw_audio/DCS_TPQuartetA/DCS_TPQuartetA_tenor_1.wav')

    mix, fs = sf.read('/home/pc2752/share//Darius/Wave-U-Net/test_set_mixes/dcs/DCS_TPQuartetA_mix.wav')

    mix = torch.from_numpy(mix).type(torch.FloatTensor)

    outputs = model.float()(mix)

    sdr, sir, sar, perm = separation.bss_eval_sources(np.array([soprano, alto, tenor, bass]), outputs[:, :soprano.shape[0]].detach().numpy())

    print("SDR: {}\nSIR: {}\nSAR: {}".format(sdr, sir, sar))

    sf.write('./DCS_TPQuartetA_Soprano.wav', outputs[0], fs)

    sf.write('./DCS_TPQuartetA_Alto.wav', outputs[1], fs)

    sf.write('./DCS_TPQuartetA_Tenor.wav', outputs[2], fs)

    sf.write('./DCS_TPQuartetA_Bass.wav', outputs[3], fs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
    )
    parser.add_argument("--exp_dir", default="./SATB/ALL_uc2/", help="Experiment root")
    parser.add_argument(
        "--n_save_ex", type=int, default=-1, help="Number of audio examples to save, -1 means all"
    )

    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    # if args.task != arg_dic["train_conf"]["data"]["task"]:
    #     print(
    #         "Warning : the task used to test is different than "
    #         "the one from training, be sure this is what you want."
    #     )

    main(arg_dic)