import sys
sys.path.append('./Wave-U-Net')
import Evaluate
import os


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

model_config = cfg["model_config"]

model_path = os.path.join('/home/pc2752/share/Darius/Wave-U-Net/Source_Estimates/unet_968405-112000_csd',"968405-112000")

input_path = os.path.join("/home/pc2752/share/Darius/Wave-U-Net/test_set_mixes","dcs","DCS_TPFullChoir_mix.wav")

output_path = './'

Evaluate.produce_source_estimates(model_config, model_path, input_path, output_path)

