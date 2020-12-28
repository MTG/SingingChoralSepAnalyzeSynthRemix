from sacred import Ingredient

#------ U-Net / Wave-U-Net ------#

config_ingredient = Ingredient("cfg_unet")

@config_ingredient.config
# Base U-Net Config
def cfg_unet():
  unet_cfg = {'model_config':\
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
          'model_path': '', \
          'input_path': './chorus_inputs/DCS_TPQuartetA_mix.wav',\
          'experiment_id': 'base_cfg'}

@config_ingredient.named_config
def unet_baseline_csd_usecase1():
    unet_cfg['model_config'] = { \
          "batch_size": 4, \
          "network" : "unet_spectrogram", \
          "num_layers" : 6, \
          "expected_sr" : 22050, \
          "num_frames" : 256 * 127 + 1024, \
          "num_initial_filters" : 16, \
          'satb_hdf5_filepath': './datasets/satb_dataset_only_csd.hdf5'}, \
        'model_path': './Wave-U-Net/checkpoints/unet_baseline_csd_usecase1/968405-84000', \
        'experiment_id': 'unet_baseline_csd_usecase1'}
    }

@config_ingredient.named_config
def unet_baseline_csd_usecase2():
    unet_cfg['model_config'] = { \
          "batch_size": 4, \
          "network" : "unet_spectrogram", \
          "num_layers" : 6, \
          "expected_sr" : 22050, \
          "num_frames" : 256 * 127 + 1024, \
          "num_initial_filters" : 16, \
          'satb_hdf5_filepath': './datasets/satb_dataset_only_csd.hdf5'}, \
        'model_path': './Wave-U-Net/checkpoints/unet_baseline_csd_usecase2/unet_baseline_csd_usecase2-222000', \
        'experiment_id': 'unet_baseline_csd_usecase2'}
    }

@config_ingredient.named_config
def unet_all_usecase1():
    unet_cfg['model_config'] = { \
          "batch_size": 4, \
          "network" : "unet_spectrogram", \
          "num_layers" : 6, \
          "expected_sr" : 22050, \
          "num_frames" : 256 * 127 + 1024, \
          "num_initial_filters" : 16, \
          'satb_hdf5_filepath': './datasets/satb_dataset_all.hdf5'}, \
        'model_path': './Wave-U-Net/checkpoints/unet_all_usecase1/895799-372000', \
        'experiment_id': 'unet_all_usecase1'}
    }

@config_ingredient.named_config
def unet_all_usecase2():
    unet_cfg['model_config'] = { \
          "batch_size": 4, \
          "network" : "unet_spectrogram", \
          "num_layers" : 6, \
          "expected_sr" : 22050, \
          "num_frames" : 256 * 127 + 1024, \
          "num_initial_filters" : 16, \
          'satb_hdf5_filepath': './datasets/satb_dataset_all.hdf5'}, \
        'model_path': './Wave-U-Net/checkpoints/unet_all_usecase2/464336-506000', \
        'experiment_id': 'unet_all_usecase2'}
    }

@config_ingredient.named_config
def waveunet_baseline_csd_usecase1():
    unet_cfg['model_config'] = { \
          "batch_size": 16, \
          "network" : "unet", \
          "num_layers" : 12, \
          "expected_sr" : 22050, \
          "num_frames" : 16384, \
          "num_initial_filters" : 24, \
          'satb_hdf5_filepath': './datasets/satb_dataset_only_csd.hdf5'}, \
        'model_path': './Wave-U-Net/checkpoints/waveunet_baseline_csd_usecase1/833222-102000', \
        'experiment_id': 'waveunet_baseline_csd_usecase1'}
    }

@config_ingredient.named_config
def waveunet_baseline_csd_usecase2():
    unet_cfg['model_config'] = { \
          "batch_size": 16, \
          "network" : "unet", \
          "num_layers" : 12, \
          "expected_sr" : 22050, \
          "num_frames" : 16384, \
          "num_initial_filters" : 24, \
          'satb_hdf5_filepath': './datasets/satb_dataset_only_csd.hdf5'}, \
        'model_path': './Wave-U-Net/checkpoints/waveunet_baseline_csd_usecase2/waveunet_baseline_csd_usecase2-192000', \
        'experiment_id': 'waveunet_baseline_csd_usecase2'}
    }

@config_ingredient.named_config
def waveunet_all_usecase1():
    unet_cfg['model_config'] = { \
          "batch_size": 16, \
          "network" : "unet", \
          "num_layers" : 12, \
          "expected_sr" : 22050, \
          "num_frames" : 16384, \
          "num_initial_filters" : 24, \
          'satb_hdf5_filepath': './datasets/satb_dataset_all.hdf5'}, \
        'model_path': './Wave-U-Net/checkpoints/waveunet_all_usecase1/874390-384000', \
        'experiment_id': 'waveunet_all_usecase1'}
    }

@config_ingredient.named_config
def waveunet_all_usecase2():
    unet_cfg['model_config'] = { \
          "batch_size": 16, \
          "network" : "unet", \
          "num_layers" : 12, \
          "expected_sr" : 22050, \
          "num_frames" : 16384, \
          "num_initial_filters" : 24, \
          'satb_hdf5_filepath': './datasets/satb_dataset_all.hdf5'}, \
        'model_path': './Wave-U-Net/checkpoints/waveunet_all_usecase2/407295-672000', \
        'experiment_id': 'waveunet_all_usecase2'}
    }

#------ Open Unmix ------#

config_ingredient = Ingredient("cfg_umix")

@config_ingredient.config
# Base UMIX Config
def cfg_umix():
  umix_cfg = {
          '--input' : './chorus_inputs/DCS_TPQuartetA_mix.wav', \
          '--targets' : ['soprano','alto','tenor','bass'], \
          '--outdir' : './predictions/umix', \
          '--start' : 0.0, \
          '--duration' : -1.0, \
          '--model' : 'open-unmix/umix_baseline_csd_usecase1', \
          '--no-cuda' : False}

@config_ingredient.named_config
def umix_baseline_csd_usecase1():
  umix_cfg = {
          '--input' : './chorus_inputs/DCS_TPQuartetA_mix.wav', \
          '--targets' : ['soprano','alto','tenor','bass'], \
          '--outdir' : './predictions/umix/umix_baseline_csd_usecase1', \
          '--start' : 0.0, \
          '--duration' : -1.0, \
          '--model' : 'open-unmix/umix_baseline_csd_usecase1', \
          '--no-cuda' : False}
    }

@config_ingredient.named_config
def umix_baseline_csd_usecase2():
  umix_cfg = {
          '--input' : './chorus_inputs/DCS_TPQuartetA_mix.wav', \
          '--targets' : ['soprano','alto','tenor','bass'], \
          '--outdir' : './predictions/umix/umix_baseline_csd_usecase2', \
          '--start' : 0.0, \
          '--duration' : -1.0, \
          '--model' : 'open-unmix/umix_baseline_csd_usecase2', \
          '--no-cuda' : False}
    }

@config_ingredient.named_config
def umix_all_usecase1():
  umix_cfg = {
          '--input' : './chorus_inputs/DCS_TPQuartetA_mix.wav', \
          '--targets' : ['soprano','alto','tenor','bass'], \
          '--outdir' : './predictions/umix/umix_all_usecase1', \
          '--start' : 0.0, \
          '--duration' : -1.0, \
          '--model' : 'open-unmix/umix_all_usecase1', \
          '--no-cuda' : False}
    }

@config_ingredient.named_config
def umix_all_usecase2():
  umix_cfg = {
          '--input' : './chorus_inputs/DCS_TPQuartetA_mix.wav', \
          '--targets' : ['soprano','alto','tenor','bass'], \
          '--outdir' : './predictions/umix/umix_all_usecase2', \
          '--start' : 0.0, \
          '--duration' : -1.0, \
          '--model' : 'open-unmix/umix_all_usecase2', \
          '--no-cuda' : False}
    }

#------ ConvTasnet ------#

config_convtasnet = {'use_gpu': 0,\
  'input_path': './chorus_inputs/DCS_TPQuartetA_mix.wav',\
  'exp_dir': './SATB/ALL_uc2/',\
   'n_save_ex': -1,\
    'sample_rate': 22020, \
    'train_conf': {'data': {'mode': 'min', '\
    nondefault_nsrc': None, \
    'sample_rate': 22020, \
    'task': 'sep_clean', \
    'train_dir': '/home/pc2752/share/Darius/Wave-U-Net/satb_dataset_all.hdf5', \
    'valid_dir': '/home/pc2752/share/Darius/Wave-U-Net/satb_dataset_all.hdf5'}, \
    'filterbank': {'kernel_size': 16, \
    'n_filters': 512,\
    'stride': 8},\
    'main_args': {'exp_dir': './SATB/ALL_uc2/', \
    'help': None}, \
    'masknet': {'bn_chan': 128, \
    'hid_chan': 512, \
    'mask_act': 'relu', \
    'n_blocks': 8, \
    'n_repeats': 3, \
    'n_src': 4, \
    'skip_chan': 128}, 'optim': {'lr': 0.001, \
    'optimizer': 'adam', \
    'weight_decay': 0.0}, \
    'positional arguments': {}, \
    'training': {'batch_size': 2, \
    'early_stop': False, \
    'epochs': 2000, 'half_lr': True, \
    'num_workers': 4}}}

#------ DPRNN ------#

config_dprnn = {'use_gpu': 0, 
'input_path': './chorus_inputs/DCS_TPQuartetA_mix.wav',\
'exp_dir': './SATB/ALL_uc1_DPRNN/', 
'n_save_ex': -1, 
'sample_rate': 22020, 
'train_conf': {'data': {'mode': 'min', 
'nondefault_nsrc': None, 
'sample_rate': 22020, 
'task': 'sep_clean', 
'train_dir': '/home/pc2752/share/Darius/Wave-U-Net/satb_dataset_all.hdf5', 
'valid_dir': '/home/pc2752/share/Darius/Wave-U-Net/satb_dataset_all.hdf5'}, 
'filterbank': {'kernel_size': 2, 
'n_filters': 64, 
'stride': 1}, 
'main_args': {'exp_dir': './SATB/ALL_uc1_DPRNN/', 
'help': None}, 
'masknet': {'bidirectional': True, 
'bn_chan': 128, 
'chunk_size': 250, 
'dropout': 0, 
'hid_size': 128, 
'hop_size': 125, 
'in_chan': 64, 
'mask_act': 'sigmoid', 
'n_repeats': 6, 
'n_src': 4, 
'out_chan': 64}, 
'optim': {'lr': 0.001, 
'optimizer': 'adam', 
'weight_decay': 1e-05}, 
'positional arguments': {}, 
'training': {'batch_size': 4, 
'early_stop': False, 
'epochs': 2000, 
'gradient_clipping': 5, 
'half_lr': True, 
'num_workers': 4}}}
