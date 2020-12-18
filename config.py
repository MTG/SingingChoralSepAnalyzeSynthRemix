cnfig_unet_baseline_csd_usecase2 = {'model_config':\
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
 'model_path': './models/968405-112000', \
 'input_path': '/home/pc2752/share//Darius/Wave-U-Net/test_set_mixes/dcs/DCS_TPQuartetA_mix.wav',\
 'experiment_id': 'unet_baseline_csd_usecase2'}

cnfig_waveunet_all_usecase2 = {'model_config': {'musdb_path': './data/musdb18', 
'estimates_path': './Source_Estimates', 
'data_path': 'data', 
'satb_path_train': '../data/satb_dst/train/raw_audio', 
'satb_path_valid': '../data/satb_dst/valid/raw_audio', 
'satb_path_test': '../data/satb_dst/test/raw_audio', 
'satb_hdf5_filepath': './satb_dataset_only_csd.hdf5', 
'satb_debug': False, 
'satb_use_case': 2, 
'model_base_dir': 'checkpoints', 
'log_dir': 'logs', 
'batch_size': 16, 
'init_sup_sep_lr': 0.0001, 
'epoch_it': 2000, 
'cache_size': 4000, 
'num_workers': 4, 
'num_snippets_per_track': 100, 
'num_layers': 12, 
'filter_size': 15, 
'merge_filter_size': 5, 
'input_filter_size': 15, 
'output_filter_size': 1, 
'num_initial_filters': 24, 
'num_frames': 16384, 
'expected_sr': 22050, 
'mono_downmix': True, 
'output_type': 'direct', 
'output_activation': 'tanh', 
'context': False, 
'network': 'unet', 
'upsampling': 'linear', 
'task': 'satb', 
'augmentation': True, 
'raw_audio_loss': True, 
'worse_epochs': 50, 
'source_names': ['soprano', 
'alto', 
'tenor', 
'bass'], 
'num_sources': 4, 
'num_channels': 1}, 
 'model_path': './models/407295-546000', \
 'input_path': './chorus_inputs/DCS_TPQuartetA_mix.wav',\
'experiment_id': 'unet_baseline_csd_usecase2'}


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
