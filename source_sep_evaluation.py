import sys, os
import mir_eval
import glob
import numpy as np
import librosa
from csv import writer
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt 


def plot(results,title=''):

	# Expect Dict to be in the form: Dict[Metric][Song][Group]

	groups = ['soprano','alto','tenor','bass']
	column_names = ["group", "metric", "value"]
	df = pd.DataFrame(columns = column_names)

	for key in results.keys():
		for song_result in results[key]:
			print(song_result)
			for idx, value in enumerate(song_result):
				row = {'group':groups[idx],'metric':key,'value':value}
				df = df.append(row,ignore_index=True)

	df_sdr = df.loc[df['metric'] == 'sdr']
	df_sir = df.loc[df['metric'] == 'sir']
	df_sar = df.loc[df['metric'] == 'sar']

	fig, axes = plt.subplots(1, 3, sharex=True)
	fig.set_size_inches(10.5, 3.5)
	t = fig.suptitle(title, fontsize=16)
	t.set_color('black')
	t.set_position([.5, 1.05])

	ax = sns.boxplot(x="value", y="group", notch=True, data=df_sdr, 
    ax=axes[0],width=0.3,whis=1.0,fliersize=0.5,color='darkslategray',linewidth=2.0)
	ax.title.set_text('SDR')
	ax.set_xlim(-3, 15)
	ax.set_ylabel('Singing Groups')
	
	ax = sns.boxplot(x="value", y="group", notch=True, data=df_sir, 
	ax=axes[1],width=0.2,whis=1.0,fliersize=0.5,color='darkslategray',linewidth=2.0)
	ax.set_xlim(-3, 15)
	ax.title.set_text('SIR')

	ax = sns.boxplot(x="value", y="group", notch=True, data=df_sar, 
	ax=axes[2],width=0.2,whis=1.0,fliersize=0.5,color='darkslategray',linewidth=2.0)
	ax.set_xlim(-3, 15)
	ax.title.set_text('SAR')
	

	
	for ax in [axes[0], axes[1], axes[2]]:
		ax.set_xlabel('dB')
		ax.yaxis.grid(True)
		ax.spines['bottom'].set_color('black')
		ax.spines['top'].set_color('black') 
		ax.spines['right'].set_color('black')
		ax.spines['left'].set_color('black')
		ax.tick_params(axis='x', colors='black')
		ax.tick_params(axis='y', colors='black')
		ax.yaxis.label.set_color('black')
		ax.xaxis.label.set_color('black')
		ax.title.set_color('black')

		# for i,artist in enumerate(ax.artists):
		# 	# Set the linecolor on the artist to the facecolor, and set the facecolor to None
		# 	col = artist.get_facecolor()
		# 	artist.set_edgecolor(col)
		# 	artist.set_facecolor('darkslategray')

		# 	# Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
		# 	# Loop over them here, and use the same colour as above
		# 	for j in range(i*6,i*6+6):
		# 		line = ax.lines[j]
		# 		line.set_color(col)
		# 		line.set_mfc(col)
		# 		line.set_mec(col)

	#plt.tight_layout()
	plt.subplots_adjust(wspace=.55)
	plt.savefig(
	    os.path.join('./',title+'_'+"boxplot_results.png"),
	    bbox_inches='tight',
	    transparent=True)

# sdr = [[2,2,3,0],[1.4,2.6,3.7,4.6],[1.1,1.1,3.5,4.7],[1,2,3,4]]
# sir = [[11,12,13,14],[11,0,13,4],[11,2,13,14],[11,12,3,14]]
# sar = [[1,1,1,1],[1,2.5,3.7,1.2],[1,1,1,1],[1,2.5,3.7,1.2]]

# metrics = {'sdr':sdr,'sir':sir,'sar':sar}

# plot(metrics,title='test')
 
if len(sys.argv) > 3:
	
	reference_sources_path = sys.argv[1]
	estimated_sources_path = sys.argv[2]
	writeTofile = sys.argv[3]

	fs = 44100   

	ref_paths = glob.glob(os.path.join(reference_sources_path,"**/*.wav"),recursive=True)
	est_paths = glob.glob(estimated_sources_path+"/*.wav",recursive=False)

	source_order = ['soprano','alto','tenor','bass']

	model_num = estimated_sources_path.split('/')[-1]
	model = estimated_sources_path.split('/')[-2]
	# use_case = estimated_sources_path.split('/')[-3]

	# import pdb;pdb.set_trace()

	if len(ref_paths) == len(est_paths):

		# Extract dataset+song key and loop through each songs
		dst_song_keys = [os.path.basename(path).split('_')[0]+'_'+os.path.basename(path).split('_')[1] for path in ref_paths]
		dst_song_keys = list(set(dst_song_keys))

		#for mixes
		results_sdr = []
		results_sir = []
		results_sar = []

		for song_key in dst_song_keys:

			print('Processing Song: '+song_key+' ...')
			ref_paths_for_song = [path for path in ref_paths if song_key in os.path.basename(path)]
			est_paths_for_song = [path for path in est_paths if song_key in os.path.basename(path)]

			ref_paths_for_song_sorted = [path for x in source_order for path in ref_paths_for_song if x in path] 
			est_paths_for_song_sorted = [path for x in source_order for path in est_paths_for_song if x in path] 

			import pdb;pdb.set_trace()

			if len(ref_paths_for_song_sorted) == len(est_paths_for_song_sorted):

				reference_sources_matrix = []
				estimated_sources_matrix = []

				for ref_track, est_track in zip(ref_paths_for_song_sorted,est_paths_for_song_sorted):

					# Get the audio
					ref_audio, _ = librosa.load(ref_track, sr=fs)
					est_audio, _ = librosa.load(est_track, sr=fs)

					# Reshape if needed
					minim = min(len(est_audio), len(ref_audio))
					est_audio = est_audio[:minim] 
					ref_audio = ref_audio[:minim]

					# Add the source to the song list
					reference_sources_matrix.append(ref_audio)
					estimated_sources_matrix.append(est_audio)
					print('check')

				# Song stems matrix to be evaluated
				np_ref_matrix = np.asarray(reference_sources_matrix, dtype=np.float32)
				np_est_matrix = np.asarray(estimated_sources_matrix, dtype=np.float32)

				print('before')
				# Evaluate the song, get metrics
				sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(np_ref_matrix, np_est_matrix, compute_permutation=False)

				print('BSS Computed For Song: '+song_key+'\n')
				print('SDR: '+str(sdr))
				print('SIR: '+str(sir))
				print('SAR: '+str(sar))

				results_sdr.append(sdr)
				results_sir.append(sir)
				results_sar.append(sar)

			else:
				print('reference sources for song must have same shape as estimated sources')

		results_sdr = np.asarray(results_sdr, dtype=np.float32)
		results_sir = np.asarray(results_sir, dtype=np.float32)
		results_sar = np.asarray(results_sar, dtype=np.float32)

		results_sdr_mean = np.mean(results_sdr,axis=0)
		results_sir_mean = np.mean(results_sir,axis=0)
		results_sar_mean = np.mean(results_sar,axis=0)

		print('\nSDR:\n')
		print(results_sdr_mean)
		print('SIR:\n')
		print(results_sir_mean)
		print('SAR:\n')
		print(results_sar_mean)

		with open(writeTofile, 'a+', newline='') as write_obj:
				csv_writer = writer(write_obj) 
				csv_writer.writerow(['Parts','SDR','SAR','SIR']) 
				csv_writer.writerow(['Soprano',results_sdr_mean[0],results_sar_mean[0],results_sir_mean[0]]) 
				csv_writer.writerow(['Alto',   results_sdr_mean[1],results_sar_mean[1],results_sir_mean[1]])  
				csv_writer.writerow(['Tenor',  results_sdr_mean[2],results_sar_mean[2],results_sir_mean[2]])  
				csv_writer.writerow(['Bass',   results_sdr_mean[3],results_sar_mean[3],results_sir_mean[3]]) 
				csv_writer.writerow(['Average',np.mean(results_sdr_mean),np.mean(results_sar_mean),np.mean(results_sir_mean)])

		plot({'sdr':results_sdr,'sir':results_sir,'sar':results_sar},title='Wave-U-Net - Train: CSD - Test: DCS')

		sdr_df = pd.DataFrame(data=results_sdr,columns=source_order)
		sir_df = pd.DataFrame(data=results_sir,columns=source_order)
		sar_df = pd.DataFrame(data=results_sar,columns=source_order)

		sdr_df.to_pickle('./sdr_'+os.path.splitext(os.path.basename(writeTofile))[0]+'.pkl')
		sir_df.to_pickle('./sir_'+os.path.splitext(os.path.basename(writeTofile))[0]+'.pkl')
		sar_df.to_pickle('./sar_'+os.path.splitext(os.path.basename(writeTofile))[0]+'.pkl')


	else:
		print(ref_paths)
		print(est_paths)
		print('reference sources for set must have same shape as estimated sources')
