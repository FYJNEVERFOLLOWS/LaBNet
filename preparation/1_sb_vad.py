import torch
import sys
sys.path.append("/Work21/2021/fuyanjie/libs/speechbrain/speechbrain/pretrained")
from interfaces import VAD
import os
import glob

"""
RUN this script for trainset, devset and testset, respectively
"""

device = torch.device('cuda')
# device = torch.device('cpu')

VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")
VAD.to(device)

# wav_path_pattern = '/CDShare3/LibriSpeech/train-clean-100/*/*/*.flac'
wav_path_pattern = '/CDShare3/LibriSpeech/dev-clean/*/*/*.flac'
files = glob.glob(wav_path_pattern)

# output_folder = '/Work21/2021/fuyanjie/pycode/LaBNet/data/metadata/vad/train-clean-100/'
output_folder = '/Work21/2021/fuyanjie/pycode/LaBNet/data/metadata/vad/dev-clean/'

for audio_file in files:
    try:
        print(f'{audio_file}', flush=True)
        boundaries = VAD.get_speech_segments(audio_file)
        
        file_name = str(audio_file).split('/')[-1]
        spkid = file_name.split('-')[0]

        save_folder = os.path.join(output_folder, spkid)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, file_name.replace('.flac', '.txt'))
        print('save_path ', save_path, flush=True)
        # Print the output
        VAD.save_boundaries(boundaries, save_path=save_path)
    except Exception as e:
        print(f'Exception: {e}')
        continue