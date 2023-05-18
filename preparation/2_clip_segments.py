import os
import numpy as np
import soundfile as sf

"""
RUN this script for trainset, devset and testset, respectively
"""

SR = 16000
SEG_DURATION = 4 # duration in seconds
SEG_LEN = SEG_DURATION * SR

def clip_train_dev_test_segs(dataset_split):
    librispeech_path = f"/CDShare3/LibriSpeech/{dataset_split}"
    vad_results_path = f"/Work21/2021/fuyanjie/pycode/LaBNet/data/metadata/vad/{dataset_split}"
    output_libri_segments_dir = f"/CDShare3/LibriSpeechSegments/{dataset_split}"

    vad_res_spks = os.listdir(vad_results_path)
    for vad_res_spk in vad_res_spks:
        spkid = vad_res_spk
        vad_res_spk = os.path.join(vad_results_path, vad_res_spk)
        vad_res_utts = os.listdir(vad_res_spk)
        for vad_res_utt in vad_res_utts:
            uttid = vad_res_utt[:-4]
            chapid = uttid.split('-')[1]
            vad_res_utt = os.path.join(vad_res_spk, vad_res_utt)
            lines = open(vad_res_utt, 'r').readlines()
            start_idx = None
            end_idx = None
            if len(lines) == 1:
                seg_data = lines[0].split(' ')
                if float(seg_data[4]) < SEG_DURATION or seg_data[-1].startswith('N'):
                    continue
                print(f'seg_data {seg_data}', flush=True)
                start_idx = 0
                end_idx = float(seg_data[4]) * SR
            else:
                for row_idx, line in enumerate(lines):
                    seg_data = line.split(' ')
                    print(f'seg_data {seg_data}', flush=True)
                    if seg_data[-1].startswith('S'):
                        duration = float(seg_data[4]) - float(seg_data[2])
                        if duration > 4:
                            start_idx = float(seg_data[2]) * SR
                            end_idx = float(seg_data[4]) * SR
                if not start_idx or not end_idx:
                    continue

            utt_path = os.path.join(librispeech_path, spkid, chapid, uttid + '.flac')
            wave_data, sr = sf.read(utt_path)
            end_idx = int(start_idx) + SEG_LEN
            seg_data = wave_data[int(start_idx):end_idx]
            dst_parent_dir = os.path.join(output_libri_segments_dir, spkid, chapid)
            os.makedirs(dst_parent_dir, exist_ok=True)
            dst_path = os.path.join(dst_parent_dir, uttid + '.flac')
            print(f'dst_path {dst_path}', flush=True)
            sf.write(dst_path, seg_data, 16000)

def clip_wer_test_segs(dataset_split):
    librispeech_path = f"/CDShare3/LibriSpeech/{dataset_split}"
    vad_results_path = f"/Work21/2021/fuyanjie/pycode/LaBNet/data/metadata/vad/{dataset_split}"
    output_libri_segments_dir = f"/CDShare3/LibriSpeechSegments/{dataset_split}_wer"

    vad_res_spks = os.listdir(vad_results_path)
    vad_res_spks.sort()
    index = 0
    for vad_res_spk in vad_res_spks:
        spkid = vad_res_spk
        vad_res_spk = os.path.join(vad_results_path, vad_res_spk)
        vad_res_utts = os.listdir(vad_res_spk)
        vad_res_utts.sort()
        for vad_res_utt in vad_res_utts:
            try:
                index += 1
                random_state = np.random.RandomState(index)

                uttid = vad_res_utt[:-4]
                chapid = uttid.split('-')[1]
                vad_res_utt = os.path.join(vad_res_spk, vad_res_utt)
                lines = open(vad_res_utt, 'r').readlines()
                print(f'vad_res_utt {vad_res_utt}', flush=True)

                seg_data = lines[-1].split(' ')
                if float(seg_data[4]) > SEG_DURATION:
                    continue
                print(f'seg_data {seg_data}', flush=True)

                utt_path = os.path.join(librispeech_path, spkid, chapid, uttid + '.flac')
                wave_data, sr = sf.read(utt_path)
                # print(f'{wave_data.shape} {wave_data.shape[-1]}')
                if wave_data.shape[-1] > SEG_LEN or wave_data.shape[-1] < SR * 3.5:
                    continue
                # randomly padding to 4 secs long segment
                wave_padding_data = np.zeros(SEG_LEN)

                start_idx = random_state.randint(0, SEG_LEN - wave_data.shape[-1])
                end_idx = start_idx + len(wave_data)
                wave_padding_data[start_idx:end_idx] = wave_data
                seg_data = wave_padding_data
                print(f'start_idx {start_idx} end_idx {end_idx}', flush=True)

                dst_parent_dir = os.path.join(output_libri_segments_dir, spkid, chapid)
                os.makedirs(dst_parent_dir, exist_ok=True)
                dst_path = os.path.join(dst_parent_dir, uttid + '.flac')
                print(f'dst_path {dst_path}', flush=True)
                sf.write(dst_path, seg_data, 16000)
            except Exception as e:
                print(f'Exception: {e}')
                continue

if __name__ == '__main__':
    # dataset_split = 'train-clean-100'
    # clip_train_dev_test_segs(dataset_split)
    # dataset_split = 'dev-clean'
    # clip_train_dev_test_segs(dataset_split)
    # dataset_split = 'test-clean'
    # clip_train_dev_test_segs(dataset_split)
    dataset_split = 'test-clean'
    clip_wer_test_segs(dataset_split)
