import librosa
import os
import math
import soundfile as sf

def generate_flac_list(libri_segments_dir, output_list_path):
    spkids = os.listdir(libri_segments_dir)
    count = 0
    with open(output_list_path, 'w', encoding='utf-8') as output_f:
        for spkid in spkids:
            spk_dir = os.path.join(libri_segments_dir, spkid)
            chapids = os.listdir(spk_dir)
            for chapid in chapids:
                chap_dir = os.path.join(spk_dir, chapid)
                wav_names = os.listdir(chap_dir)
                for wav_name in wav_names:
                    wav_path = os.path.join(chap_dir, wav_name)
                    print(f'wav_path {wav_path}')
                    output_f.write(wav_path + '\n')
                    count += 1
                    print('Finish count:{} | {}'.format(count, wav_path))

def generate_spk_list(libri_segments_dir, output_list_path):
    spkids = os.listdir(libri_segments_dir)
    spkids.sort()
    with open(output_list_path, 'w', encoding='utf-8') as output_f:
        for spkid in spkids:
            output_f.write(spkid + '\n')

def summarize_transcription(librispeech_path, libri_segments_dir, output_txt_path):
    spkids = os.listdir(libri_segments_dir)
    count = 0
    with open(output_txt_path, 'w', encoding='utf-8') as output_f:
        for spkid in spkids:
            spk_dir = os.path.join(libri_segments_dir, spkid)
            chapids = os.listdir(spk_dir)
            for chapid in chapids:
                chap_dir = os.path.join(spk_dir, chapid)
                wav_names = os.listdir(chap_dir)
                uttids = []
                for wav_name in wav_names:
                    uttid = wav_name[:-5]
                    print(f'uttid {uttid}')
                    uttids.append(uttid)
                src_txt_path = os.path.join(librispeech_path, spkid, chapid, f'{spkid}-{chapid}.trans.txt')
                src_txt = open(src_txt_path, 'r').readlines()
                for line in src_txt:
                    line = line.strip()
                    uttid, txt = line.split(' ', 1)
                    print(f'uttid: {uttid} txt: {txt}')
                    if uttid in uttids:
                        output_f.write(uttid + ' ' + txt + '\n')
                        count += 1
                        print('Finish count:{} | {}'.format(count, uttid))


if __name__ == '__main__':
    # dataset_split = 'test-clean'
    dataset_split = 'test-clean_wer'
    libri_segments_dir = f'/CDShare3/LibriSpeechSegments/{dataset_split}'
    output_flac_list_path = f'/Work21/2021/fuyanjie/pycode/LaBNet/data/libri_segments_list/{dataset_split}.lst'
    output_spk_list_path = f'/Work21/2021/fuyanjie/pycode/LaBNet/data/libri_segments_list/{dataset_split}_spk.lst'
    generate_flac_list(libri_segments_dir, output_flac_list_path)
    generate_spk_list(libri_segments_dir, output_spk_list_path)

    librispeech_path = f"/CDShare3/LibriSpeech/test-clean"
    libri_segments_dir = '/CDShare3/LibriSpeechSegments/test-clean_wer'
    output_txt_path = '/CDShare3/LibriSpeechSegments/test-clean_wer/test-clean_wer.txt'
    summarize_transcription(librispeech_path, libri_segments_dir, output_txt_path)