# -*- coding:utf-8 _*-
from itertools import groupby
import json
import os
import pickle
import random
from acoustics.signal import highpass

import librosa
import numpy as np
import soundfile as sf

SEED = 7
random.seed(SEED)
sr = 16000

def audioread(path, fs=16000):
    wave_data, sr = sf.read(path)
    if sr != fs:
        wave_data = librosa.resample(wave_data, sr, fs)
    return wave_data


def generate_vad_label(orig_signal, sr = 16000, win_len = 0.032, hop_len = 0.016):
    original_signal = orig_signal / np.max(orig_signal) * 25000
    original_signal = highpass(original_signal, 100, sr, order =8)

    vad = np.ones(int(len(original_signal)/(hop_len * sr)))

    frame_shift = int(hop_len * sr) # 256
    frame_len = int(win_len * sr) # 512

    value_threshold = 110

    sig_100ms_len = int(0.01 * sr)
    frame_max_value = np.max(original_signal[:sig_100ms_len])
    if frame_max_value > value_threshold:
        value_threshold = frame_max_value
    
    for i in range(len(vad)):
        frame_data = original_signal[i * frame_shift : (i + 1) * frame_shift] # old: i -> i+1, changed： i -> i + length
        frame_data = np.abs(frame_data)
        frame_data = np.where(frame_data > value_threshold,0,1)
        if np.sum(frame_data) > frame_shift * 0.9 and vad[i] == 1:
            vad[i] = 0
    
    mid_len_speech = round(0.1 / hop_len + 0.1)
    mid_len_non_speech = round(0.05 / hop_len + 0.1)
    for i in range(len(vad)):
        if i < len(vad) - mid_len_speech - 2:
            if vad[i] == 0 and vad[i+mid_len_speech+1] == 0 and sum(vad[i+1:i+mid_len_speech+1]<mid_len_speech):
                vad[i+1:i+mid_len_speech+1] = 0
            
            if vad[i] == 1 and vad[i+mid_len_non_speech+1] == 1 and sum(vad[i+1:i+mid_len_non_speech+1]<mid_len_non_speech):
                vad[i+1:i+mid_len_non_speech+1]=1
    vad = vad.reshape(-1)

    return vad

def generate_angle_list(generate_angle_num, angle_range=None, min_interval=5):
    '''
    :param generate_angle_num:
    :param angle_range:
    :param min_interval:
    :return:
    '''
    angle_min = angle_range[0]
    angle_max = angle_range[1]
    available_list = []
    result = []
    count = 0
    if angle_max - angle_min < (generate_angle_num - 1) * min_interval:
        raise Exception
    available_list.append([angle_min, angle_max])
    while count < generate_angle_num:
        select_index = random.randint(0, len(available_list) - 1)
        selected_range = available_list[select_index]
        select_angle_min = selected_range[0]
        select_angle_max = selected_range[1]

        select_angle = random.randint(select_angle_min, select_angle_max)
        result.append(select_angle)
        count += 1

        available_list.remove([select_angle_min, select_angle_max])
        if select_angle - min_interval > select_angle_min:
            available_list.append([select_angle_min, select_angle - min_interval])
        if select_angle + min_interval < select_angle_max:
            available_list.append([select_angle + min_interval, select_angle_max])
    return result

def data_mix_angle_wav_cos(generate_sample_num, max_source_num, input_single_source_folder_path, out_folder_path, add_noise=0.2):
    '''
    从多个混合音频中分离出一个目标角度的音频
    :param generate_sample_num:
    :param max_source_num:
    :param input_single_source_folder_path:
    :param out_folder_path:
    :param add_noise:
    :return:
    '''
    # 3 audio types for one room
    stationary_noise_flag = 'stationary_noise'
    speech_flag = 'speech'
    # non_stationary_noise_flag = 'nonstationary_noise'

    all_log = {}
    num_samples = np.zeros(4)
    max_num = 0
    dists = []
    sqrt_sample_num = int(np.sqrt(generate_sample_num))
    if not os.path.exists(out_folder_path):
        for folder_idx in range(sqrt_sample_num + 1):
            os.makedirs(os.path.join(out_folder_path, f'{folder_idx*sqrt_sample_num}-{(folder_idx+1)*sqrt_sample_num-1}'), exist_ok=True)
    else:
        exist_folders = os.listdir(out_folder_path)
        for folder in exist_folders:
            num_part = folder.split('_')[0]
            num = int(num_part.split('-')[1])
            if num > max_num:
                max_num = num
        print('existing samples: ', max_num)

    sample_pos = 0
    sample_idx = 0
    try:
        while 1:
            if sample_pos >= generate_sample_num:
                print(f'num_samples < 15: {num_samples[0]} 15-45 {num_samples[1]} 45-90 {num_samples[2]} >90 {num_samples[3]}', )
                break
            sample_log = {}

            random.seed(sample_idx)
            random_state = np.random.RandomState(sample_idx)
            sample_idx += 1

            # randomly choose a room
            rooms = os.listdir(input_single_source_folder_path)
            room = rooms[random.randint(0, len(rooms) - 1)]
            file_1st = input_single_source_folder_path + os.sep + room
            # get RIR
            tmp = room.split('_')
            rir = float(tmp[-1])
            file_2nd = file_1st # no reverb
            sample_log['room'] = room
            sample_log['rir'] = rir
            speech = file_1st + os.sep + speech_flag
            # non_stationary = file_1st + os.sep + non_stationary_noise_flag
            stationary = file_1st + os.sep + stationary_noise_flag

            stationary_noise_log = {}
            stationary_list = os.listdir(stationary)
            choose_stationary = random.sample(stationary_list, 1)[0]

            stationary_noise_log["wave_path"] = stationary + os.sep + choose_stationary.split('multichannel')[0] + 'multichannel'

            SNR = [10,20]

            stationary_noise_log["SNR"] = random.randint(SNR[0], SNR[1])
            sample_log["stationary_noise"] = stationary_noise_log

            if not os.path.exists(speech):
                continue
            all_speech = os.listdir(speech)
            # print("all_speech ", all_speech)
            
            if len(all_speech) < max_source_num:
                continue

            source_list = []
            angle_list = []
            spkid_list = []
            available_list = all_speech
            for i in range(max_source_num):
                chosen_subfolder = random.sample(available_list, 1)[0]
                chosen_angle, chosen_spkid = chosen_subfolder.split('-')
                source_list.append(chosen_subfolder)
                angle_list.append(chosen_angle)
                spkid_list.append(chosen_spkid)
                new_available_list = []
                for available_subfolder in available_list:
                    available_angle, available_spkid = available_subfolder.split('-')
                    if available_spkid not in spkid_list:
                    # if np.abs(int(available_angle)-int(chosen_angle))>5 and available_spkid not in spkid_list:
                        new_available_list.append(available_subfolder)
                available_list = new_available_list

            SIR = [-10, 10]

            for index, source in enumerate(source_list):
                angle = float(source.split('-')[0])
                sample_source_log = {}
                # 从某个房间的某种混响下选取 single_source_num 个角度进行混合，每个角度选取一个 wav 文件，保证角度之间的 wav 不重复
                wav_path = speech + os.sep + source_list[index]

                choosen_wav_list = os.listdir(wav_path)
                if index == 1:
                    if 'far' in choosen_wav:
                        close_or_middle = random_state.choice(['close', 'middle', 'far'], 1, p=[0.4, 0.3, 0.3])[0]
                        choosen_wav = [wavname for wavname in choosen_wav_list if close_or_middle in wavname][0]
                    else:
                        choosen_wav = [wavname for wavname in choosen_wav_list if 'far' in wavname][0]
                else:
                    choosen_wav = random.sample(choosen_wav_list, 1)[0]

                choosen_wav_path = choosen_wav.split('multichannel')
                wav_path = wav_path + os.sep + choosen_wav_path[0] + 'multichannel'
                sample_source_log['wave_path'] = wav_path
                sample_source_log['azimuth'] = angle

                # compute azimuth w.r.t. all mics
                s2m_dist = float(wav_path.split('/')[-1].split('-')[3].replace('m', ''))
                print(f"s2m_dist {s2m_dist} wav_path {wav_path}")
                mic_itvals = [0.14, 0.1, 0.06]
                rad = angle / 180.0 * np.pi
                for idx, mic_itval in enumerate(mic_itvals):
                    third_side = np.sqrt(mic_itval ** 2 + s2m_dist ** 2 - 2 * mic_itval * s2m_dist * np.cos(rad))
                    azi_rad = np.arccos(np.clip((third_side ** 2 + mic_itval ** 2 - s2m_dist ** 2) / (2 * third_side * mic_itval), -1+1e-8, 1-1e-8))
                    azimuth = azi_rad * 180.0 / np.pi
                    if idx == 0:
                        sample_source_log['azimuth1'] = round(180 - round(azimuth, 2), 2)
                    elif idx == 1:
                        sample_source_log['azimuth2'] = round(180 - round(azimuth, 2), 2)
                    elif idx == 2:
                        sample_source_log['azimuth3'] = round(180 - round(azimuth, 2), 2)
                rad = (180 - angle) / 180.0 * np.pi
                for idx, mic_itval in enumerate(mic_itvals):
                    third_side = np.sqrt(mic_itval ** 2 + s2m_dist ** 2 - 2 * mic_itval * s2m_dist * np.cos(rad))
                    azi_rad = np.arccos(np.clip((third_side ** 2 + mic_itval ** 2 - s2m_dist ** 2) / (2 * third_side * mic_itval), -1+1e-8, 1-1e-8))
                    azimuth = azi_rad * 180.0 / np.pi
                    if idx == 0:
                        sample_source_log['azimuth6'] = round(azimuth, 2)
                    elif idx == 1:
                        sample_source_log['azimuth5'] = round(azimuth, 2)
                    elif idx == 2:
                        sample_source_log['azimuth4'] = round(azimuth, 2)
                
                if index != 0:
                    sample_source_log["SIR"] = random.randint(SIR[0], SIR[1])
                wave = audioread(wav_path + '_0.wav')
                wave = wave[0:4*sr] # 4s
                vad_label = generate_vad_label(wave)

                sample_source_log['vad_label'] = list(vad_label)
                sample_log["source" + str(index)] = sample_source_log

            angular_dist = abs(int(sample_log['source0']['azimuth']) - int(sample_log['source1']['azimuth']))
            # if angular_dist <= 5:
            #     continue
            if angular_dist > 15:
                if np.random.rand() > 0.5:
                    continue
            # source-to-microphone distance
            s2m_dist1 = float(sample_log['source0']['wave_path'].split('/')[-1].split('-')[3].replace('m', '')) * 100
            s2m_dist2 = float(sample_log['source1']['wave_path'].split('/')[-1].split('-')[3].replace('m', '')) * 100
            sample_log['source0']['s2m_dist'] = int(s2m_dist1)
            sample_log['source1']['s2m_dist'] = int(s2m_dist2)        
            # distance between two sources
            social_dist = np.round(np.sqrt(np.square(s2m_dist1) + np.square(s2m_dist2) - 2 * s2m_dist1 * s2m_dist2 * np.cos(angular_dist / 180 * np.pi)), 2)
            sample_log['social_dist'] = social_dist
            # print(f's2m_dist1 {int(s2m_dist1)} s2m_dist2 {int(s2m_dist2)} social_dist {social_dist}', flush=True)

            if social_dist <= 100:
                continue
            if angular_dist <= 15:
                num_samples[0] += 1
            elif angular_dist > 15 and angular_dist < 45:
                num_samples[1] += 1
            elif angular_dist >= 45 and angular_dist < 90:
                num_samples[2] += 1
            elif angular_dist >= 90:
                num_samples[3] += 1

            sample_pos += 1
            dists.append(int(s2m_dist1))
            dists.append(int(s2m_dist2))
            print(f'sample_pos {sample_pos} source_list {source_list} angular_dist {angular_dist} s2m_dist1 {int(s2m_dist1)} s2m_dist2 {int(s2m_dist2)} social_dist {social_dist}', flush=True)

            folder_idx = (max_num + sample_pos) // sqrt_sample_num
            output_subfolder = out_folder_path + os.sep + f'{folder_idx*sqrt_sample_num}-{(folder_idx+1)*sqrt_sample_num-1}'
            write_file_path = output_subfolder + os.sep + f'sample-{max_num+sample_pos}-' + room + '.json'
            os.makedirs(output_subfolder, exist_ok=True)
            with open(write_file_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(sample_log, ensure_ascii=False))
    except Exception as e:
        print(f'e {e}')

    print('----- Statistics about the source-to-microphone distance -----')
    for k, g in groupby(sorted(dists), key=lambda x: x//50):
        print('{}-{}: {}'.format(k*50, (k+1)*50-1, len(list(g))))
    save_path = os.path.dirname(out_folder_path) + os.sep + out_folder_path.split('/')[-1] + '.pkl'
    dists_dict = {"dists": dists}
    pkl_file = open(save_path, 'wb')
    pickle.dump(dists_dict, pkl_file)


if __name__ == '__main__':
    # set_sample_num = 40000
    set_sample_num = 3000
    set_source_max_num = 2

    # dataset_split = "train-clean-100"
    # dataset_split = "dev-clean"
    dataset_split = "test-clean_0226"
    # dataset_split = "test-clean_wer"
    # input_single_source_folder_path = f'/CDShare3/Libri-SIM/rooms_0129/{dataset_split}_10rooms' # 包含多个房间的输入目录
    input_single_source_folder_path = f'/CDShare3/Libri-SIM/rooms_0129/test-clean_10rooms' # 包含多个房间的输入目录
    out_folder_path = f'/CDShare3/Libri-SIM/jsons_0129/{dataset_split}' # json输出目录
    data_mix_angle_wav_cos(set_sample_num, set_source_max_num, input_single_source_folder_path,
                           out_folder_path, add_noise=0.0)



