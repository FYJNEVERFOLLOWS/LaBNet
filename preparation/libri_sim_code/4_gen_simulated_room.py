import math
import multiprocessing as mp
import os
import random
import re
import shutil
import time
from concurrent.futures.process import ProcessPoolExecutor
from functools import partial
from joblib import Parallel, delayed
import sys
import wave
import glob
sys.path.append("/Work21/2021/fuyanjie/pycode/LaBNet/libri_sim_code")

import numpy as np
import soundfile as sf

from gen_room_para import gen_room_para, gen_mulchannel_data_random, gen_mulchannel_data_angle

MAX_WORKERS = 10
SEED = 42

def get_first_level_folder(dir):
    folder_list = []
    for entry in os.scandir(dir):
        if entry.is_dir():
            folder_list.append(entry.path)

    return folder_list


def generate_wav_list_from_lst_file(lst_file):
    with open(lst_file, "r", encoding='utf-8-sig') as f:
        file_list = f.readlines()

    return file_list


def save_audio_separately(audio, path, fs):
    channels, length = audio.shape

    for channel in range(channels):
        suffix = '_multichannel_' + str(channel) + '.wav'

        out_data = np.reshape(audio[channel, :], [length, 1])

        out_data = out_data.astype(np.int16)

        with wave.open(path+suffix, 'wb') as f:
            f.setframerate(fs)
            f.setsampwidth(2)
            f.setnchannels(1)
            f.writeframes(out_data.tostring())


def __sim_stationary_noise(folder, max_gen_num, wav_list, fs, speech_length):
    room_para_path = os.path.join(folder, 'room_para.npy')
    room_para = np.load(room_para_path, allow_pickle=True).item()

    # used_wav_num = random.randint(5, 10)
    used_wav_num = 1
    wav_samples = random.sample(wav_list, used_wav_num)
    print(f'num of wav_samples {len(wav_samples)}')

    # judge if multi-channel stationary noise wav files exist
    # file_list = os.listdir(os.path.join(folder, 'stationary_noise'))
    st_noise_dir = os.path.join(folder, 'stationary_noise')
    if os.path.exists(st_noise_dir) and len(os.listdir(st_noise_dir)) > 0:
        print(f'{st_noise_dir} is not empty!')
        return

    for num in range(max_gen_num):
        # 生成多个多通道平稳噪声数据
        mulchannel_audio_data_list = []
        for wav_file in wav_samples:
            multichannel_audio_data, _, angle_degree = gen_mulchannel_data_random(wav_file.strip(),
                                                                                room_para,
                                                                                folder,
                                                                                audio_type=0,
                                                                                fs=fs,
                                                                                segment_length=speech_length)
            multichannel_audio_data = multichannel_audio_data[:, 0:fs * speech_length]
            mulchannel_audio_data_list.append(multichannel_audio_data)

        # 对生成数据幅度进行一下调整，再进行叠加
        out_mulchannel_audio_data = np.zeros_like(mulchannel_audio_data_list[0], dtype=np.int16)
        for data in mulchannel_audio_data_list:
            out_mulchannel_audio_data += data

        # 将生成好的数据进行保存
        mic_num, _ = out_mulchannel_audio_data.shape
        save_folder = os.path.join(folder, 'stationary_noise')
        output_file_path = os.path.join(save_folder, 'stationary_noise_{}'.format(num))
        save_audio_separately(out_mulchannel_audio_data, output_file_path, fs=fs)


def sim_stationary_noise(base_dir, stationary_noise_lst_file, sim_room_num, max_stationary_noise_num, fs, room_transpose_prob, speech_length):
    '''
    利用多个平稳噪声模拟真实环境下的多通道平稳噪声，每个房间就仿真一条数据，主要目的是为了更好模拟不同环境下的平稳噪声
    :param base_dir: 根目录，用于保存不同房间的数据
    :param stationary_noise_lst_file: 平稳噪声列表文件
    :param sim_room_num: 模拟不同房间的数目
    :param max_stationary_noise_num: 最多使用平稳噪声的数目
    :param fs:
    :return:
    '''
    for i in range(sim_room_num):
        _ = gen_room_para(base_dir, room_transpose_prob=room_transpose_prob)

    room_folders = get_first_level_folder(base_dir)
    wav_list = list(generate_wav_list_from_lst_file(stationary_noise_lst_file))
    
    Parallel(n_jobs=MAX_WORKERS)(delayed(__sim_stationary_noise)(room_folder, max_stationary_noise_num, wav_list, fs, speech_length) for room_folder in room_folders)
    # with ProcessPoolExecutor(MAX_WORKERS) as ex:
    #     func = partial(__sim_stationary_noise, max_gen_num=max_stationary_noise_num, wav_list=wav_list,
    #                    fs=fs, speech_length=speech_length)
    #     ex.map(func, room_folders)


def __sim_speech(seed_idx, folder, libri_segments_dir, fs, cover_angle_range, multi_list, segment_length, speaker_list, angular_spacing):
    random_state = np.random.RandomState(seed_idx * SEED)

    room_para_path = os.path.join(folder, 'room_para.npy')
    room_para = np.load(room_para_path, allow_pickle=True).item()

    # if room_para['room_dim'][1] < 5:
    #     cover_angle_range = [20, 160]
    # elif room_para['room_dim'][1] >= 5 and room_para['room_dim'][1] < 7:
    #     cover_angle_range = [15, 165]
    # elif room_para['room_dim'][1] >= 7:
    #     cover_angle_range = [10, 170]
    # 声明多通道人声数据保存路径
    mulchannel_speech_folder = os.path.join(folder, 'speech')

    cover_angle_num = (cover_angle_range[1] - cover_angle_range[0]) // angular_spacing

    speaker_folder = ""
    speaker_folders = []
    speaker_lsts = []
    seen_speaker_lsts = []

    with open(speaker_list) as fid:
        for line in fid:
            seen_speaker_lsts.append(line.strip())

    if not multi_list:
        speaker_folder = libri_segments_dir
        print(f'speaker_folder {speaker_folder}', flush=True)
        speaker_lst = seen_speaker_lsts
        print(f'speaker_lst {speaker_lst}', flush=True)
        speaker_idx_lst = np.arange(len(speaker_lst))
        random_state.shuffle(speaker_idx_lst)
        speaker_idx_lst = speaker_idx_lst[:cover_angle_num * 2]

    # get the wav_id list
    wav_id_file_path = os.path.join(folder, 'speech_wav_id_lst.lst')
    if os.path.exists(wav_id_file_path):
        wav_id_lst = []
        with open(wav_id_file_path, 'r', encoding='utf-8') as f:
            wav_id_set = f.readlines()

            for wav_id in wav_id_set:
                wav_id = wav_id.strip()
                wav_id_lst.append(wav_id)
    else:
        wav_id_lst = []
    angle = cover_angle_range[0]
    index = 0
    while True:
        index %= len(speaker_idx_lst)
        k = speaker_idx_lst[index]
        index += 1
        wav_lst = glob.glob(os.path.join(speaker_folder, speaker_lst[k]) + '/*/*.flac')
        wav_name = random.sample(wav_lst, 1)[0]

        if speaker_lst[k] not in wav_id_lst:
            wav_path = os.path.join(speaker_folder, speaker_lst[k])
            wav_path = os.path.join(wav_name.split('-')[1], wav_name)

            # if angle >= 55:
            #     angular_spacing = 5
            # if angle >= 120:
            #     angular_spacing = 10
            # random_range = random.randint(0, angular_spacing - 1)
            # random_angle = angle + random_range
            random_angle = angle


            for dis_idx in range(3):
                multichannel_audio_data, wav_id, distance = gen_mulchannel_data_angle(wav_path,
                                                                                      room_para,
                                                                                      folder,
                                                                                      angle=random_angle,
                                                                                      distance_flag=dis_idx,
                                                                                      fs=fs,
                                                                                      segment_length=segment_length,
                                                                                      audio_type=1)

                max_speech_value = np.max(np.abs(multichannel_audio_data))


                if max_speech_value > 32767:
                    multichannel_audio_data = multichannel_audio_data / max_speech_value * 30000
                out_mulchannel_audio_data = multichannel_audio_data


                # 生成文件保存路径
                if dis_idx == 0:
                    distance_type = "close"
                elif dis_idx == 1:
                    distance_type = "middle"
                else:
                    distance_type = "far"
                file_name = "{}-{}m-{}".format(wav_id, distance, distance_type)
                angle_speech_folder = os.path.join(mulchannel_speech_folder, f"{random_angle}-{wav_id.split('-')[0]}")
                os.makedirs(angle_speech_folder, exist_ok=True)
                save_file_path = os.path.join(angle_speech_folder, file_name)

                save_audio_separately(out_mulchannel_audio_data, save_file_path, fs=fs)
                print(save_file_path + " finish!")

            wav_id_lst.append(wav_id)
            angle = angle + angular_spacing
            if angle >= cover_angle_range[1] + 1 - angular_spacing:
                break
        else:
            continue
    # wait for all the wav simulation finish, write wav_id_lst in the wav_id_file_path
    with open(wav_id_file_path, 'w', encoding='utf-8') as f:
        for wav_id in wav_id_lst:
            f.write(wav_id + '\n')

def sim_speech(base_dir, libri_segments_dir, fs, cover_angle_range, segment_length, speaker_list,
               angular_spacing):
    '''
    模拟多通道人声数据
    :param base_dir: 根目录
    :param libri_segments_dir: 单通道人声文件夹
    :param max_wav_num:
    :param fs:
    :return:
    '''
    room_folders = get_first_level_folder(base_dir)

    Parallel(n_jobs=MAX_WORKERS)(delayed(__sim_speech)(seed_idx, folder, libri_segments_dir, fs, cover_angle_range, False, segment_length, speaker_list, angular_spacing) for seed_idx, folder in enumerate(room_folders))

    # with ProcessPoolExecutor(MAX_WORKERS) as ex:
    #     func = partial(__sim_speech, libri_segments_dir=libri_segments_dir, fs=fs, cover_angle_range=cover_angle_range, multi_list=False,
    #                    segment_length=segment_length, speaker_list=speaker_list, angular_spacing=angular_spacing)
    #     ex.map(func, room_folders)
    # debug
    # for folder in room_folders:
    #     __sim_speech(folder, libri_segments_dir, fs, cover_angle_range, False, segment_length, speaker_list, angular_spacing)
    # debug


def __sim_non_stationary_noise(folder, sim_wav_num, wav_list, fs, cover_angle_range, speech_length, angular_spacing):
    room_para_path = os.path.join(folder, 'room_para.npy')
    room_para = np.load(room_para_path, allow_pickle=True).item()

    speech_path = os.path.join(folder, 'speech')
    angle_of_speech = os.listdir(speech_path)
    angle_of_speech = list(map(int, angle_of_speech))
    angle_of_speech.sort()
    print(f'angle_list_of_speech: {angle_of_speech}')

    # 声明多通道非平稳噪声数据保存路径
    mulchannel_nonstationary_noise_folder = os.path.join(folder, 'nonstationary_noise')

    cover_angle_num = (cover_angle_range[1] - cover_angle_range[0]) / angular_spacing
    cover_angle_num = math.floor(cover_angle_num)

    non_stationary_list = wav_list
    np.random.shuffle(non_stationary_list)
    non_stationary_list = non_stationary_list[:3 * cover_angle_num]

    angle = angle_of_speech[0]
    index = 0
    for wav in non_stationary_list:
        multichannel_audio_data, wav_id, distance = gen_mulchannel_data_angle(wav.strip(),
                                                                              room_para,
                                                                              folder,
                                                                              angle=angle,
                                                                              distance_flag=index % 3,
                                                                              fs=fs,
                                                                              segment_length=speech_length,
                                                                              audio_type=2)

        # out_mulchannel_audio_data = multichannel_audio_data - np.mean(multichannel_audio_data, axis=0)
        out_mulchannel_audio_data = multichannel_audio_data

        # 生成文件保存路径
        if index % 3 == 0:
            distance_type = "close"
        elif index % 3 == 1:
            distance_type = "middle"
        else:
            distance_type = "far"
        file_name = "{}-{}m-{}".format(wav_id, distance, distance_type)
        angle_speech_folder = os.path.join(mulchannel_nonstationary_noise_folder, str(angle))

        save_file_path = os.path.join(angle_speech_folder, file_name)

        save_audio_separately(out_mulchannel_audio_data, save_file_path, fs=fs)
        print(save_file_path + " finish!")

        if index % 3 == 2:
            angle = angle + angular_spacing
        index = index + 1



def sim_non_stationary_noise(base_dir, mono_nonstationary_noise_lst_file, max_wav_num, fs, cover_angle_range,
                             speech_length, angular_spacing):
    '''
    模拟多通道非平稳噪声
    :param base_dir: 根目录
    :param mono_nonstationary_noise_lst_file: 单通道非平稳噪声文件列表文件
    :param max_wav_num:
    :param fs: 采样率
    :return:
    '''
    room_folders = get_first_level_folder(base_dir)

    wav_list = list(generate_wav_list_from_lst_file(mono_nonstationary_noise_lst_file))
    if len(wav_list) > max_wav_num:
        sim_wav_num = max_wav_num
    else:
        sim_wav_num = len(wav_list)

    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        func = partial(__sim_non_stationary_noise, sim_wav_num=sim_wav_num, wav_list=wav_list, fs=fs,
                       cover_angle_range=cover_angle_range, speech_length=speech_length,
                       angular_spacing=angular_spacing)
        ex.map(func, room_folders)


def single_proc(stationary_noise_lst, libri_segments_dir, non_stationary_noise_lst, save_path,
                sim_room_num, stationary_noise_num, fs, speech_length, cover_angle_range, room_transpose_prob, speaker_list, augular_spacing):
    print("MAX_WORKERS:", MAX_WORKERS)

    # Start timing¬
    start_time = time.time()

    # make directory of output data
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        print(f'{save_path} already exists!')
        shutil.rmtree(save_path)
        print(f'remove {save_path}')

    # generate room parameters and stationary noise
    sim_stationary_noise(save_path, stationary_noise_lst, sim_room_num, stationary_noise_num, fs, room_transpose_prob, speech_length)
    print('Finish sim_stationary_noise')

    # generate multi-channel speech
    sim_speech(save_path, libri_segments_dir, fs, cover_angle_range, speech_length, speaker_list, augular_spacing)
    print('Finish sim_speech')

    # DO NOT INVOKE FOR NOW!
    # generate multi-channel non-stationary noise
    # sim_non_stationary_noise(save_path, non_stationary_noise_lst, non_stationary_num, fs, cover_angle_range, speech_length, augular_spacing)
    # print('Finish sim_non_stationary_noise')

    # End timing
    print('time spent: %s mins' % str((time.time() - start_time) / 60.0))


if __name__ == '__main__':
    '''
    convert single channel format to multi-channel(6-channel) format of single voice.

    :param stationary_noise_lst: contains the absolute path of stationary noise .wav
    :param libri_segments_dir: contains clean audio files
    :param non_stationary_noise_lst: contains the absolute path of non-stationary noise .wav
    :param save_path: the output path of multi-channel data
    :param sim_room_num: the number of simulated rooms, each room contains specified number of stationary noise, speech, and non-stationary noise
    :param stationary_noise_num: the upper bound of the kind of stationary noise we choose
    :param speech_num: the number of simulated speech in each room
    :param non_stationary_num: the number of simulated non-stationary noise in each room
    :param fs: sample rate
    :return: 
    '''

    # 设置路径参数
    # dataset_split = "train-clean-100"
    # dataset_split = "dev-clean"
    # dataset_split = "test-clean"
    dataset_split = "test-clean_wer"
    stationary_noise_lst = '/Work21/2020/yinhaoran/VCTK_simulated_data/list/clean_speech_3.lst'
    libri_segments_dir = f'/CDShare3/LibriSpeechSegments/{dataset_split}'
    non_stationary_noise_lst = '/non_stationary_noise.lst'

    speaker_lst = f'/Work21/2021/fuyanjie/pycode/LaBNet/data/libri_segments_list/{dataset_split}_spk.lst'

    # 设定合成参数
    sim_room_num = 10
    stationary_noise_num = 1
    fs = 16000 
    speech_length = 4
    cover_angle_range = [0, 180]
    angular_spacing = 1 # 角度最小间隔
    room_transpose_prob = 0.5

    save_path = f'/CDShare3/Libri-SIM/rooms_0129/{dataset_split}_{sim_room_num}rooms'

    single_proc(stationary_noise_lst, libri_segments_dir, non_stationary_noise_lst, save_path, sim_room_num,
                stationary_noise_num, fs, speech_length, cover_angle_range, room_transpose_prob,
                speaker_lst, angular_spacing)

