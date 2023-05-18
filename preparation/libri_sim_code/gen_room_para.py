import os
import random
import re

import pyroomacoustics as pra
import numpy as np
import soundfile as sf

MIN_ROOM_WIDTH= 3
MAX_ROOM_WIDTH = 9
MIN_ROOM_LEN = 4
MAX_ROOM_LEN = 12
MIN_ROOM_HEIGHT = 2.5
MAX_ROOM_HEIGHT = 5

EPS = 1e-8
SEED = 77

# 随机生成一个房间参数，生成对应的文件夹，并保存房间参数
def gen_room_para(dir, room_transpose_prob):
    '''
    :param dir: the save path of room
    :param room_transpose_prob: the probability of whether to switch the room length and the room width
    :return:
    '''

    room_para = dict()

    if random.uniform(0, 1) < room_transpose_prob:
        room_length = random.randint(MIN_ROOM_LEN, MAX_ROOM_LEN)
        room_width = np.round(random.uniform(max(room_length / 2, MIN_ROOM_WIDTH), room_length), 2)
    else:
        room_width = random.randint(MIN_ROOM_LEN, MAX_ROOM_LEN)
        room_length = np.round(random.uniform(max(room_width / 2, MIN_ROOM_WIDTH), room_width), 2)

    room_height = np.round(random.uniform(MIN_ROOM_HEIGHT, MAX_ROOM_HEIGHT), 2)

    room_para['room_dim'] = [room_length, room_width, room_height]

    max_room_size = max(room_length, room_width)
    # 根据房间大小随机生成rt60
    if max_room_size >= 4 and max_room_size < 8:
        rt60_tgt = random.uniform(0.3, 0.6)
    elif max_room_size >= 8 and max_room_size < 10:
        rt60_tgt = random.uniform(0.4, 0.7)
    elif max_room_size >= 10:
        rt60_tgt = random.uniform(0.5, 0.8)

    rt60_tgt = np.round(rt60_tgt, 3)

    room_para['rt60'] = rt60_tgt

    folder_name = 'room_{}_{}_{}_rt60_{}'.format(room_length, room_width, room_height, rt60_tgt)

    folder_path = os.path.join(dir, folder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 保存房间参数
    room_para_file_path = os.path.join(folder_path, 'room_para.npy')
    np.save(room_para_file_path, room_para)

    return room_para


def gen_mulchannel_data_random(wave_file_path, room_para, folder, audio_type, fs, segment_length):
    '''
    利用 Pyroomacoustics 生成多通道音频文件
    :param wave_file_path: 单通道音频文件路径
    :param room_para: 房间参数
    :param folder: 保存生成多通道声音文件的文件夹路径
    :param audio_type: 0：平稳噪声，1：人声，2：非平稳噪声
    :param fs: 音频采样率
    :return: 多通道音频data
    '''
    # 根据房间信息生成麦克风阵列位置
    room_dim = room_para['room_dim']
    rt60_tgt = room_para['rt60']
    room_length = room_dim[0]
    room_width = room_dim[1]
    room_height = room_dim[2]
    mic_locations = np.c_[
        [0.5, room_width / 2 - 0.14, 2],
        [0.5, room_width / 2 - 0.1, 2],
        [0.5, room_width / 2 - 0.06, 2],
        [0.5, room_width / 2 + 0.06, 2],
        [0.5, room_width / 2 + 0.1, 2],
        [0.5, room_width / 2 + 0.14, 2],
    ]

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
    # 根据参数创建房间
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order, air_absorption=True, humidity=50)
    audio_data, sr = sf.read(wave_file_path, dtype=np.int16)

    target_length = segment_length * sr
    if len(audio_data) < target_length:
        pad_audio_data = np.zeros(target_length)

        start = random.randint(0, target_length - len(audio_data))
        end = start + len(audio_data)

        pad_audio_data[start:end] = audio_data
        audio_data = pad_audio_data

    if sr != fs:
        raise ValueError("input wav file samplerate is not {}".format(fs))

    # 根据 wave 文件名获取说话人ID
    wave_file_name = wave_file_path.split('/')[-1]

    # 随机生成一个声源角度及距离
    if audio_type == 0:
        source_location = np.array([random.uniform(0, room_length),
                                    random.uniform(0, room_width),
                                    random.uniform(0, room_height)])
    elif audio_type == 1:
        source_location = np.array([random.uniform(mic_locations[0][0], room_length),
                                    random.uniform(0.5, room_width - 0.5),
                                    random.uniform(1.4, 1.8)])
    else:
        source_location = np.array([random.uniform(0.5, room_length),
                                    random.uniform(0, room_width),
                                    random.uniform(0, room_height)])
    if source_location[1] < room_width / 2:
        target_angle = np.arctan((source_location[0] - 0.5) / (room_width / 2 - source_location[1]))
    else:
        target_angle = np.pi / 2 + np.arctan((source_location[1] - room_width / 2) / (source_location[0] - 0.5))

    # 将弧度转为度
    angle_degree = int(target_angle * 180 / np.pi)

    # 生成多通道数据
    c = 345
    dist = np.linalg.norm(source_location - mic_locations[:, 0])

    if audio_type == 0:
        delay = 0
    else:
        delay = dist / c

    # 将声源放置在房间中
    room.add_source(source_location, signal=audio_data, delay=delay)

    # 将麦克风阵列放置在房间中
    room.add_microphone_array(mic_locations)

    # Run the simulation
    room.simulate()

    # 得到仿真的语音信号
    orig_max_value = np.max(np.abs(audio_data))
    multichannel_audio_data = room.mic_array.signals[:, 0:len(audio_data)]
    # multichannel_audio_data = multichannel_audio_data.astype(np.int16)

    multichannel_audio_data = multichannel_audio_data / np.max(np.abs(multichannel_audio_data)) * orig_max_value
    multichannel_audio_data = multichannel_audio_data.astype(np.int16)

    # print(dist, rt60_tgt, max_order, orig_max_value, np.max(np.abs(multichannel_audio_data)))

    # 生成对应的文件夹
    if audio_type == 0:
        new_folder = os.path.join(folder, 'stationary_noise')
    elif audio_type == 1:
        new_folder = os.path.join(folder, 'speech')
    else:
        new_folder = os.path.join(folder, 'nonstationary_noise')

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    # 生成对应文件名
    if audio_type == 0:
        # wav_id = wave_file_name.replace('.wav', '')
        return multichannel_audio_data, -1, angle_degree
    elif audio_type == 1:
        wav_id = re.findall('\d+', wave_file_name.split('/')[-1])[0]
        # wav_id = wave_file_name.replace('.wav', '') # for FYJ
        return multichannel_audio_data, wav_id, angle_degree
    else:
        wav_id = wave_file_name.replace('.wav', '')
        return multichannel_audio_data, wav_id, angle_degree

def gen_mulchannel_data_angle(wave_file_path, room_para, folder, angle, distance_flag, fs, segment_length, audio_type):
    '''
    利用 Pyroomacoustics 生成多通道音频文件
    :param wave_file_path: 单通道音频文件路径
    :param room_para: 房间参数
    :param folder: 保存生成多通道声音文件的文件夹路径
    :param audio_type: 0：平稳噪声，1：人声，2：非平稳噪声
    :param fs: 音频采样率
    :return: 多通道音频data
    '''
    # 根据房间信息生成麦克风阵列位置
    room_dim = room_para['room_dim']
    rt60_tgt = room_para['rt60']
    room_length = room_dim[0]
    room_width = room_dim[1]
    room_height = room_dim[2]
    room_mid = room_width / 2
    mic_locations = np.c_[
        [0.5, room_width / 2 - 0.14, 2],
        [0.5, room_width / 2 - 0.1, 2],
        [0.5, room_width / 2 - 0.06, 2],
        [0.5, room_width / 2 + 0.06, 2],
        [0.5, room_width / 2 + 0.1, 2],
        [0.5, room_width / 2 + 0.14, 2],
    ]

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
    # 根据参数创建房间
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order, air_absorption=True, humidity=50)

    audio_data, sr = sf.read(wave_file_path, dtype=np.int16)

    target_length = segment_length * sr
    if len(audio_data) < target_length:
        pad_audio_data = np.zeros(target_length)

        start = random.randint(0, target_length - len(audio_data))
        end = start + len(audio_data)

        pad_audio_data[start:end] = audio_data
        audio_data = pad_audio_data

    if sr != fs:
        raise ValueError("input wav file samplerate is not {}".format(fs))

    # 根据 wave 文件名获取说话人ID
    wave_file_name = wave_file_path.split('/')[-1]
    min_distance = 0.5

    # 随机生成一个声源角度及距离
    if angle < 90:
        rad = angle / 180 * np.pi
        max_distance = min(room_mid / (np.cos(rad) + EPS), (room_length - 0.5) / (np.sin(rad) + EPS)) - 0.5
        one_third_distance = (max_distance - min_distance) / 3
        # relative_distance = random.uniform(distance_flag * one_third_distance, (distance_flag + 1) * one_third_distance)
        relative_distance = max(0, random.normalvariate(mu = (distance_flag + 0.5) * one_third_distance, sigma = one_third_distance / 6))
        distance = min_distance + relative_distance
        distance = round(distance, 2)
        if distance > 8:
            rand_temp = random.random()
            if rand_temp < 0.1:
                distance = round(random.uniform(5, 5.5), 2)
            elif rand_temp >= 0.1 and rand_temp < 0.2:
                distance = round(random.uniform(5.5, 6), 2)
            elif rand_temp >= 0.2 and rand_temp < 0.35:
                distance = round(random.uniform(6, 6.5), 2)
            elif rand_temp >= 0.35 and rand_temp < 0.55:
                distance = round(random.uniform(6.5, 7), 2)
            elif rand_temp >= 0.55 and rand_temp < 0.75:
                distance = round(random.uniform(7, 7.5), 2)
            elif rand_temp >= 0.75:
                distance = round(random.uniform(7.5, 8), 2)
        source_location = np.array([distance * np.sin(rad) + 0.5,
                                    room_mid - distance * np.cos(rad),
                                    random.uniform(1.4, 1.8)])
    elif angle == 90:
        max_distance = room_length - 1
        one_third_distance = (max_distance - min_distance) / 3
        # relative_distance = random.uniform(distance_flag * one_third_distance, (distance_flag + 1) * one_third_distance)
        relative_distance = max(0, random.normalvariate(mu = (distance_flag + 0.5) * one_third_distance, sigma = one_third_distance / 6))
        distance = min_distance + relative_distance
        distance = round(distance, 2)
        if distance > 8:
            rand_temp = random.random()
            if rand_temp < 0.1:
                distance = round(random.uniform(5, 5.5), 2)
            elif rand_temp >= 0.1 and rand_temp < 0.2:
                distance = round(random.uniform(5.5, 6), 2)
            elif rand_temp >= 0.2 and rand_temp < 0.35:
                distance = round(random.uniform(6, 6.5), 2)
            elif rand_temp >= 0.35 and rand_temp < 0.55:
                distance = round(random.uniform(6.5, 7), 2)
            elif rand_temp >= 0.55 and rand_temp < 0.75:
                distance = round(random.uniform(7, 7.5), 2)
            elif rand_temp >= 0.75:
                distance = round(random.uniform(7.5, 8), 2)
        source_location = np.array([distance + 0.5,
                                    room_mid,
                                    random.uniform(1.4, 1.8)])
    else:
        rad = (180 - angle) / 180 * np.pi
        max_distance = min(room_mid / (np.cos(rad) + EPS), (room_length - 0.5) / (np.sin(rad) + EPS)) - 0.5
        one_third_distance = (max_distance - min_distance) / 3
        # relative_distance = random.uniform(distance_flag * one_third_distance, (distance_flag + 1) * one_third_distance)
        relative_distance = max(0, random.normalvariate(mu = (distance_flag + 0.5) * one_third_distance, sigma = one_third_distance / 6))
        distance = min_distance + relative_distance
        distance = round(distance, 2)
        if distance > 8:
            rand_temp = random.random()
            if rand_temp < 0.1:
                distance = round(random.uniform(5, 5.5), 2)
            elif rand_temp >= 0.1 and rand_temp < 0.2:
                distance = round(random.uniform(5.5, 6), 2)
            elif rand_temp >= 0.2 and rand_temp < 0.35:
                distance = round(random.uniform(6, 6.5), 2)
            elif rand_temp >= 0.35 and rand_temp < 0.55:
                distance = round(random.uniform(6.5, 7), 2)
            elif rand_temp >= 0.55 and rand_temp < 0.75:
                distance = round(random.uniform(7, 7.5), 2)
            elif rand_temp >= 0.75:
                distance = round(random.uniform(7.5, 8), 2)
        source_location = np.array([distance * np.sin(rad) + 0.5,
                                    room_mid + distance * np.cos(rad),
                                    random.uniform(1.4, 1.8)])

    # 生成多通道数据
    c = 343
    dist = np.linalg.norm(source_location - mic_locations[:, 0])

    delay = dist / c

    # 将声源放置在房间中
    room.add_source(source_location, signal=audio_data, delay=delay)

    # 将麦克风阵列放置在房间中
    room.add_microphone_array(mic_locations)

    # Run the simulation
    room.simulate()

    # 得到仿真的语音信号
    orig_max_value = np.max(np.abs(audio_data))
    multichannel_audio_data = room.mic_array.signals[:, 0:len(audio_data)]

    multichannel_audio_data = multichannel_audio_data / np.max(np.abs(multichannel_audio_data)) * orig_max_value
    multichannel_audio_data = multichannel_audio_data.astype(np.int16)

    # print(dist, rt60_tgt, max_order, orig_max_value, np.max(np.abs(multichannel_audio_data)))

    # 生成对应的文件夹
    if audio_type == 0:
        new_folder = os.path.join(folder, 'stationary_noise')
    elif audio_type == 1:
        new_folder = os.path.join(folder, 'speech')
    else:
        new_folder = os.path.join(folder, 'nonstationary_noise')

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    ### TODO
    wav_id = wave_file_name.split(".")[0]

    return multichannel_audio_data, wav_id, distance

def gen_mulchannel_data(wave_file_path, room_para, folder, audio_type=0, fs=16000):
    '''
    利用 Pyroomacoustics 生成多通道音频文件
    :param wave_file_path: 单通道音频文件路径
    :param room_para: 房间参数
    :param folder: 保存生成多通道声音文件的文件夹路径
    :param audio_type: 0：平稳噪声，1：人声，2：非平稳噪声
    :param fs: 音频采样率
    :return: 多通道音频data
    '''
    # 根据房间信息生成麦克风阵列位置
    room_dim = room_para['room_dim']
    rt60_tgt = room_para['rt60']
    room_length = room_dim[0]
    room_width = room_dim[1]
    room_height = room_dim[2]
    mic_locations = np.c_[
        [0.5, room_width / 2 - 0.07, 1],
        [0.5, room_width / 2 - 0.035, 1],
        [0.5, room_width / 2, 1],
        [0.5, room_width / 2 + 0.035, 1],
        [0.5, room_width / 2 + 0.07, 1],
        [0.5, room_width / 2 + 0.105, 1],
    ]

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
    # 根据参数创建房间
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order, air_absorption=True, humidity=50)
    audio_data, sr = sf.read(wave_file_path, dtype=np.int16)
    print("audio_data.shape = {}, sr = {}".format(audio_data.shape, sr))
    if sr != fs:
        raise ValueError("input wav file samplerate is not {}".format(fs))

    # 根据 wave 文件名获取说话人ID
    wave_file_name = wave_file_path.split('/')[-1]

    # 随机生成一个声源角度及距离
    if audio_type == 0:
        source_location = np.array([random.uniform(0, room_length),
                                    random.uniform(0, room_width),
                                    random.uniform(0, room_height)])
    elif audio_type == 1:
        source_location = np.array([random.uniform(mic_locations[0][0], room_length),
                                    random.uniform(0.5, room_width - 0.5),
                                    random.uniform(1.4, 1.8)])
    else:
        source_location = np.array([random.uniform(0.5, room_length),
                                    random.uniform(0, room_width),
                                    random.uniform(0, room_height)])
    if source_location[1] < room_width / 2:
        target_angle = np.arctan((source_location[0] - 0.5) / (room_width / 2 - source_location[1]))
    else:
        target_angle = np.pi / 2 + np.arctan((source_location[1] - room_width / 2) / (source_location[0] - 0.5))


    # 将弧度转为度
    angle_degree = int(target_angle * 180 / np.pi)

    # 生成多通道数据
    c = 345
    dist = np.linalg.norm(source_location - mic_locations[:, 0])


    if audio_type == 0:
        delay = 0
    else:
        delay = dist / c

    # 将声源放置在房间中
    room.add_source(source_location, signal=audio_data, delay=delay)

    # 将麦克风阵列放置在房间中
    room.add_microphone_array(mic_locations)

    # Run the simulation
    room.simulate()

    # 得到仿真的语音信号
    orig_max_value = np.max(np.abs(audio_data))
    multichannel_audio_data = room.mic_array.signals[:, 0:len(audio_data)]
    # multichannel_audio_data = multichannel_audio_data.astype(np.int16)

    multichannel_audio_data = multichannel_audio_data / np.max(np.abs(multichannel_audio_data)) * orig_max_value
    multichannel_audio_data = multichannel_audio_data.astype(np.int16)

    # print(dist, rt60_tgt, max_order, orig_max_value, np.max(np.abs(multichannel_audio_data)))

    # 生成对应的文件夹
    if audio_type == 0:
        new_folder = os.path.join(folder, 'stationary_noise')
    elif audio_type == 1:
        new_folder = os.path.join(folder, 'speech')
    else:
        new_folder = os.path.join(folder, 'nonstationary_noise')

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    # 生成对应文件名
    if audio_type == 0:
        wav_id = wave_file_name.replace('.wav', '')
        return multichannel_audio_data, wav_id, angle_degree
    elif audio_type == 1:
        # wav_id = re.findall('\d+', wave_file_name.split('/')[-1])[0]
        wav_id = wave_file_name.replace('.wav', '')
        return multichannel_audio_data, wav_id, angle_degree
    else:
        wav_id = wave_file_name.replace('.wav', '')
        return multichannel_audio_data, wav_id, angle_degree
