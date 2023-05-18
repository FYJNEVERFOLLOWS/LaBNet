import os
import copy
import json
import librosa
import numpy as np
import multiprocessing as mp
import soundfile as sf
import torch

import torch.utils.data as tud
from torch.utils.data import Dataset

def audioread(path, duration, fs = 16000):
    wave_data, sr = sf.read(path)
    if sr != fs:
        wave_data = librosa.resample(wave_data,sr,fs)
    return wave_data[:fs*duration]


def activelev(data):
    eps = np.finfo(np.float32).eps
    max_val = (1. + eps) / (np.std(data)+eps)
    data = data * max_val
    return data

def gaussian_func(gt_idx, output_dimention, sigma):
    indices = np.arange(output_dimention)
    out = np.array(np.exp(-1 * np.square(indices - gt_idx) / sigma ** 2))
    return out

def encode_AS(input_list):
    # Azimuth Spectrum
    AS = []
    for input_idx in input_list:   
        AS.append(gaussian_func(input_idx, 210, 8)) 
    AS = np.asarray(AS) # [1, 210]
    return AS

def encode_DS(input_list):
    # Distance Spectrum
    DS = []
    for input_idx in input_list:   
        DS.append(gaussian_func(input_idx, 200, 20))
    DS = np.asarray(DS)
    return DS

def gaussian_func_2d(x_len=10, y_len=10, granularity=0.5, x_mu=5, y_mu=5, x_sigma_pow2=0.1, y_sigma_pow2=0.1):
    '''
    Input: x_mu, y_mu
    Output: 2D array (each element indicates the probability of active sound source at each coordinate)
    '''
    X, Y = np.meshgrid(np.arange(0, x_len, granularity), np.arange(0, y_len, granularity))
    print(f'X.shape {X.shape}')
    # TODO
    Z = np.array(np.exp(-0.5 * np.square(X - x_mu) / x_sigma_pow2 - 0.5 * np.square(Y - y_mu) / y_sigma_pow2))
    print(f'Z.shape {X.shape}')

    return Z.flatten() # [400]

def encode_LS(x, y):
    # location heatmap
    DS = []
    DS.append(gaussian_func_2d(x_mu=x, y_mu=y))
    DS = np.asarray(DS) # [1, 400]
    return DS

def parse_scp(scp, path_list):
    with open(scp, encoding='utf-8') as fid:
        for line in fid:
            path_list.append(line.strip())

class TFDataset(Dataset):
    def __init__(self, wav_scp, data_mix_info, n_mics = 6, duration = 4, sample_rate= 16000,
    perturb_prob = 0.0, negatives=0.2, hold_coff=0.003, n_avb_mics = 2):
        mgr = mp.Manager()
        self.file_list = mgr.list()
        self.noise_list = mgr.list()
        self.index = mgr.list()
        
        self.data_mix_info = data_mix_info
        self.duration = duration

        self.sr = sample_rate
        self.n_mics = n_mics
        self.n_avb_mics = n_avb_mics
        self.perturb_prob = perturb_prob
        self.negatives = negatives
        self.hlod_coff = hold_coff

        self.angle_dimension = 210
        self.time_bins = 249
        self.speaker_num = 2

        pc_list = []
        p = mp.Process(target = parse_scp, args=(wav_scp,self.file_list))
        p.start()
        pc_list.append(p)

        for p in pc_list:
            p.join()

        self.index = [idx for idx in range(len(self.file_list))]

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        item = self.index[idx]
        file_index = item
        file_path = self.file_list[file_index]

        with open(file_path) as json_file:
            metadata = json.load(json_file)

        all_sources, doa_as_array, target_1, target_2, wave_paths, doa_idx_array, xy_coordinates = self.get_mixture_and_gt(metadata)

        all_sources = torch.stack(all_sources,dim=0)
        mixed_data = torch.sum(all_sources,dim=0)
        channel_num, _ = mixed_data.size()
        target_1 = torch.stack(target_1,dim=0)
        target_1 = torch.sum(target_1,dim=0)
        target_2 = torch.stack(target_2,dim=0)
        target_2 = torch.sum(target_2,dim=0)
        
        scale = 0.5
        for channel_idx in range(channel_num):
            mix_single_channel_wav = mixed_data[channel_idx,:]
            max_amp = torch.max(torch.abs(mix_single_channel_wav))
            if max_amp == 0:
                max_amp =1
            mix_scale = 1/max_amp*scale
            mixed_data[channel_idx,:] = mixed_data[channel_idx,:] * mix_scale
            target_1[channel_idx, :] = target_1[channel_idx, :] * mix_scale
            target_2[channel_idx, :] = target_2[channel_idx, :] * mix_scale
        
        # without noise data
        target_1 = target_1[0, :]
        target_2 = target_2[0, :]
        
        return {
            "mixed_data": mixed_data, 
            "target_1": target_1, 
            "target_2": target_2,
            "doa_as_array": doa_as_array, # [B, T, S, n_mics, 210]
            "doa_idx_array": doa_idx_array, # [B, T, S, n_mics]
            "wave_paths": wave_paths,
            "xy": xy_coordinates # [B, T, S, 2]
        }
        
        # # with noise data (need test)
        # target_1 = target_1[0, :]
        # target_2 = target_1[0, :]

        # mixed_data = np.array(mixed_data)
        # target_1 = np.array(target_1)
        # target_2 = np.array(target_2)
        # interference = np.zeros(len(target_1))

        # stationary_noise_wav_path_part = metadata["stationary_noise"]['wave_path']

        # SNR = metadata["stationary_noise"]['SNR']

        # for channel_idx in range(channel_num):
        #     stationary_noise_wav_path = stationary_noise_wav_path_part + '_' + str(channel_idx) + '.wav'
              
        #     noise_change_weight = np.max(mixed_data)*(10 ** (-SNR / 20))
        #     noise_wav = activelev(audioread(stationary_noise_wav_path, self.duration)) * noise_change_weight
        #     mixed_data[channel_idx, :] = mixed_data[channel_idx, :] + noise_wav
            
        #     if channel_idx == 0:
        #         interference = interference + noise_wav

        # mixed_data = torch.from_numpy(mixed_data)
        # target_1 = torch.from_numpy(target_1)
        # target_2 = torch.from_numpy(target_2)
        # interference = torch.from_numpy(interference)
        
        # return mixed_data, doa_as_array, doa_ont_hot_array, target_1, target_2, interference
    
    def get_mixture_and_gt(self, metadata):
        # dataset_prefix = "/local02/fuyanjie"
        # dataset_prefix = "/sata/fuyanjie"
        all_sources = []
        source_index = 0
        target_data_1 = []
        target_data_2 = []
        wave_paths= {}

        as_dict = dict([])
        doa_idx_array = np.zeros([self.time_bins, self.speaker_num, self.n_avb_mics], dtype=np.int16)
        doa_as_array = np.zeros([self.time_bins, self.speaker_num, self.n_avb_mics, self.angle_dimension])
        angle_list = np.zeros([self.speaker_num], dtype=np.int16)
        xy_coordinates = np.zeros([self.time_bins, self.speaker_num, 2])
        
        for key in metadata.keys():
            if "source" in key:
                channel_index_list = np.arange(self.n_mics)
                flag = metadata[key]['wave_path']
                # flag.replace(dataset_prefix, "/CDShare3")
                gt_audio_files = [flag + '_'+ str(channel_index) + '.wav' for channel_index in channel_index_list]
                gt_waveforms = []
                for index, gt_audio_file in enumerate(gt_audio_files):
                    gt_waveform = audioread(gt_audio_file, self.duration)
                    single_channel_wav = activelev(gt_waveform)
                    gt_waveforms.append(torch.from_numpy(single_channel_wav))
                
                shifted_gt = np.stack(gt_waveforms)
                perturbed_source = shifted_gt

                perturbed_source = torch.from_numpy(perturbed_source)
                perturbed_source = perturbed_source.to(torch.float32)

                if source_index !=0:
                    # ignore the SIR between diffrent speakers
                    #  SIR = metadata[key]['SIR']
                    SIR = 0
                    change_weight = 10 ** (SIR/20)
                    perturbed_source = perturbed_source * change_weight
                
                all_sources.append(perturbed_source)

                source_azimuth = int(metadata[key]['azimuth'])
                source2mic_dist = int(metadata[key]['s2m_dist'])

                source_azimuth1 = int(round(metadata[key]['azimuth1']))
                source_azimuth3 = int(round(metadata[key]['azimuth3']))
                source_azimuth4 = int(round(metadata[key]['azimuth4']))
                source_azimuth6 = int(round(metadata[key]['azimuth6']))

                if source_index == 0:
                    target_data_1.append(perturbed_source)
                    wave_paths["spk1"] = flag
                    angle_list[source_index] = source_azimuth
                elif source_index == 1:
                    target_data_2.append(perturbed_source)
                    wave_paths["spk2"] = flag
                    angle_list[source_index] = source_azimuth

                ASs = np.zeros([self.n_avb_mics, self.angle_dimension]) # [4, 210]
                if self.n_avb_mics == 2:
                    ASs[0, :] = encode_AS([source_azimuth1 + 15])
                    ASs[1, :] = encode_AS([source_azimuth6 + 15])
                elif self.n_avb_mics == 4:
                    ASs[0, :] = encode_AS([source_azimuth1 + 15])
                    ASs[1, :] = encode_AS([source_azimuth3 + 15])
                    ASs[2, :] = encode_AS([source_azimuth4 + 15])
                    ASs[3, :] = encode_AS([source_azimuth6 + 15])
                elif self.n_avb_mics == 1:
                    ASs[0, :] = encode_AS([source_azimuth + 15]) 

                as_dict[source_index] = ASs
                vad_label = metadata[key]['vad_label']
                for vad_index in range(len(vad_label)-1):
                    if vad_label[vad_index] == 1:
                        source_azimuth_rad = source_azimuth / 180.0 * np.pi
                        xy_coordinates[vad_index, source_index, 0] = source2mic_dist * np.cos(source_azimuth_rad)
                        xy_coordinates[vad_index, source_index, 1] = source2mic_dist * np.sin(source_azimuth_rad)
                        if self.n_avb_mics == 2:
                            doa_idx_array[vad_index, source_index, 0] = source_azimuth1 + 15
                            doa_idx_array[vad_index, source_index, 1] = source_azimuth6 + 15
                        elif self.n_avb_mics == 4:
                            doa_idx_array[vad_index, source_index, 0] = source_azimuth1 + 15
                            doa_idx_array[vad_index, source_index, 1] = source_azimuth3 + 15
                            doa_idx_array[vad_index, source_index, 2] = source_azimuth4 + 15
                            doa_idx_array[vad_index, source_index, 3] = source_azimuth6 + 15
                        elif self.n_avb_mics == 1:
                            doa_idx_array[vad_index, source_index, 0] = source_azimuth + 15
                    else:
                        doa_idx_array[vad_index, source_index, :] = -1
                source_index = source_index + 1
        

        # Sort by Azimuth
        for i in range(0, self.speaker_num):
            for j in range(i+1, self.speaker_num):
                if angle_list[i] > angle_list[j]:
                    temp_doa_arr_1 = copy.deepcopy(doa_idx_array[:, i, :])
                    temp_doa_arr_2 = copy.deepcopy(doa_idx_array[:, j, :])
                    doa_idx_array[:, i, :], doa_idx_array[:, j, :] = temp_doa_arr_2, temp_doa_arr_1
                    temp_target_1 = copy.deepcopy(target_data_1)
                    temp_target_2 = copy.deepcopy(target_data_2)
                    target_data_1, target_data_2 = temp_target_2, temp_target_1
                    temp1 = copy.deepcopy(angle_list[i])
                    temp2 = copy.deepcopy(angle_list[j])
                    angle_list[i], angle_list[j] = temp2, temp1
                    temp_as_1 = copy.deepcopy(as_dict[i])
                    temp_as_2 = copy.deepcopy(as_dict[j])
                    as_dict[0], as_dict[1] = temp_as_2, temp_as_1
                    temp_xy_1 = copy.deepcopy(xy_coordinates[:, i, :])
                    temp_xy_2 = copy.deepcopy(xy_coordinates[:, j, :])
                    xy_coordinates[:, i, :], xy_coordinates[:, j, :] = temp_xy_2, temp_xy_1

        # Assign after sorting            
        for source_idx in range(0, source_index):
            for time_idx in range(0, self.time_bins):
                if (doa_idx_array[time_idx, source_idx, :] == -1).any():
                    doa_as_array[time_idx, source_idx, :, :] = np.zeros([self.n_avb_mics, self.angle_dimension])
                else:
                    azi_s = as_dict[source_idx]
                    doa_as_array[time_idx, source_idx, :, :] = azi_s

        return all_sources, doa_as_array, target_data_1, target_data_2, wave_paths, doa_idx_array, xy_coordinates

class Sampler(tud.sampler.Sampler):
    def __init__(self, data_source, batch_size):
        it_end = len(data_source) - batch_size + 1
        self.batches = [range(i,i+batch_size) for i in range(0, it_end, batch_size)]
        self.data_source = data_source
    
    def __iter__(self):
        return (i for b in self.batches for i in b)
    
    def __len__(self):
        return len(self.data_source)

def static_loader(clean_scp, batch_size = 4, shuffle = True, num_workers = 8, duration = 4, sample_rate = 16000, data_mix_info = None, n_avb_mics = 2):
    dataset = TFDataset(
        wav_scp= clean_scp,
        data_mix_info = data_mix_info,
        duration = duration,
        sample_rate = sample_rate,
        n_avb_mics = n_avb_mics
    )

    loader = tud.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = False
    )
    return loader


if __name__ == "__main__":
    lst_path = '/Work21/2021/fuyanjie/pycode/LaBNet/data/exp_list/test-clean_0101.lst'
    data_loader = static_loader(lst_path, shuffle=True, batch_size=4)
    print(f'len(data_loader) {len(data_loader)}') # len(data_loader) is samples / batch_size
    one_batch_data = next(iter(data_loader)) 
    print('AAA ', one_batch_data["wave_paths"])
    print('B ', one_batch_data["doa_as_array"].dtype)
    print('C ', one_batch_data["doa_idx_array"].dtype)
    # AAA  torch.Size([224, 12, 7, 257]) torch.Size([224])
    # one_batch_data = next(iter(data_loader))
    # print('BBB ', one_batch_data["clean_ris"].shape, one_batch_data["target_doa"].shape)
    os.makedirs('../debug_plot', exist_ok=True)
    import matplotlib.pyplot as plt
    for idx, data in enumerate(data_loader):
        fig, ax = plt.subplots() # 创建图实例
        ax.plot(np.linspace(0, 210, 210), data['doa_as_array'][3, 50, 0, 0], color='r')
        # ax.plot(np.linspace(0, 210, 210), data['doa_idx_array'][2, 50, 0, 0], color='b')
        print(f' AAA DOA {data["doa_as_array"].shape}')
        print(f' A doa_as_array {data["doa_as_array"][3, 50, 0, 0].shape}')
        print(f' B DOA {data["doa_idx_array"][3, 50, 0]}')
        plt.savefig(f'../debug_plot/test_as_spk1_{idx}.png')
        plt.cla()
        ax.plot(np.linspace(0, 210, 210), data['doa_as_array'][3, 50, 1, 0], color='r')
        # ax.plot(np.linspace(0, 210, 210), data['doa_one_hot_array'][2, 50, 1, 0], color='b')
        print(f' C doa_as_array {data["doa_as_array"][3, 50, 1, 0].shape}')
        print(f' D doa_idx_array {data["doa_idx_array"][3, 50, 1]}')
        plt.savefig(f'../debug_plot/test_as_spk2_{idx}.png')
        # plt.cla()
        # ax.plot(np.linspace(0, 400, 400), data['dist_ds_array'][2, 50, 1], color='r')
        # ax.plot(np.linspace(0, 400, 400), data['dist_one_hot_array'][2, 50, 1], color='b')
        # print(f' C Dist {data["dist_idx_array"][2, 50, 1]}')
        # print(f' D Dist {np.where(data["dist_one_hot_array"][2, 50, 1] == 1)[0]}')
        # plt.savefig(f'./test_ds_spk2_{idx}.png')
        # print(data['wave_paths'])
        # print(data['mixed_data'].shape)
        # print(data['target_1'].shape)
        # print(data['target_2'].shape)
        # print(data['doa_as_array'].shape)
        # print(data['doa_one_hot_array'].shape)

        print('------------')
        if idx >= 3:
            break