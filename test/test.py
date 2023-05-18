import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import heapq
import json
import copy
sys.path.append("/Work21/2021/fuyanjie/pycode/LaBNetPro")

from model.Tree import Tree, pit_sisdr_numpy, wsdr_loss
from dataloader.test_dataloader import static_loader
from utils.utils import doa_err_2_source, cal_si_snr_np, cal_pesq, write_wav, split_spec_into_subbands_timedomain


sr = 16000
doa_resolution = 5
spkid2gender_path = "/Work21/2021/fuyanjie/pycode/LaBNet/data/libri_spkid2gender.json"

def load_obj(obj, device):
    """
    Offload tensor object in obj to cuda device
    """

    def cuda(obj):
        return obj.to(device, dtype=torch.float32) if isinstance(obj, torch.Tensor) else obj

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)

def test(model, args, epoch, device, alpha=1, beta=1):
    dataloader = static_loader(
        clean_scp=args.tt_clean,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_threads,
        sample_rate=args.sample_rate,
        data_mix_info=None,
        n_avb_mics=args.n_avb_mics
    )

    num_batch = len(dataloader)
    print("Len dataloader ", num_batch)
    stime = time.time()

    mse_loss = nn.MSELoss()

    sisnr_ori_total_1 = 0
    sisnr_ori_total_2 = 0
    sisnr_imp_total_1 = 0
    sisnr_imp_total_2 = 0
    sisnr_sep_total_1 = 0
    sisnr_sep_total_2 = 0
    sisnr_sep_total = 0

    sisnr_sep_lt4k_total = 0
    sisnr_sep_gt4k_total = 0
    pesq_sep_lt4k_total = 0
    pesq_sep_gt4k_total = 0
    sisnr_ori_lt4k_total = 0
    sisnr_ori_gt4k_total = 0
    pesq_ori_lt4k_total = 0
    pesq_ori_gt4k_total = 0

    sisnr_ori_total_angle_1 = np.zeros([12])
    sisnr_ori_total_angle_2 = np.zeros([12])
    sisnr_imp_total_angle_1 = np.zeros([12])
    sisnr_imp_total_angle_2 = np.zeros([12])
    sisnr_sep_total_angle_1 = np.zeros([12])
    sisnr_sep_total_angle_2 = np.zeros([12])
    sisnr_azidiff = np.zeros([4])
    sisnr_num_angle_1 = np.zeros([12])
    sisnr_num_angle_2 = np.zeros([12])
    pesq_ori_total_1 = 0
    pesq_ori_total_2 = 0
    pesq_imp_total_1 = 0
    pesq_imp_total_2 = 0
    pesq_sep_total_1 = 0
    pesq_sep_total_2 = 0
    pesq_ori_total_angle_1 = np.zeros([12])
    pesq_ori_total_angle_2 = np.zeros([12])
    pesq_imp_total_angle_1 = np.zeros([12])
    pesq_imp_total_angle_2 = np.zeros([12])
    pesq_sep_total_angle_1 = np.zeros([12])
    pesq_sep_total_angle_2 = np.zeros([12])
    pesq_azidiff = np.zeros([4])


    sisnr_sep_total_angle = np.zeros([12])
    sisnr_num_angle = np.zeros([12])
    sisnr_sep_total_sex = np.zeros([2])
    pesq_sep_total_sex = np.zeros([2])
    num_sex = np.zeros([2])
    sisnr_ori_total_sex = np.zeros([2])
    pesq_ori_total_sex = np.zeros([2])

    sumerr_per_angle = np.zeros([4])
    num_per_angle = np.zeros([4])
    doa_num_per_angle = np.zeros([4])
    num_acc_per_angle = np.zeros([4])
    num_pred_per_angle = np.zeros([4])

    num_acc = 0
    num_target = 0
    num_pred = 0
    sum_err = 0.0
    azi_num_acc = 0
    azi_num_pred = 0
    azi_sum_err = 0.0
    loss_total = 0.0
    loss_as_1_total = 0.0
    loss_as_2_total = 0.0
    loss_sig_1_total = 0.0
    loss_sig_2_total = 0.0
    azi_MAE_total = 0
    azi_MAE_1_total = 0
    azi_MAE_2_total = 0
    kvs = []
    sisdr_dist_ad = []
    spkid2gender = json.loads(open(spkid2gender_path, 'r').read())
    exp_dir = args.ckpt_path.split('/')[-2]
    with torch.no_grad():
        model.eval()
        for idx, egs in enumerate(dataloader):
            # load to gpu
            egs = load_obj(egs, device)
            inputs = egs["mixed_data"] # [B, C, T]
            gt_AS_arr = egs["doa_as_array"] # [B, T, S, n_mics, 210]
            gt_azi_arr = egs["doa_idx_array"] # [B, T, S, n_mics]
            target_1 = egs["target_1"] # [B, T]
            target_2 = egs["target_2"] # [B, T]
            azi_diff = egs["angular_distance"] # [B]
            social_distance = egs["social_distance"] # [B]
            wave_paths = egs["wave_paths"] # [B]

            es_AS_1, es_AS_2, es_sig_1, es_sig_2 = model(inputs)
            es_AS_1 = es_AS_1[:, 0:249, :, :] # [B, T, n_avb_mics, F]
            es_AS_2 = es_AS_2[:, 0:249, :, :] # [B, T, n_avb_mics, F]
            
            loss_as_1 = mse_loss(es_AS_1, gt_AS_arr[:, :, 0, :, :])
            loss_as_2 = mse_loss(es_AS_2, gt_AS_arr[:, :, 1, :, :])
            loss_sig_1 = wsdr_loss(es_sig_1, target_1, target_2)
            loss_sig_2 = wsdr_loss(es_sig_2, target_2, target_1)

            mixture = inputs[:, 0, :]
            # [Batch, Time_sample] numpy
            mixture_bt = mixture.cpu().numpy() # [batch, time_sample]
            target_1_bt = target_1.cpu().numpy() # [batch, time_sample]
            target_2_bt = target_2.cpu().numpy() # [batch, time_sample]
            es_sig_1_bt = es_sig_1.cpu().numpy() # [batch, 1, time_sample]
            es_sig_2_bt = es_sig_2.cpu().numpy() # [batch, 1, time_sample]

            mixture_bt = np.expand_dims(mixture_bt, 1)  # batch, 1, time_sample
            target_1_bt = np.expand_dims(target_1_bt, 1)  # batch, 1, time_sample
            target_2_bt = np.expand_dims(target_2_bt, 1)  # batch, 1, time_sample

            es_utt_1_bt = np.zeros_like(es_sig_1_bt)
            es_utt_2_bt = np.zeros_like(es_sig_2_bt)
            es_as_1_bt = torch.zeros(es_AS_1.shape, device=es_AS_1.device) # [B, T, n_avb_mics, F]
            es_as_2_bt = torch.zeros(es_AS_2.shape, device=es_AS_2.device) # [B, T, n_avb_mics, F]
            
            # s_ denotes the an example result in one batch
            for batch_index in range(args.batch_size):
                angular_dist = azi_diff[batch_index].item()
                social_dist = social_distance[batch_index].item()
                angle_index = int(angular_dist//15)
                s_mixture = mixture_bt[batch_index, :, :] # [1, T]
                s_es_sig_1 = es_sig_1_bt[batch_index, :, :]
                s_es_sig_2 = es_sig_2_bt[batch_index, :, :]
                s_target_1 = target_1_bt[batch_index, :, :]
                s_target_2 = target_2_bt[batch_index, :, :]

                s_mixture = np.expand_dims(s_mixture, 0)  # 1, 1, time_sample
                s_target_1 = np.expand_dims(s_target_1, 0)  # 1, 1, time_sample
                s_target_2 = np.expand_dims(s_target_2, 0)  # 1, 1, time_sample
                s_es_sig_1 = np.expand_dims(s_es_sig_1, 0)  # 1, 1, time_sample
                s_es_sig_2 = np.expand_dims(s_es_sig_2, 0)  # 1, 1, time_sample

                s_sep_sisnr, perm_idx = pit_sisdr_numpy(s_es_sig_1, s_es_sig_2, s_target_1, s_target_2)

                if perm_idx.item() == 1:
                    temp1 = copy.deepcopy(s_es_sig_1)
                    temp2 = copy.deepcopy(s_es_sig_2)
                    s_es_utt_1 = temp2
                    s_es_utt_2 = temp1
                    temp1 = copy.deepcopy(es_sig_1_bt[batch_index, :, :])
                    temp2 = copy.deepcopy(es_sig_2_bt[batch_index, :, :])
                    es_utt_1_bt[batch_index, :, :] = temp2
                    es_utt_2_bt[batch_index, :, :] = temp1
                    temp1 = copy.deepcopy(es_AS_1[batch_index, ...])
                    temp2 = copy.deepcopy(es_AS_2[batch_index, ...])
                    es_as_1_bt[batch_index, ...] = temp2
                    es_as_2_bt[batch_index, ...] = temp1
                    temp1 = copy.deepcopy(wave_paths['spk1'][batch_index])
                    temp2 = copy.deepcopy(wave_paths['spk2'][batch_index])
                    wave_path_spk1 = temp1
                    wave_path_spk2 = temp2
                    print(f'After swapping: {wave_path_spk1} {wave_path_spk2}', flush=True)
                else:
                    temp1 = copy.deepcopy(s_es_sig_1)
                    temp2 = copy.deepcopy(s_es_sig_2)
                    s_es_utt_1 = temp1
                    s_es_utt_2 = temp2
                    temp1 = copy.deepcopy(es_sig_1_bt[batch_index, :, :])
                    temp2 = copy.deepcopy(es_sig_2_bt[batch_index, :, :])
                    es_utt_1_bt[batch_index, :, :] = temp1
                    es_utt_2_bt[batch_index, :, :] = temp2
                    temp1 = copy.deepcopy(es_AS_1[batch_index, ...])
                    temp2 = copy.deepcopy(es_AS_2[batch_index, ...])
                    es_as_1_bt[batch_index, ...] = temp1
                    es_as_2_bt[batch_index, ...] = temp2
                    wave_path_spk1 = wave_paths['spk1'][batch_index]
                    wave_path_spk2 = wave_paths['spk2'][batch_index]
                sisnr_sep_total_angle[angle_index] += s_sep_sisnr
                sisnr_sep_total += s_sep_sisnr
                sisnr_num_angle[angle_index] += 1

                s_ori_sisnr_1 = cal_si_snr_np(s_target_1, s_mixture)
                s_ori_sisnr_2 = cal_si_snr_np(s_target_2, s_mixture)
                sisnr_ori_total_angle_1[angle_index] += s_ori_sisnr_1
                sisnr_ori_total_angle_2[angle_index] += s_ori_sisnr_2

                s_sep_sisnr_1 = cal_si_snr_np(s_target_1, s_es_utt_1)
                s_sep_sisnr_2 = cal_si_snr_np(s_target_2, s_es_utt_2)
                sisnr_sep_total_angle_1[angle_index] += s_sep_sisnr_1
                sisnr_sep_total_angle_2[angle_index] += s_sep_sisnr_2

                s_imp_sisnr_1 = s_sep_sisnr_1 - s_ori_sisnr_1
                s_imp_sisnr_2 = s_sep_sisnr_2 - s_ori_sisnr_2
                sisnr_imp_total_angle_1[angle_index] += s_imp_sisnr_1
                sisnr_imp_total_angle_2[angle_index] += s_imp_sisnr_2
                sisnr_num_angle_1[angle_index] += 1
                sisnr_num_angle_2[angle_index] += 1

                # PESQ
                s_mixture_bt = s_mixture[0, :, :]  # 1, time_sample
                s_target_1_bt = s_target_1[0, :, :]  # 1, time_sample
                s_target_2_bt = s_target_2[0, :, :]  # 1, time_sample
                s_es_sig_1_bt = s_es_utt_1[0, :, :]  # 1, time_sample
                s_es_sig_2_bt = s_es_utt_2[0, :, :]  # 1, time_sample
                
                # > 4k Hz & < 4k Hz
                # [1, T]
                s_mixture_lt4k, s_mixture_gt4k = split_spec_into_subbands_timedomain(s_mixture_bt, 4000, 512, 256)
                s_es_sig_1_lt4k, s_es_sig_1_gt4k = split_spec_into_subbands_timedomain(s_es_sig_1_bt, 4000, 512, 256)
                s_es_sig_2_lt4k, s_es_sig_2_gt4k = split_spec_into_subbands_timedomain(s_es_sig_2_bt, 4000, 512, 256)
                s_sep_pesq_1_lt4k = cal_pesq(s_target_1_bt, s_es_sig_1_lt4k)
                s_sep_pesq_1_gt4k = cal_pesq(s_target_1_bt, s_es_sig_1_gt4k)
                s_sep_pesq_2_lt4k = cal_pesq(s_target_2_bt, s_es_sig_2_lt4k)
                s_sep_pesq_2_gt4k = cal_pesq(s_target_2_bt, s_es_sig_2_gt4k)
                pesq_sep_lt4k_total += (s_sep_pesq_1_lt4k + s_sep_pesq_2_lt4k)
                pesq_sep_gt4k_total += (s_sep_pesq_1_gt4k + s_sep_pesq_2_gt4k)

                s_ori_pesq_1_lt4k = cal_pesq(s_target_1_bt, s_mixture_lt4k)
                s_ori_pesq_1_gt4k = cal_pesq(s_target_1_bt, s_mixture_gt4k)
                s_ori_pesq_2_lt4k = cal_pesq(s_target_2_bt, s_mixture_lt4k)
                s_ori_pesq_2_gt4k = cal_pesq(s_target_2_bt, s_mixture_gt4k)
                pesq_ori_lt4k_total += (s_ori_pesq_1_lt4k + s_ori_pesq_2_lt4k)
                pesq_ori_gt4k_total += (s_ori_pesq_1_gt4k + s_ori_pesq_2_gt4k)


                s_ori_pesq_1 = cal_pesq(s_target_1_bt, s_mixture_bt)
                s_ori_pesq_2 = cal_pesq(s_target_2_bt, s_mixture_bt)
                pesq_ori_total_angle_1[angle_index] += s_ori_pesq_1
                pesq_ori_total_angle_2[angle_index] += s_ori_pesq_2

                s_sep_pesq_1 = cal_pesq(s_target_1_bt, s_es_sig_1_bt)
                s_sep_pesq_2 = cal_pesq(s_target_2_bt, s_es_sig_2_bt)
                pesq_sep_total_angle_1[angle_index] += s_sep_pesq_1
                pesq_sep_total_angle_2[angle_index] += s_sep_pesq_2

                s_imp_pesq_1 = s_sep_pesq_1 - s_ori_pesq_1
                s_imp_pesq_2 = s_sep_pesq_2 - s_ori_pesq_2
                pesq_imp_total_angle_1[angle_index] += s_imp_pesq_1
                pesq_imp_total_angle_2[angle_index] += s_imp_pesq_2

                s_mixture_lt4k = np.expand_dims(s_mixture_lt4k, 0)  # 1, 1, time_sample
                s_mixture_gt4k = np.expand_dims(s_mixture_gt4k, 0)  # 1, 1, time_sample
                s_es_sig_1_lt4k = np.expand_dims(s_es_sig_1_lt4k, 0)  # 1, 1, time_sample
                s_es_sig_1_gt4k = np.expand_dims(s_es_sig_1_gt4k, 0)  # 1, 1, time_sample
                s_es_sig_2_lt4k = np.expand_dims(s_es_sig_2_lt4k, 0)  # 1, 1, time_sample
                s_es_sig_2_gt4k = np.expand_dims(s_es_sig_2_gt4k, 0)  # 1, 1, time_sample
                s_sep_sisnr_1_lt4k = cal_si_snr_np(s_target_1, s_es_sig_1_lt4k)
                s_sep_sisnr_1_gt4k = cal_si_snr_np(s_target_1, s_es_sig_1_gt4k)
                s_sep_sisnr_2_lt4k = cal_si_snr_np(s_target_2, s_es_sig_2_lt4k)  
                s_sep_sisnr_2_gt4k = cal_si_snr_np(s_target_2, s_es_sig_2_gt4k)  

                sisnr_sep_lt4k_total += (s_sep_sisnr_1_lt4k + s_sep_sisnr_2_lt4k)
                sisnr_sep_gt4k_total += (s_sep_sisnr_1_gt4k + s_sep_sisnr_2_gt4k)
                s_ori_sisnr_1_lt4k = cal_si_snr_np(s_target_1, s_mixture_lt4k)
                s_ori_sisnr_1_gt4k = cal_si_snr_np(s_target_1, s_mixture_gt4k)
                s_ori_sisnr_2_lt4k = cal_si_snr_np(s_target_2, s_mixture_lt4k)
                s_ori_sisnr_2_gt4k = cal_si_snr_np(s_target_2, s_mixture_gt4k)
                sisnr_ori_lt4k_total += (s_ori_sisnr_1_lt4k + s_ori_sisnr_2_lt4k)
                sisnr_ori_gt4k_total += (s_ori_sisnr_1_gt4k + s_ori_sisnr_2_gt4k)

                if spkid2gender[wave_path_spk1.split('/')[-1].split('-')[0].split('_')[0]] == spkid2gender[wave_path_spk2.split('/')[-1].split('-')[0].split('_')[0]]:
                    same_sex = 1
                else:
                    same_sex = 0
                sisnr_sep_total_sex[same_sex] += s_sep_sisnr * 2
                sisnr_ori_total_sex[same_sex] += (s_ori_sisnr_1 + s_ori_sisnr_2)
                num_sex[same_sex] += 2
                pesq_sep_total_sex[same_sex] += (s_sep_pesq_1 + s_sep_pesq_2)
                pesq_ori_total_sex[same_sex] += (s_ori_pesq_1 + s_ori_pesq_2)
                kv_eg = dict()
                kv_stat = dict()
                kv_stat["sisdr"] = s_sep_sisnr
                kv_stat["sisdr1"] = s_sep_sisnr_1
                kv_stat["sisdr2"] = s_sep_sisnr_2
                kv_stat["ad"] = angular_dist
                kv_stat["dist"] = social_dist
                print(f'{idx}thBatch_{batch_index} {kv_stat}')
                kv_eg[f'{idx}thBatch_{batch_index}'] = kv_stat
                sisdr_dist_ad.append(kv_eg)
                if angular_dist < 15:
                    num_per_angle[0] += 1
                    sisnr_azidiff[0] += (s_sep_sisnr_1 + s_sep_sisnr_2) / 2
                    pesq_azidiff[0] += (s_sep_pesq_1 + s_sep_pesq_2) / 2
                elif angular_dist >= 15 and angular_dist < 45:
                    num_per_angle[1] += 1
                    sisnr_azidiff[1] += (s_sep_sisnr_1 + s_sep_sisnr_2) / 2
                    pesq_azidiff[1] += (s_sep_pesq_1 + s_sep_pesq_2) / 2
                elif angular_dist >= 45 and angular_dist < 90:
                    num_per_angle[2] += 1
                    sisnr_azidiff[2] += (s_sep_sisnr_1 + s_sep_sisnr_2) / 2
                    pesq_azidiff[2] += (s_sep_pesq_1 + s_sep_pesq_2) / 2
                else:
                    num_per_angle[3] += 1
                    sisnr_azidiff[3] += (s_sep_sisnr_1 + s_sep_sisnr_2) / 2
                    pesq_azidiff[3] += (s_sep_pesq_1 + s_sep_pesq_2) / 2


                if args.write_wav:
                    # write to .wav
                    separated_audio_subfolders = ["estimatedlt15", "estimated15-45", "estimated45-90", "estimatedgt90"]
                    same_sex_or_diff_sex_subfolders = ["estimateddiffsex", "estimatedsamesex"]
                    os.makedirs(os.path.join("exp", exp_dir, "estimatedlt15"), exist_ok=True)
                    os.makedirs(os.path.join("exp", exp_dir, "estimated15-45"), exist_ok=True)
                    os.makedirs(os.path.join("exp", exp_dir, "estimated45-90"), exist_ok=True)
                    os.makedirs(os.path.join("exp", exp_dir, "estimatedgt90"), exist_ok=True)
                    os.makedirs(os.path.join("exp", exp_dir, "estimatedsamesex"), exist_ok=True)
                    os.makedirs(os.path.join("exp", exp_dir, "estimateddiffsex"), exist_ok=True)

                    if angular_dist < 15:
                        group = 0
                    elif angular_dist >= 15 and angular_dist < 45:
                        group = 1
                    elif angular_dist >= 45 and angular_dist < 90:
                        group = 2
                    else:
                        group = 3
                    
                    s_mixture_lt4k = np.squeeze(s_mixture_lt4k)
                    s_mixture_gt4k = np.squeeze(s_mixture_gt4k)
                    s_es_sig_1_lt4k = np.squeeze(s_es_sig_1_lt4k)
                    s_es_sig_1_gt4k = np.squeeze(s_es_sig_1_gt4k)
                    s_es_sig_2_lt4k = np.squeeze(s_es_sig_2_lt4k)
                    s_es_sig_2_gt4k = np.squeeze(s_es_sig_2_gt4k)

                    s_es_sig_1_bt = np.squeeze(s_es_sig_1_bt)
                    s_mixture_bt = np.squeeze(s_mixture_bt)
                    norm = np.linalg.norm(s_mixture_bt, np.inf)
                    # norm
                    samps_1 = s_es_sig_1_bt / np.max(np.abs(s_es_sig_1_bt)) * norm
                    
                    spkid1 = wave_path_spk1.split('/')[-1].split('-')[0]
                    chapid1 = wave_path_spk1.split('/')[-1].split('-')[1]
                    uttid1 = wave_path_spk1.split('/')[-1].split('-')[2]
                    uttid1 = f'{spkid1}-{chapid1}-{uttid1}'
                    spkid2 = wave_path_spk2.split('/')[-1].split('-')[0]
                    chapid2 = wave_path_spk2.split('/')[-1].split('-')[1]
                    uttid2 = wave_path_spk2.split('/')[-1].split('-')[2]
                    uttid2 = f'{spkid2}-{chapid2}-{uttid2}'
                    write_wav(
                        os.path.join("exp", exp_dir, separated_audio_subfolders[group], "{}thBatch_{}+{}[{}°].wav".format(idx, batch_index, uttid1, int(angular_dist))), 
                        samps_1,
                        fs=args.sample_rate)
                    write_wav(
                        os.path.join("exp", exp_dir, same_sex_or_diff_sex_subfolders[same_sex], "{}thBatch_{}+{}[{}°].wav".format(idx, batch_index, uttid1, int(angular_dist))), 
                        samps_1,
                        fs=args.sample_rate)
                    kv1 = dict()
                    kv1[f'{idx}thBatch_{batch_index}_spk1'] = wave_path_spk1
                    kvs.append(kv1)

                    s_es_sig_2_bt = np.squeeze(s_es_sig_2_bt)
                    # norm
                    samps_2 = s_es_sig_2_bt / np.max(np.abs(s_es_sig_2_bt)) * norm
                    write_wav(
                        os.path.join("exp", exp_dir, separated_audio_subfolders[group], "{}thBatch_{}+{}[{}°].wav".format(idx, batch_index, uttid2, int(angular_dist))), 
                        samps_2,
                        fs=args.sample_rate)
                    write_wav(
                        os.path.join("exp", exp_dir, same_sex_or_diff_sex_subfolders[same_sex], "{}thBatch_{}+{}[{}°].wav".format(idx, batch_index, uttid2, int(angular_dist))), 
                        samps_2,
                        fs=args.sample_rate)
                    kv2 = dict()
                    kv2[f'{idx}thBatch_{batch_index}_spk2'] = wave_path_spk2
                    kvs.append(kv2)
                    
                    print(f"{idx}thBatch_{batch_index} {wave_path_spk1} \n {wave_path_spk2}", flush=True)
            
            # SI-SDR: [B, C=1, T]
            original_sisnr_1 = cal_si_snr_np(target_1_bt, mixture_bt)
            original_sisnr_2 = cal_si_snr_np(target_2_bt, mixture_bt)
            sisnr_ori_total_1 += original_sisnr_1
            sisnr_ori_total_2 += original_sisnr_2

            separated_sisnr_1 = cal_si_snr_np(target_1_bt, es_utt_1_bt)
            separated_sisnr_2 = cal_si_snr_np(target_2_bt, es_utt_2_bt)
            sisnr_sep_total_1 += separated_sisnr_1
            sisnr_sep_total_2 += separated_sisnr_2

            sisnr_imp_total_1 += (separated_sisnr_1-original_sisnr_1)
            sisnr_imp_total_2 += (separated_sisnr_2-original_sisnr_2)
            # PESQ: [B, T]
            mixture_bt = mixture_bt[:, 0, :]
            target_1_bt = target_1_bt[:, 0, :]
            target_2_bt = target_2_bt[:, 0, :]
            es_utt_1_bt = es_utt_1_bt[:, 0, :]
            es_utt_2_bt = es_utt_2_bt[:, 0, :]

            original_pesq_1 = cal_pesq(target_1_bt, mixture_bt)
            original_pesq_2 = cal_pesq(target_2_bt, mixture_bt)
            pesq_ori_total_1 += original_pesq_1
            pesq_ori_total_2 += original_pesq_2

            separated_pesq_1 = cal_pesq(target_1_bt, es_utt_1_bt)
            separated_pesq_2 = cal_pesq(target_2_bt, es_utt_2_bt)
            pesq_sep_total_1 += separated_pesq_1
            pesq_sep_total_2 += separated_pesq_2

            pesq_imp_total_1 += (separated_pesq_1 - original_pesq_1)
            pesq_imp_total_2 += (separated_pesq_2 - original_pesq_2)

            azi_mae_1, azi_mae_2, azi_num_acc_1, azi_num_acc_2, azi_num_pred_1, azi_num_pred_2 = doa_err_2_source(gt_azi_arr, es_as_1_bt, es_as_2_bt)
            azi_mae = (azi_mae_1 + azi_mae_2)/2
            azi_num_acc += azi_num_acc_1 + azi_num_acc_2
            azi_num_pred += azi_num_pred_1 + azi_num_pred_2
            azi_sum_err += azi_mae_1 * azi_num_pred_1 + azi_mae_2 * azi_num_pred_2

            gt_azi_arr_cpu = gt_azi_arr.data.cpu()
            es_as_1_cpu = es_as_1_bt.data.cpu()
            es_as_2_cpu = es_as_2_bt.data.cpu()
            gt_azi_arr_cpu = gt_azi_arr_cpu.numpy()
            es_as_1_cpu = es_as_1_cpu.numpy()
            es_as_2_cpu = es_as_2_cpu.numpy()
            for batch_idx in range(args.batch_size):
                for time_idx in range(249):
                    gt_angle_1 = gt_azi_arr_cpu[batch_idx, time_idx, 0, 0]
                    source_num_1 = 1 if (gt_azi_arr_cpu[batch_idx, time_idx, 0, :] != -1).any() else 0
                    gt_angle_2 = gt_azi_arr_cpu[batch_idx, time_idx, 1, 0]
                    source_num_2 = 1 if (gt_azi_arr_cpu[batch_idx, time_idx, 1, :] != -1).any() else 0                   
                    if source_num_1 != 0 and source_num_2 != 0:
                        num_pred += 2
                        es_angle_1 = heapq.nlargest(source_num_1,
                        range(len(es_as_1_cpu[batch_idx, time_idx, 0, :])),
                        es_as_1_cpu[batch_idx, time_idx, 0, :].take)
                        es_angle_2 = heapq.nlargest(source_num_2,
                        range(len(es_as_2_cpu[batch_idx, time_idx, 0, :])),
                        es_as_2_cpu[batch_idx, time_idx, 0, :].take)
                        if np.abs(es_angle_1[0] - gt_angle_1) <= doa_resolution:
                            num_acc += 1
                        if np.abs(es_angle_2[0] - gt_angle_2) <= doa_resolution:
                            num_acc += 1
                        sum_err += np.abs(es_angle_1[0] - gt_angle_1) + np.abs(es_angle_2[0] - gt_angle_2)

                        angular_dist = np.abs(gt_angle_1 - gt_angle_2)
                        if angular_dist < 15:
                            doa_num_per_angle[0] += 1
                            num_pred_per_angle[0] += 2
                            if np.abs(es_angle_1[0] - gt_angle_1) <= doa_resolution:
                                num_acc_per_angle[0] += 1
                            if np.abs(es_angle_2[0] - gt_angle_2) <= doa_resolution:
                                num_acc_per_angle[0] += 1
                            sumerr_per_angle[0] += np.abs(es_angle_1[0] - gt_angle_1) + np.abs(es_angle_2[0] - gt_angle_2)
                        elif angular_dist >= 15 and angular_dist < 45:
                            doa_num_per_angle[1] += 1
                            num_pred_per_angle[1] += 2
                            if np.abs(es_angle_1[0] - gt_angle_1) <= doa_resolution:
                                num_acc_per_angle[1] += 1
                            if np.abs(es_angle_2[0] - gt_angle_2) <= doa_resolution:
                                num_acc_per_angle[1] += 1
                            sumerr_per_angle[1] += np.abs(es_angle_1[0] - gt_angle_1) + np.abs(es_angle_2[0] - gt_angle_2)
                        elif angular_dist >= 45 and angular_dist < 90:
                            doa_num_per_angle[2] += 1
                            num_pred_per_angle[2] += 2
                            if np.abs(es_angle_1[0] - gt_angle_1) <= doa_resolution:
                                num_acc_per_angle[2] += 1
                            if np.abs(es_angle_2[0] - gt_angle_2) <= doa_resolution:
                                num_acc_per_angle[2] += 1
                            sumerr_per_angle[2] += np.abs(es_angle_1[0] - gt_angle_1) + np.abs(es_angle_2[0] - gt_angle_2)                            
                        else:
                            doa_num_per_angle[3] += 1
                            num_pred_per_angle[3] += 2
                            if np.abs(es_angle_1[0] - gt_angle_1) <= doa_resolution:
                                num_acc_per_angle[3] += 1
                            if np.abs(es_angle_2[0] - gt_angle_2) <= doa_resolution:
                                num_acc_per_angle[3] += 1
                            sumerr_per_angle[3] += np.abs(es_angle_1[0] - gt_angle_1) + np.abs(es_angle_2[0] - gt_angle_2)      

            loss_sig_1_total += loss_sig_1.data.cpu()
            loss_sig_2_total += loss_sig_2.data.cpu()
            loss_as_1_total += loss_as_1.data.cpu()
            loss_as_2_total += loss_as_2.data.cpu()
            azi_MAE_total += azi_mae
            azi_MAE_1_total += azi_mae_1
            azi_MAE_2_total += azi_mae_2

            del inputs, gt_azi_arr, gt_AS_arr, es_AS_1, es_AS_2, loss_as_1, loss_as_2

    json_path = "estimated2oraclepath.json"
    with open(json_path, 'w') as f:
        f.write(json.dumps(kvs))
    json_path = "sisnr_ad_dist.json"
    with open(json_path, 'w') as f:
        f.write(json.dumps(sisdr_dist_ad))
    mae_per_angle = sumerr_per_angle / num_pred_per_angle
    acc_per_angle = num_acc_per_angle / num_pred_per_angle
    print(f'TOTAL EXAMPLES: <15 {doa_num_per_angle[0]} 15-45 {doa_num_per_angle[1]} 45-90 {doa_num_per_angle[2]} >90 {doa_num_per_angle[3]}')
    print('<15 | ACC: {:2.4f} | MAE:{:2.4f}'.format(acc_per_angle[0], mae_per_angle[0]))
    print('15-45 | ACC: {:2.4f} | MAE:{:2.4f}'.format(acc_per_angle[1], mae_per_angle[1]))
    print('45-90 | ACC: {:2.4f} | MAE:{:2.4f}'.format(acc_per_angle[2], mae_per_angle[2]))
    print('>90 | ACC: {:2.4f} | MAE:{:2.4f}'.format(acc_per_angle[3], mae_per_angle[3]))
    if not num_pred == 0:
        print('DOA Overall ACC frame-level {:2.4f} '.format(azi_num_acc / azi_num_pred))
        print('DOA Overall MAE frame-level {:2.4f} '.format(azi_sum_err / azi_num_pred))
        print('AZI Overall ACC frame-level {:2.4f} (Compute frame by frame, 1 mic only)'.format(num_acc / num_pred))
        print('AZI Overall MAE frame-level {:2.4f} (Compute frame by frame, 1 mic only))'.format(sum_err / num_pred))
        
    etime = time.time()
    eplashed = (etime - stime) / num_batch

    loss_avg = loss_total / num_batch
    loss_as_1_avg = loss_as_1_total / num_batch
    loss_as_2_avg = loss_as_2_total / num_batch
    loss_sig_1_avg = loss_sig_1_total / num_batch
    loss_sig_2_avg = loss_sig_2_total / num_batch
    azi_MAE_avg = azi_MAE_total / num_batch
    azi_MAE_1_avg = azi_MAE_1_total / num_batch
    azi_MAE_2_avg = azi_MAE_2_total / num_batch

    # OVERALL
    print('---------------Overall Performance---------------')

    # Overall - original performance
    pesq_ori_avg_1 = pesq_ori_total_1 / num_batch
    pesq_ori_avg_2 = pesq_ori_total_2 / num_batch
    sisnr_ori_avg_1 = sisnr_ori_total_1 / num_batch
    sisnr_ori_avg_2 = sisnr_ori_total_2 / num_batch
    print('===============1.Original===============')
    print('PESQ   || source 1 : {:2.4f} | source 2 : {:2.4f} | AVG: {:2.4f}'.format(pesq_ori_avg_1, pesq_ori_avg_2, (pesq_ori_avg_1+pesq_ori_avg_2)/2))
    print('SI-SNR || source 1 : {:2.4f} | source 2 : {:2.4f} | AVG: {:2.4f}'.format(sisnr_ori_avg_1, sisnr_ori_avg_2, (sisnr_ori_avg_1+sisnr_ori_avg_2)/2))

    # Overall - separated performance
    pesq_sep_avg_1 = pesq_sep_total_1 / num_batch
    pesq_sep_avg_2 = pesq_sep_total_2 / num_batch
    sisnr_sep_avg_1 = sisnr_sep_total_1 / num_batch
    sisnr_sep_avg_2 = sisnr_sep_total_2 / num_batch
    sisnr_sep_avg = sisnr_sep_total / (num_batch * args.batch_size)

    print('===============2.Separated===============')
    print('PESQ   || source 1 : {:2.4f} | source 2 : {:2.4f} | AVG: {:2.4f}'.format(pesq_sep_avg_1, pesq_sep_avg_2, (pesq_sep_avg_1+pesq_sep_avg_2)/2))
    print('SI-SNR || source 1 : {:2.4f} | source 2 : {:2.4f} | AVG: {:2.4f}'.format(sisnr_sep_avg_1, sisnr_sep_avg_2, (sisnr_sep_avg_1+sisnr_sep_avg_2)/2))
    print('SI-SNR || AVG : {:2.4f}'.format(sisnr_sep_avg))

    # Overall - improved performance
    pesq_imp_avg_1 = pesq_imp_total_1 / num_batch
    pesq_imp_avg_2 = pesq_imp_total_2 / num_batch
    sisnr_imp_avg_1 = sisnr_imp_total_1 / num_batch
    sisnr_imp_avg_2 = sisnr_imp_total_2 / num_batch
    print('===============3.Improved===============')
    print('PESQ   || source 1 : {:2.4f} | source 2 : {:2.4f} | AVG: {:2.4f}'.format(pesq_imp_avg_1, pesq_imp_avg_2, (pesq_imp_avg_1+pesq_imp_avg_2)/2))
    print('SI-SNR || source 1 : {:2.4f} | source 2 : {:2.4f} | AVG: {:2.4f}'.format(sisnr_imp_avg_1, sisnr_imp_avg_2, (sisnr_imp_avg_1+sisnr_imp_avg_2)/2))
    
    sisnr_sep_lt4k_avg = sisnr_sep_lt4k_total / (num_batch * args.batch_size * 2)
    sisnr_sep_gt4k_avg = sisnr_sep_gt4k_total / (num_batch * args.batch_size * 2)
    pesq_sep_lt4k_avg = pesq_sep_lt4k_total / (num_batch * args.batch_size * 2)
    pesq_sep_gt4k_avg = pesq_sep_gt4k_total / (num_batch * args.batch_size * 2)
    sisnr_ori_lt4k_avg = sisnr_ori_lt4k_total / (num_batch * args.batch_size * 2)
    sisnr_ori_gt4k_avg = sisnr_ori_gt4k_total / (num_batch * args.batch_size * 2)
    pesq_ori_lt4k_avg = pesq_ori_lt4k_total / (num_batch * args.batch_size * 2)
    pesq_ori_gt4k_avg = pesq_ori_gt4k_total / (num_batch * args.batch_size * 2)
  
    print('===============4.Freq Subbands===============')
    print('SI-SNR Mixture || < 4kHz 1 : {:2.4f} | 4-16 kHz {:2.4f}'.format(sisnr_ori_lt4k_avg, sisnr_ori_gt4k_avg))
    print('SI-SNR Separated || < 4kHz 1 : {:2.4f} | 4-16 kHz {:2.4f}'.format(sisnr_sep_lt4k_avg, sisnr_sep_gt4k_avg))
    print('PESQ Mixture || < 4kHz 1 : {:2.4f} | 4-16 kHz {:2.4f}'.format(pesq_ori_lt4k_avg, pesq_ori_gt4k_avg))
    print('PESQ Separated || < 4kHz 1 : {:2.4f} | 4-16 kHz {:2.4f}'.format(pesq_sep_lt4k_avg, pesq_sep_gt4k_avg))


    print('---------------------------------------------')

    sisnr_sep_avg_sex = sisnr_sep_total_sex / num_sex
    pesq_sep_avg_sex = pesq_sep_total_sex / num_sex
    sisnr_ori_avg_sex = sisnr_ori_total_sex / num_sex
    pesq_ori_avg_sex = pesq_ori_total_sex / num_sex
    print('---------------Gender Performance---------------')
    for sex_index in range(len(num_sex)):
        print(f'Same Gender?: {sex_index} | Num: {num_sex[sex_index]}')
        print(f'SEP SI-SDR: {sisnr_sep_avg_sex[sex_index]} | PESQ: {pesq_sep_avg_sex[sex_index]}')
        print(f'ORI SI-SDR: {sisnr_ori_avg_sex[sex_index]} | PESQ: {pesq_ori_avg_sex[sex_index]}')


    pesq_ori_avg_angle_1 = pesq_ori_total_angle_1 / sisnr_num_angle_1
    pesq_ori_avg_angle_2 = pesq_ori_total_angle_2 / sisnr_num_angle_2
    pesq_imp_avg_angle_1 = pesq_imp_total_angle_1 / sisnr_num_angle_1
    pesq_imp_avg_angle_2 = pesq_imp_total_angle_2 / sisnr_num_angle_2
    pesq_sep_avg_angle_1 = pesq_sep_total_angle_1 / sisnr_num_angle_1
    pesq_sep_avg_angle_2 = pesq_sep_total_angle_2 / sisnr_num_angle_2
    sisnr_ori_avg_angle_1 = sisnr_ori_total_angle_1 / sisnr_num_angle_1
    sisnr_ori_avg_angle_2 = sisnr_ori_total_angle_2 / sisnr_num_angle_2
    sisnr_imp_avg_angle_1 = sisnr_imp_total_angle_1 / sisnr_num_angle_1
    sisnr_imp_avg_angle_2 = sisnr_imp_total_angle_2 / sisnr_num_angle_2
    sisnr_sep_avg_angle_1 = sisnr_sep_total_angle_1 / sisnr_num_angle_1
    sisnr_sep_avg_angle_2 = sisnr_sep_total_angle_2 / sisnr_num_angle_2
    sisnr_sep_avg_angle = sisnr_sep_total_angle / sisnr_num_angle

    print('---------------Azi diff Performance---------------')

    for azidist_index in range(len(sisnr_imp_avg_angle_1)):
        angle_start = azidist_index * 15
        angle_end = angle_start + 14
        print('Range: {}~{} | Num: {}'.format(angle_start, angle_end, sisnr_num_angle_1[azidist_index]))
        print('ORI | PESQ_1: {:2.4f} | PESQ_2:{:2.4f} | SI-SNR_1: {:2.4f} | SI-SNR_2: {:2.4f} | PESQ_AVG: {:2.4f} | SI-SNR_AVG: {:2.4f}'.format(
            pesq_ori_avg_angle_1[azidist_index], pesq_ori_avg_angle_2[azidist_index],
            sisnr_ori_avg_angle_1[azidist_index], sisnr_ori_avg_angle_2[azidist_index],
            (pesq_ori_avg_angle_1[azidist_index]+pesq_ori_avg_angle_2[azidist_index])/2, (sisnr_ori_avg_angle_1[azidist_index]+sisnr_ori_avg_angle_2[azidist_index])/2))
        print('SEP | PESQ_1: {:2.4f} | PESQ_2:{:2.4f} | SI-SNR_1: {:2.4f} | SI-SNR_2: {:2.4f} | PESQ_AVG: {:2.4f} | SI-SNR_AVG: {:2.4f} | SI-SNR_AVG: {:2.4f}'.format(
            pesq_sep_avg_angle_1[azidist_index], pesq_sep_avg_angle_2[azidist_index],
            sisnr_sep_avg_angle_1[azidist_index], sisnr_sep_avg_angle_2[azidist_index],
            (pesq_sep_avg_angle_1[azidist_index]+pesq_sep_avg_angle_2[azidist_index])/2, (sisnr_sep_avg_angle_1[azidist_index]+sisnr_sep_avg_angle_2[azidist_index])/2,
            sisnr_sep_avg_angle[azidist_index]))
        print('IMP | PESQ_1: {:2.4f} | PESQ_2:{:2.4f} | SI-SNR_1: {:2.4f} | SI-SNR_2: {:2.4f} | PESQ_AVG: {:2.4f} | SI-SNR_AVG: {:2.4f}'.format(
            pesq_imp_avg_angle_1[azidist_index], pesq_imp_avg_angle_2[azidist_index],
            sisnr_imp_avg_angle_1[azidist_index], sisnr_imp_avg_angle_2[azidist_index],
            (pesq_imp_avg_angle_1[azidist_index]+pesq_imp_avg_angle_2[azidist_index])/2, (sisnr_imp_avg_angle_1[azidist_index]+sisnr_imp_avg_angle_2[azidist_index])/2))
    
    print('<15 | PESQ: {:2.4f} | SI-SNR:{:2.4f}'.format(pesq_azidiff[0] / num_per_angle[0], sisnr_azidiff[0] / num_per_angle[0]))
    print('15-45 | PESQ: {:2.4f} | SI-SNR:{:2.4f}'.format(pesq_azidiff[1] / num_per_angle[1], sisnr_azidiff[1] / num_per_angle[1]))
    print('45-90 | PESQ: {:2.4f} | SI-SNR:{:2.4f}'.format(pesq_azidiff[2] / num_per_angle[2], sisnr_azidiff[2] / num_per_angle[2]))
    print('>90 | PESQ: {:2.4f} | SI-SNR:{:2.4f}'.format(pesq_azidiff[3] / num_per_angle[3], sisnr_azidiff[3] / num_per_angle[3]))

    print('CKPT {} '
          '| {:2.3f}s/batch | time {:2.1f}mins '
          '| loss {:2.6f} | loss_as {:2.6f} | loss_sig {:2.6f} '
          '| loss_as_1 {:2.6f} | loss_as_2 {:2.6f} | loss_sig_1 {:2.6f} | loss_sig_2 {:2.6f} '
          '| Azi MAE {:2.4f} | Azi MAE_1 {:2.4f} | Azi MAE_2 {:2.4f} '.format(
              epoch,
              eplashed,
              (etime - stime) / 60.0,
              loss_avg,
              loss_as_1_avg + loss_as_2_avg,
              loss_sig_1_avg + loss_sig_2_avg,
              loss_as_1_avg,
              loss_as_2_avg,
              loss_sig_1_avg,
              loss_sig_2_avg,
              azi_MAE_avg,
              azi_MAE_1_avg,
              azi_MAE_2_avg,
              ))

    sys.stdout.flush()
    return loss_avg


def load_checkpoint(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def main(args):
    cuda_flag = 1
    device = torch.device('cuda' if cuda_flag else 'cpu')
    torch.cuda.set_device(0)
    model = Tree(n_avb_mics=args.n_avb_mics)

    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# of parameters:', k, flush=True)

    print("=" * 40, "Model Structures", "=" * 40)
    for module_name, m in model.named_modules():
        if module_name == '':
            print(m)
    print("=" * 98)
    model.to(device)

    checkpoint = load_checkpoint(args.ckpt_path, cuda_flag)
    model.load_state_dict(checkpoint['model'], strict=False)
    test(model, args, 1, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PyTorch Version Enhancement')
    # model path
    parser.add_argument(
        '--ckpt-path',
        dest='ckpt_path',
        type=str,
        default='/Work21/2020/yinhaoran/MIMO_DBnet/plan_d_2sources_1_1/best_0',
        help='the exp dir')
    parser.add_argument(
        '--log-dir',
        dest='log_dir',
        type=str,
        default='/Work21/2020/yinhaoran/MIMO_DBnet/plan_d_2sources_1_1/best_0/log',
        help='the random seed')
    # data path/
    parser.add_argument(
        '--tt-clean ',
        dest='tt_clean',
        type=str,
        default='/Work21/2020/yinhaoran/simulated_list/0220_2speakers/test_0220.lst',
        help='the test clean data list')
    # train process configuration
    parser.add_argument(
        '--segment_length',
        dest='segment_length',
        type=int,
        default=4,
        help='the segment length')
    parser.add_argument(
        '--learn-rate',
        dest='learn_rate',
        type=float,
        default=1e-4,
        help='the learning rate in training')
    parser.add_argument(
        '--max-epoch',
        dest='max_epoch',
        type=int,
        default=100,
        help='the max epochs ')
    parser.add_argument(
        '--dropout',
        dest='dropout',
        type=float,
        default=0.4,
        help='the probility of dropout')
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        type=int,
        default=4,
        help='the batch size in train')
    parser.add_argument(
        '--use-cuda',
        dest='use_cuda',
        default=1,
        type=int,
        help='use cuda')
    parser.add_argument(
        '--seed ',
        dest='seed',
        type=int,
        default=20,
        help='the random seed')
    parser.add_argument(
        '--num-threads',
        dest='num_threads',
        type=int,
        default=10)
    parser.add_argument(
        '--num-gpu',
        dest='num_gpu',
        type=int,
        default=1,
        help='the num gpus to use')
    parser.add_argument(
        '--weight-decay',
        dest='weight_decay',
        type=float,
        default=0.0000001)
    parser.add_argument(
        '--clip-grad-norm',
        dest='clip_grad_norm',
        type=float,
        default=3)
    parser.add_argument(
        '--sample-rate',
        dest='sample_rate',
        type=int,
        default=16000)
    parser.add_argument(
        '--write-wav',
        dest='write_wav',
        type=bool,
        default=False)
    parser.add_argument(
        '--n_avb_mics',
        dest='n_avb_mics',
        type=int,
        default=2)
    parser.add_argument('--retrain', dest='retrain', type=int, default=1)
    FLAGS, _ = parser.parse_known_args()
    FLAGS.use_cuda = FLAGS.use_cuda and torch.cuda.is_available()
    print('torch.cuda.is_available(): ', torch.cuda.is_available())
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    torch.cuda.manual_seed(FLAGS.seed)
    import pprint

    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__dict__)
    main(FLAGS)
