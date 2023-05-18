import os
import sys
sys.path.append("/Work21/2021/fuyanjie/pycode/LaBNetPro")
import torch
import torch.nn as nn
import numpy as np
import argparse
import torch.optim as optim
import time
import logging
from tools.misc import get_learning_rate, save_checkpoint, reload_model, setup_lr
from utils.utils import doa_err_2_source
from model.Tree import Tree, si_sdr_loss, wsdr_loss
from dataloader.dataloader import static_loader

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

def validation(model, args, lr, epoch, device):
    dataloader = static_loader(
        clean_scp=args.cv_clean,
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
    num_target = 0
    num_acc = 0
    num_pred = 0
    sum_err = 0.0
    loss_total = 0.0
    loss_as_1_total = 0.0
    loss_as_2_total = 0.0
    loss_sig_1_total = 0.0
    loss_sig_2_total = 0.0
    MAE_total = 0
    MAE_1_total = 0
    MAE_2_total = 0

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

            es_AS_1, es_AS_2, es_sig_1, es_sig_2 = model(inputs)
            es_AS_1 = es_AS_1[:, 0:249, :, :] # [B, T, n_avb_mics, F]
            es_AS_2 = es_AS_2[:, 0:249, :, :] # [B, T, n_avb_mics, F]
            
            loss_as_1 = mse_loss(es_AS_1, gt_AS_arr[:, :, 0, :, :])
            loss_as_2 = mse_loss(es_AS_2, gt_AS_arr[:, :, 1, :, :])
            loss_sig_1 = wsdr_loss(es_sig_1, target_1, target_2)
            loss_sig_2 = wsdr_loss(es_sig_2, target_2, target_1)

            # weighted
            loss_as_1 = loss_as_1 * args.w_azimuth
            loss_as_2 = loss_as_2 * args.w_azimuth
            loss_sig_1 = loss_sig_1 * args.w_separation
            loss_sig_2 = loss_sig_2 * args.w_separation

            loss = loss_as_1 + loss_as_2 + loss_sig_1 + loss_sig_2

            mae_1, mae_2, num_acc_1, num_acc_2, num_pred_1, num_pred_2 = doa_err_2_source(gt_azi_arr, es_AS_1, es_AS_2)
            mae = (mae_1 + mae_2)/2
            num_acc += num_acc_1 + num_acc_2
            num_pred += num_pred_1 + num_pred_2
            sum_err += mae_1 * num_pred_1 + mae_2 * num_pred_2

            loss_total += loss.data.cpu()
            loss_as_1_total += loss_as_1.data.cpu()
            loss_as_2_total += loss_as_2.data.cpu()
            loss_sig_1_total += loss_sig_1.data.cpu()
            loss_sig_2_total += loss_sig_2.data.cpu()
            MAE_total += mae
            MAE_1_total += mae_1
            MAE_2_total += mae_2

            del inputs, gt_AS_arr, es_AS_1, es_AS_2, loss, loss_as_1, loss_as_2, loss_sig_1, loss_sig_2, mae, mae_1, mae_2
    
    if not num_pred == 0:
        print('DOA Overall ACC frame-level {:2.4f} '.format(num_acc / num_pred))
        print('DOA Overall MAE frame-level {:2.4f} '.format(sum_err / num_pred))
        print(f'TOTAL pred frames: DOA {num_pred}')

    etime = time.time()
    eplashed = (etime - stime) / num_batch

    loss_avg = loss_total / num_batch
    loss_as_1_avg = loss_as_1_total / num_batch
    loss_as_2_avg = loss_as_2_total / num_batch
    loss_sig_1_avg = loss_sig_1_total / num_batch
    loss_sig_2_avg = loss_sig_2_total / num_batch
    MAE_avg = MAE_total / num_batch
    MAE_1_avg = MAE_1_total / num_batch
    MAE_2_avg = MAE_2_total / num_batch

    print('CKPT {} '
          '| {:2.3f}s/batch | time {:2.1f}mins '
          '| loss {:2.6f} | loss_as {:2.6f} | loss_sig {:2.6f} '
          '| loss_as_1 {:2.6f} | loss_as_2 {:2.6f} | loss_sig_1 {:2.6f} | loss_sig_2 {:2.6f} '
          '| MAE {:2.4f} | MAE_1 {:2.4f} | MAE_2 {:2.4f} '.format(
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
              MAE_avg,
              MAE_1_avg,
              MAE_2_avg,
              ))
    sys.stdout.flush()
    return loss_avg


def train_process(model, args, device, writer, mix_info_list):
    print('preparing data... args.segment_length = ', args.segment_length)
    # torch.cuda.empty_cache()
    dataloader = static_loader(
        clean_scp=args.tr_clean,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_threads,
        sample_rate=args.sample_rate,
        data_mix_info=None,
        n_avb_mics=args.n_avb_mics
    )
    print_freq = 2000
    num_batch = len(dataloader)
    print("num_batch ", num_batch)
    print(f'args.num_gpu {args.num_gpu} {type(args.num_gpu)}')
    # multi-gpu
    if(args.num_gpu > 1):
        params = model.module.get_params(args.weight_decay)
    # single-gpu
    else:
        params = model.get_params(args.weight_decay)

    optimizer = optim.Adam(params, lr=args.learn_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=2, verbose=True)

    if args.retrain:
        start_epoch, step = reload_model(model, optimizer, args.exp_dir,
                                         args.use_cuda)
    else:
        start_epoch, step = 0, 0

    print('---------PRERUN-----------')
    print('(Initialization)')


    warmup_epoch = 1
    check_steps = 100 * print_freq
    warmup_lr = args.learn_rate / (2 ** warmup_epoch)
    
    model.to(device)

    mse_loss = nn.MSELoss()

    for epoch in range(start_epoch, args.max_epoch):
        torch.manual_seed(args.seed + epoch)
        if args.use_cuda:
            torch.cuda.manual_seed(args.seed + epoch)
        model.train()
        loss_total = 0.0
        loss_print = 0.0
        loss_as_1_total = 0.0
        loss_as_1_print = 0.0
        loss_as_2_total = 0.0
        loss_as_2_print = 0.0
        loss_sig_1_total = 0.0
        loss_sig_1_print = 0.0
        loss_sig_2_total = 0.0
        loss_sig_2_print = 0.0
        stime = time.time()
        if epoch == 0 and warmup_epoch > 0:
            print(
                'Use warmup stragery,and the lr is set to {:.5f} '.format(warmup_lr))
            setup_lr(optimizer, warmup_lr)
            warmup_lr *= 2
        elif epoch == warmup_epoch:
            print('The warmup was end,and the lr is set to {:.5f}'.format(
                args.learn_rate))
            setup_lr(optimizer, args.learn_rate)

        lr = get_learning_rate(optimizer)
        for idx, egs in enumerate(dataloader):
            batch_start = time.time()

            # load to gpu
            egs = load_obj(egs, device)
            inputs = egs["mixed_data"] # [B, C, T]
            gt_AS_arr = egs["doa_as_array"] # [B, T, S, n_mics, 210]
            gt_azi_arr = egs["doa_idx_array"] # [B, T, S, n_mics]
            target_1 = egs["target_1"] # [B, T]
            target_2 = egs["target_2"] # [B, T]

            model.zero_grad()
            es_AS_1, es_AS_2, es_sig_1, es_sig_2 = model(inputs)

            es_AS_1 = es_AS_1[:, 0:249, :, :] # [B, T, n_avb_mics, F]
            es_AS_2 = es_AS_2[:, 0:249, :, :] # [B, T, n_avb_mics, F]

            loss_as_1 = mse_loss(es_AS_1, gt_AS_arr[:, :, 0, :, :])
            loss_as_2 = mse_loss(es_AS_2, gt_AS_arr[:, :, 1, :, :])
            loss_sig_1 = wsdr_loss(es_sig_1, target_1, target_2)
            loss_sig_2 = wsdr_loss(es_sig_2, target_2, target_1)

            # weighted
            loss_as_1 = loss_as_1 * args.w_azimuth
            loss_as_2 = loss_as_2 * args.w_azimuth
            loss_sig_1 = loss_sig_1 * args.w_separation
            loss_sig_2 = loss_sig_2 * args.w_separation

            loss = loss_as_1 + loss_as_2 + loss_sig_1 + loss_sig_2

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            optimizer.step()
            
            step += 1
            loss_total += loss.data.cpu()
            loss_print += loss.data.cpu()
            loss_as_1_total += loss_as_1.data.cpu()
            loss_as_1_print += loss_as_1.data.cpu()
            loss_as_2_total += loss_as_2.data.cpu()
            loss_as_2_print += loss_as_2.data.cpu()
            loss_sig_1_total += loss_sig_1.data.cpu()
            loss_sig_1_print += loss_sig_1.data.cpu()
            loss_sig_2_total += loss_sig_2.data.cpu()
            loss_sig_2_print += loss_sig_2.data.cpu()

            if (idx + 1) % print_freq == 0:
                batch_time = time.time() - batch_start
                avg_time = (time.time() - stime)/(idx + 1)
                loss_print_avg = loss_print / print_freq
                loss_as_1_print_avg = loss_as_1_print / print_freq
                loss_as_2_print_avg = loss_as_2_print / print_freq
                loss_sig_1_print_avg = loss_sig_1_print / print_freq
                loss_sig_2_print_avg = loss_sig_2_print / print_freq
                
                print('Epoch {:3d}/{:3d} | batches {:5d}/{:5d} | lr {:1.4e} | Current {:2.3f}s/batches '
                      '| AVG {:2.3f}s/batches | loss {:2.6f} | loss_as {:2.6f} | loss_sig {:2.6f} '
                      '| loss_as_1 {:2.6f} | loss_as_2 {:2.6f} | loss_sig_1 {:2.6f} | loss_sig_2 {:2.6f}'.format(
                          epoch + 1, args.max_epoch, idx + 1, num_batch, lr,
                          batch_time, avg_time, loss_print_avg, loss_as_1_print_avg + loss_as_2_print_avg,
                          loss_sig_1_print_avg + loss_sig_2_print_avg,
                          loss_as_1_print_avg, loss_as_2_print_avg,
                          loss_sig_1_print_avg, loss_sig_2_print_avg), flush=True)
                # sys.stdout.flush()
                loss_print = 0.0
                loss_as_1_print = 0.0
                loss_as_2_print = 0.0
                loss_sig_1_print = 0.0
                loss_sig_2_print = 0.0

        eplashed = time.time() - stime
        loss_avg = loss_total / (step - (epoch) * num_batch)
        loss_as_1_avg = loss_as_1_total / (step - (epoch) * num_batch)
        loss_as_2_avg = loss_as_2_total / (step - (epoch) * num_batch)
        loss_sig_1_avg = loss_sig_1_total / (step - (epoch) * num_batch)
        loss_sig_2_avg = loss_sig_2_total / (step - (epoch) * num_batch)

        print(
            'Training AVG.LOSS |'
            'Epoch {:3d}/{:3d} | lr {:1.4e} | '
            '{:2.3f}s/batch | time {:3.2f}mins | '
            'loss {:2.6f} | loss_as {:2.6f} | loss_sig {:2.6f} '
            'loss_as_1 {:2.6f} | loss_as_2 {:2.6f} | loss_sig_1 {:2.6f} | loss_sig_2 {:2.6f}'.format(
                epoch + 1,
                args.max_epoch,
                lr,
                eplashed / check_steps,
                eplashed / 60.0,
                loss_avg,
                loss_as_1_avg + loss_as_2_avg,
                loss_sig_1_avg + loss_sig_2_avg,
                loss_as_1_avg,
                loss_as_2_avg,
                loss_sig_1_avg,
                loss_sig_2_avg,
                ), flush=True)
        val_loss = validation(model, args, lr, epoch, device)
        model.train()

        # if iteration after warmup_epoch,reset lr sechel to normal
        if epoch >= warmup_epoch:
            print('Rejected !!! The best is {:2.6f} '.format(scheduler.best))
            logging.info(' Rejected !!! The best is {:2.6f} model epoch = {:3d} '.format(
                scheduler.best, epoch))
            save_checkpoint(model, optimizer, epoch + 1,
                            step, args.exp_dir, val_loss)
            scheduler.step(val_loss)
            sys.stdout.flush()
        else:
            save_checkpoint(model, optimizer, epoch + 1,
                            step, args.exp_dir, val_loss)


def main(args):
    cuda_flag = 1
    device = torch.device('cuda' if cuda_flag else 'cpu')
    torch.cuda.set_device(0)
    model = Tree(n_avb_mics=args.n_avb_mics)

    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)
    if not os.path.exists(os.path.join(args.exp_dir, 'log')):
        os.mkdir(os.path.join(args.exp_dir, 'log'))

    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# of parameters:', k, flush=True)

    print("=" * 40, "Model Structures", "=" * 40)
    for module_name, m in model.named_modules():
        if module_name == '':
            print(m)
    print("=" * 98)

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    train_process(model, FLAGS, device, 0, mix_info_list=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PyTorch Version Enhancement')
    # model path
    parser.add_argument(
        '--exp-dir',
        dest='exp_dir',
        type=str,
        default='/Work21/2021/fuyanjie/pycode/MIMO_DBnet/1-10/exp0915',
        help='the exp dir')
    parser.add_argument(
        '--log-dir',
        dest='log_dir',
        type=str,
        default='/Work21/2021/fuyanjie/pycode/MIMO_DBnet/1-10/exp0915/log',
        help='the random seed')

    # data path
    parser.add_argument(
        '--tr-clean',
        dest='tr_clean',
        type=str,
        default='/Work21/2021/fuyanjie/pycode/LaBNetwoDE/data/exp_list/train-clean-100_1126.lst',
        help='the train clean data list')
    parser.add_argument(
        '--cv-clean ',
        dest='cv_clean',
        type=str,
        default='/Work21/2021/fuyanjie/pycode/LaBNetwoDE/data/exp_list/dev-clean_1126.lst',
        help='the validation clean data list')
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
        default=1,
        help='the batch size in train')
    parser.add_argument(
        '--use-cuda',
        dest='use_cuda',
        default=1,
        type=int,
        help='use cuda')
    parser.add_argument(
        '--seed',
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
        '--alpha',
        dest='w_azimuth',
        type=float,
        default=1)
    parser.add_argument(
        '--beta',
        dest='w_separation',
        type=float,
        default=10)
    parser.add_argument(
        '--n_avb_mics',
        dest='n_avb_mics',
        type=int,
        default=2)
    parser.add_argument('--retrain', dest='retrain', type=int, default=1)
    FLAGS, _ = parser.parse_known_args()
    FLAGS.use_cuda = FLAGS.use_cuda and torch.cuda.is_available()
    print('torch.cuda.is_available(): ', torch.cuda.is_available())
    os.makedirs(FLAGS.exp_dir, exist_ok=True)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    
    torch.cuda.manual_seed(FLAGS.seed)
    import pprint

    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__dict__)
    main(FLAGS)
