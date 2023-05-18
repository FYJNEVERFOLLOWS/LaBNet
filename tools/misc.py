
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import soundfile as sf


def load_checkpoint(checkpoint_path,use_cuda) :
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path,map_location=lambda storage, loc: storage)
    return checkpoint


def get_learning_rate(optimizer):
    """Get learning rate"""
    return optimizer.param_groups[0]["lr"]

def reload_for_eval(model,checkpoint_dir, use_cuda) :
    ckpt_name = os.path.join(checkpoint_dir,'checkpoint_decode')
    if not os.path.exists(ckpt_name):
        print(f'file does not exists: {ckpt_name}')
        exit(1)
    if os.path.isfile(ckpt_name):
        with open(ckpt_name,'r') as f:
            model_name = f.readline().strip()
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        print(f'use model file: {checkpoint_path}')
        checkpoint = load_checkpoint(checkpoint_path,use_cuda)
        print(checkpoint['model'].keys( ))
        model.load_state_dict(checkpoint ['model'], strict=True)
        print('=> Reload well-trained model {} for decoding.'. format(model_name))


def reload_model(model, optimizer, checkpoint_dir, use_cuda=True, strict=True):
    ckpt_name = os.path.join(checkpoint_dir, 'checkpoint')
    if os.path.isfile(ckpt_name):
        with open(ckpt_name,'r') as f:
            model_name = f.readline().strip()
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)
        model.load_state_dict(checkpoint['model'],strict=strict)
        # optimizer.load state dict(checkpoint [ 'optimizer' ])
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        print ('=>Reload previous model and optimizer.')
    else:
        print('[!] checkpoint directory is empty. Train a new model ...')
        epoch = 0
        step = 0
    return epoch, step

def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir, val_loss):
    checkpoint_path = os.path.join(
        checkpoint_dir, 'model.ckpt-{}-{}.pt'.format(epoch, val_loss))
    torch.save({'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch' : epoch,
                'step' : step}, checkpoint_path)
    with open(os.path.join(checkpoint_dir, 'checkpoint'),'w' ) as f:
        f.write('model.ckpt-{}-{}.pt'.format(epoch, val_loss))
    print("=>Save checkpoint:", checkpoint_path)


def setup_lr(opt,lr):
    for param_group in opt.param_groups :
        param_group['lr'] = lr
