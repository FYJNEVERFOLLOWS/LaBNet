import sys
sys.path.append("/Work21/2021/fuyanjie/pycode/LaBNetPro")

from dataloader.dataloader import static_loader
from model.GRU import RNN_MASK
from torch_complex import ComplexTensor
from torch_complex import functional as FC
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from scipy.signal import get_window

EPS = 1e-8


def get_ipd(spec, dim):
    '''
    : param spec: (batch, channels, freq_bins:257 × 2,  time_bins)
    : param dim:
    : return:
    '''

    real = spec[:, :, :dim, :]
    imag = spec[:, :, dim:, :]
    phase = torch.atan2(imag,real) # (batch,channels, freq_bins:257, time_bins)
    # 8, 12, 16, 16, 20 cm
    # diff1 = torch.cos((phase[:, 2, :, :] - phase[:, 0, :, :])).unsqueeze(1)
    # diff2 = torch.cos((phase[:, 3, :, :] - phase[:, 2, :, :])).unsqueeze(1)
    # diff3 = torch.cos((phase[:, 4, :, :] - phase[:, 2, :, :])).unsqueeze(1)
    # diff4 = torch.cos((phase[:, 3, :, :] - phase[:, 1, :, :])).unsqueeze(1)
    # diff5 = torch.cos((phase[:, 3, :, :] - phase[:, 0, :, :])).unsqueeze(1)

    # 8, 8, 12, 16, 16 cm
    # diff1 = torch.cos((phase[:, 2, :, :] - phase[:, 0, :, :])).unsqueeze(1)
    # diff2 = torch.cos((phase[:, 3, :, :] - phase[:, 1, :, :])).unsqueeze(1)
    # diff3 = torch.cos((phase[:, 3, :, :] - phase[:, 2, :, :])).unsqueeze(1)
    # diff4 = torch.cos((phase[:, 4, :, :] - phase[:, 2, :, :])).unsqueeze(1)
    # diff5 = torch.cos((phase[:, 5, :, :] - phase[:, 3, :, :])).unsqueeze(1)

    diff1 = torch.cos((phase[:, 1, :, :] - phase[:, 0, :, :])).unsqueeze(1)
    diff2 = torch.cos((phase[:, 2, :, :] - phase[:, 0, :, :])).unsqueeze(1)
    diff3 = torch.cos((phase[:, 3, :, :] - phase[:, 0, :, :])).unsqueeze(1)
    diff4 = torch.cos((phase[:, 4, :, :] - phase[:, 0, :, :])).unsqueeze(1)
    diff5 = torch.cos((phase[:, 5, :, :] - phase[:, 0, :, :])).unsqueeze(1)

    sin_diff1 = torch.sin((phase[:, 1, :, :] - phase[:, 0, :, :])).unsqueeze(1)
    sin_diff2 = torch.sin((phase[:, 2, :, :] - phase[:, 0, :, :])).unsqueeze(1)
    sin_diff3 = torch.sin((phase[:, 3, :, :] - phase[:, 0, :, :])).unsqueeze(1)
    sin_diff4 = torch.sin((phase[:, 4, :, :] - phase[:, 0, :, :])).unsqueeze(1)
    sin_diff5 = torch.sin((phase[:, 5, :, :] - phase[:, 0, :, :])).unsqueeze(1)

    # result = torch.cat((diff1, diff2, diff3, diff4, diff5), dim=1)
    result = torch.cat((diff1,diff2,diff3,diff4,diff5,sin_diff1,sin_diff2,sin_diff3,sin_diff4,sin_diff5), dim=1)
    #(batch, (m-1) pair microphones, freq bins:257, time_bins)
    return result

def get_lps(spec, dim):
    '''
    : param spec: (batch,channels, freq _bins:257 × 2, time_bins)
    : param dim:
    : return:
    '''
    real = spec[:, :, :dim, :]
    imag = spec[:, :, dim:, :]
    mags = torch.sqrt(real ** 2 + imag ** 2)
    mags_refch = mags[:, 0, :, : ].unsqueeze(1)
    result = torch.log(mags_refch ** 2 + EPS) - np.log(EPS) # (batch, 1,freq_bins :257, time_bins)
    return result

def get_spec_mag(spec, dim):
    '''
    :param spec: (batch, channels, freq bins:257 × 2, time_bins)
    :param dim:
    :return:
    '''
    real = spec[:, :, :dim, :]
    imag = spec[:, :, dim:, :]
    mags = torch.sqrt(real ** 2 + imag ** 2)
    mags_refch = mags[:, 0, :, :].unsqueeze(1)
    return mags_refch


def get_covariance_v2(spec, crf, filter_size):
    batch, channels, two_freq_bins, time_bins = spec.size()
    freq_bins = two_freq_bins // 2
    spec = spec.reshape(batch, channels, 2, freq_bins, time_bins)
    spec = spec.permute(0, 1, 2, 4, 3)
    pad = nn.ZeroPad2d(1)
    spec_horizental = pad(spec)
    result_horizental = spec_horizental[:,:,:,:time_bins,:freq_bins+1]
    for i in range(1, filter_size):
        result_horizental = torch.cat((result_horizental, spec_horizental[:,:,:,i:i+time_bins,:freq_bins+1]),dim=3)
    
    spec_vertical = pad(result_horizental)
    result_vertical = spec_vertical[:,:,:,1:filter_size*time_bins+1,1:1+freq_bins]
    for j in range(1,filter_size):
        result_vertical = torch.cat((result_vertical,spec_vertical[:,:,:,1:3*time_bins+1,1+j:1+j+freq_bins]),dim=4)
    
    spec_final = result_vertical.reshape(batch, channels, 2, filter_size, time_bins, filter_size * freq_bins)
    spec_final = spec_final.reshape(batch, channels, 2, filter_size, time_bins, filter_size, freq_bins)
    spec_final = spec_final.permute(0, 1, 2, 4, 6, 3, 5)

    spec_real = spec_final[:,:,0,...]
    spec_imag = spec_final[:,:,1,...]
    spec_com = ComplexTensor(spec_real, spec_imag)
    spec_com = spec_com.permute(0, 3, 1, 2, 4, 5)
    crf_com = ComplexTensor(crf[:,0,...],crf[:,1,...])
    crf_com = crf_com.permute(0, 2, 1, 3, 4)
    psd_Y = FC.einsum("...ctkl,...etkl->...tcekl",[spec_com,spec_com.conj()])
    psd = psd_Y * crf_com[...,None,None,:,:]
    psd = psd.sum(dim=-1)
    psd = psd.sum(dim=-1)

    convariance_final_real = psd.real
    convariance_final_real = convariance_final_real.permute(0, 2, 1, 3, 4)
    convariance_final_imag = psd.imag
    convariance_final_imag = convariance_final_imag.permute(0, 2, 1, 3, 4)

    convariance_final = torch.stack((convariance_final_real,convariance_final_imag),dim=1)

    return convariance_final

def remove_dc(signal):
    """Normalized to zero mean"""
    mean = torch.mean(signal, dim=-1, keepdim=True)
    signal = signal - mean
    return signal

def dotproduct(y, y_hat) :
    #batch x channel x nsamples
    return torch.bmm(y.reshape(y.shape[0], 1, y.shape[ -1]), y_hat.reshape(y_hat.shape[0], y_hat.shape[-1], 1)).reshape(-1)

def si_sdr_loss(e1, c1, c2):
    # [B, T]
    def sisdr(estimated, original, eps=1e-8):
        # estimated = remove_dc(estimated)
        # original = remove_dc(original)
        target = pow_norm(estimated, original) * original / (pow_p_norm(original) + eps)
        noise = estimated - target
        return -10 * torch.log10(eps + pow_p_norm(target) / pow_p_norm(noise) + eps)

    sisdr_loss = sisdr(e1, c1)
    avg_loss = torch.mean(sisdr_loss)

    return avg_loss

def wsdr_loss(output, target_signal, inference_signal):
    """
    : param output: B, 1, T
    : param target_signal: (batch, time samples)
    : param inference_signal: (batch, time_samples)
    : return:
    """

    output = torch.squeeze(output, 1)

    y = target_signal
    z = inference_signal

    # target size: torch.Size([32, 15988B]) noise torch.Size([32, 159888]) output: torch.Size([32, 159800])# print ( "target size:", target_sig.size(),"noise ", noise.size(), 'output : ' , output .size(
    y_hat = output
    z_hat = y + z - y_hat  # expected noise signal

    y_norm = torch.norm(y, dim=-1)
    z_norm = torch.norm(z, dim=-1)
    y_hat_norm = torch.norm(y_hat, dim=-1)
    z_hat_norm = torch.norm(z_hat, dim=-1)

    def loss_sdr(a, a_hat,  a_norm, a_hat_norm):
        return dotproduct(a, a_hat) / (a_norm * a_hat_norm + EPS)

    alpha = y_norm.pow(2) / (y_norm.pow(2) + z_norm.pow(2) + EPS)
    loss_wSDR = -alpha * loss_sdr(y, y_hat, y_norm, y_hat_norm) - (1 - alpha) * loss_sdr(z, z_hat, z_norm, z_hat_norm)

    return loss_wSDR.mean()

def pow_p_norm_np(signal):
    """Compute 2 Norm"""
    return np.square(np.linalg.norm(signal, ord=2, axis=-1))

def pow_norm_np(s1, s2):
    return np.sum(s1 * s2, axis=-1)

def pow_p_norm(signal):
    """Compute 2 Norm"""
    return torch.pow(torch.norm(signal, p=2, dim=-1, keepdim=True), 2)

def pow_norm(s1, s2):
    return torch.sum(s1 * s2, dim=-1, keepdim=True)
    
def pit_sisdr_loss(e1, e2, c1, c2):
    # [1, T]
    def sisdr(estimated, original):
        # estimated = remove_dc(estimated)
        # original = remove_dc(original)
        target = pow_norm(estimated, original) * original / pow_p_norm(original)
        noise = estimated - target
        return 10 * torch.log10(pow_p_norm(target) / pow_p_norm(noise))

    e1b = e1.squeeze() # [T]
    e2b = e2.squeeze() # [T]
    c1b = c1.squeeze() # [T]
    c2b = c2.squeeze() # [T]

    sdr1 = (sisdr(e1b, c1b) + sisdr(e2b, c2b)) * 0.5
    sdr2 = (sisdr(e2b, c1b) + sisdr(e1b, c2b)) * 0.5

    loss, idx = torch.max(torch.stack((sdr1, sdr2), dim=-1), dim=-1)
    avg_loss = torch.mean(loss)

    return avg_loss, idx

def pit_sisdr_numpy(e1, e2, c1, c2):
    # [1, T]
    def sisdr(estimated, original):
        # estimated = remove_dc(estimated)
        # original = remove_dc(original)
        target = pow_norm_np(estimated, original) * original / pow_p_norm_np(original)
        noise = estimated - target
        return 10 * np.log10(pow_p_norm_np(target) / pow_p_norm_np(noise))

    e1b = np.squeeze(e1) # [T]
    e2b = np.squeeze(e2) # [T]
    c1b = np.squeeze(c1) # [T]
    c2b = np.squeeze(c2) # [T]

    sdr1 = (sisdr(e1b, c1b) + sisdr(e2b, c2b)) * 0.5
    sdr2 = (sisdr(e2b, c1b) + sisdr(e1b, c2b)) * 0.5
    print(f'sdr for permutation 1 {sdr1} sdr for permutation 2 {sdr2}', flush=True)
    loss = np.max(np.stack((sdr1, sdr2), axis=-1), axis=-1)
    idx = np.argmax(np.stack((sdr1, sdr2), axis=-1), axis=-1)
    avg_loss = np.mean(loss)

    return avg_loss, idx

def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)  # **0.5

    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T

    if invers:
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None, :, None].astype(np.float32))


class ConvSTFT(nn.Module):
    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConvSTFT, self).__init__()

        if fft_len == None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len

        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        # self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)
        inputs = F.pad(inputs, [self.win_len - self.stride, self.win_len - self.stride])
        outputs = F.conv1d(inputs, self.weight, stride=self.stride)

        if self.feature_type == 'complex':
            return outputs
        else:
            dim = self.dim // 2 + 1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real ** 2 + imag ** 2)
            phase = torch.atan2(imag, real)
            return mags, phase


class ConviSTFT(nn.Module):
    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConviSTFT, self).__init__()
        if fft_len == None:   
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)
        # self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.stride = win_inc
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:, None, :])

    def forward(self, inputs, phase=None):
        """
        inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        """

        if phase is not None:
            real = inputs * torch.cos(phase)
            imag = inputs * torch.sin(phase)
            inputs = torch.cat([real, imag], 1)
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride)

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        outputs = outputs / (coff + 1e-8)
        # outputs = torch.where(coff == 0, outputs, outputs/coff)
        outputs = outputs[..., self.win_len - self.stride:-(self.win_len - self.stride)]

        return outputs

class Triangulation(nn.Module):
    def __init__(self):
        super(Triangulation, self).__init__()

    def forward(self, doas, mic_itval=28):
        """
        doas : [B*F, T, n_avb_mics]
        xy: [B*F, T, 2]
        """
        if doas.shape[-1] == 4:
            doas = torch.index_select(doas, -1, torch.tensor([0, 3], device=doas.device)) # [B*F, T, 2]

        # doas = torch.max(azi_s, dim=-1)[1] # [B, T, n_avb_mics] if input [B, T, n_avb_mics, 210] 
        # if azi_s.shape[-1] == 4:
        #     doas = torch.index_select(azi_s, -1, torch.tensor([0, 3])) # [B, T, 2]
        
        doa1 = doas[:, :, 0]
        doa2 = doas[:, :, 1] 
        if (doa1 == doa2).any():
            doa1 = doa1 + 1
        doa1 = doa1 / 180.0 * np.pi
        doa2 = doa2 / 180.0 * np.pi

        side = torch.clamp(torch.sin(doa2) * mic_itval / (torch.sin(doa1-doa2) + 1e-8), min=50, max=800)

        x = -side * torch.cos(doa1)
        y = side * torch.sin(doa1)
        xy = torch.stack((x, y), dim=2)
        # print(f'side {side} doa1 {doa1} doa2 {doa2} xy {xy}', flush=True)
        return xy


class Tree(nn.Module):
    def __init__(self, fft_len=512,speaker_num=2,n_avb_mics=2):
        super(Tree, self).__init__()
        self.channels = 6
        self.dim = fft_len // 2 + 1
        self.win_len = fft_len
        self.win_inc = fft_len // 2
        self.fft_len = fft_len
        self.win_type = 'hamming'
        self.filter_size = 3
        self.stftConv = ConvSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, feature_type='complex')
        self.istftConv = ConviSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, feature_type='complex')
        self.P = 3
        self.R = 4
        self.X = 8
        self.causal = True
        self.norm_type = 'cLN'
        self.n_mics = 6 # len(mic_location)
        self.n_avb_mics = n_avb_mics
        self.speaker_num = speaker_num
        self.reduction_linear = nn.Linear(in_features = (10+1)*257,out_features=256)
    
        self.repeats = RNN_MASK()
    
        self.width_window_length = 2
    
        self.RELU = nn.ReLU()
        self.PRELU = nn.PReLU()
        self.Sigmoid = nn.Sigmoid()
        self.covariance_ln_ss = nn.LayerNorm([self.n_mics, self.n_mics])
        self.covariance_ln_nn = nn.LayerNorm([self.n_mics, self.n_mics])
    
        self.conv1_doa = nn.Sequential(
            nn.Conv2d(144, 210*self.n_avb_mics, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(210*self.n_avb_mics), nn.ReLU(inplace=True)
        )
        self.conv_emb_doa = nn.Sequential(
            nn.Conv2d(210, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(1), nn.ReLU(inplace=True)
        )
        self.conv2_doa = nn.Sequential(
            nn.Conv2d(257, 1, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2)),
            nn.BatchNorm2d(1), nn.ReLU(inplace=True)
        )

        self.gru_doa = nn.GRU(210, hidden_size=210, num_layers=2, batch_first=True)

        ###
        # self.triangulation = nn.Linear(self.n_avb_mics * 210, 400)
        self.triangulation = Triangulation()
        # self.conv1_loc = nn.Sequential(
        #     nn.Conv2d(210*self.n_avb_mics, 400, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2)),
        #     nn.BatchNorm2d(400), nn.ReLU(inplace=True)
        # )
        # self.conv2_loc = nn.Sequential(
        #     nn.Conv2d(257, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.BatchNorm2d(1), nn.ReLU(inplace=True)
        # )
        ###

        # Branch 1 (gru_bss)
        self.linear_bss_1 = nn.Linear(self.n_mics * self.n_mics * 2 * 2 + self.n_avb_mics * 210 + 2, 300)
        # self.linear_bss_1 = nn.Linear(self.n_mics * self.n_mics * 2 * 2 + self.n_avb_mics * 210, 300)
        self.gru_bss_1 = nn.GRU(300, hidden_size=300, num_layers=2, batch_first=True)
        self.linear_w1 = nn.Linear(300, self.channels * 2)

        # Branch 2 (gru_bss)
        self.linear_bss_2 = nn.Linear(self.n_mics * self.n_mics * 2 * 2 + self.n_avb_mics * 210 + 2, 300)
        # self.linear_bss_2 = nn.Linear(self.n_mics * self.n_mics * 2 * 2 + self.n_avb_mics * 210, 300)
        self.gru_bss_2 = nn.GRU(300, hidden_size=300, num_layers=2, batch_first=True)
        self.linear_w2 = nn.Linear(300, self.channels * 2)
    
    def get_params(self, weight_decay=0.0):
        #add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def forward(self, inputs):
        """
        Returns
        -------
        azis_1 / azis_2: [B, T, n_mics=self.n_avb_mics, 210]
        es_sig: [B, 1, T]
        """
        self.gru_doa.flatten_parameters()
        self.gru_bss_1.flatten_parameters()
        self.gru_bss_2.flatten_parameters()
        inputs = torch.unsqueeze(inputs, 2) # bs * channels * 1 * time_samples
        batch, channels, _, time_samples = inputs.size()
        inputs = inputs.view(batch * channels, 1, time_samples) # bs * channels, l, time_samples

        #STFT
        #que stftConv返回的维度是2 * 257吗
        spectrograms = self.stftConv(inputs)
        _, double_freq_bins, time_bins = spectrograms.size() # batch * channels, 2 * 257,time bins
        freq_bins = double_freq_bins // 2
        spectrograms = spectrograms.reshape(batch, channels, double_freq_bins, time_bins)

        '''compute ipd , lps and width feature,concat'''
        ipd = get_ipd(spectrograms, self.dim)       # batch, M microphone pair num, freq_bins, time_bin$
        lps = get_spec_mag(spectrograms, self.dim)  # batch, 1, freq_bins, time_bins

        audio_blocks = torch.cat((ipd, lps), dim=1) #(batch, n_mics, freq_bins:257, time_bins
        audio_blocks = audio_blocks.view(batch, (10+1) * 257, time_bins)
        audio_blocks = audio_blocks.permute(0,2,1) # [B, time_bins, 6*freq_bins]

        '''pass through front-GRU '''
        audio_reduction = self.reduction_linear(audio_blocks)  # batch, time_bins, 256 channels
        masks = self.repeats(audio_reduction) # batch, time_bins, 256 channels

        convariance_set_doa_list = []
        convariance_set_bss_list = []

        """audio embeddings经过blstm得到 crf,crf与spectrograms得到目标信号的协方差矩阵"""

        for mask_idx in range(self.speaker_num):
            comlex_filter_ss = masks[mask_idx * 2].permute(0,2,1)
            comlex_filter_nn = masks[mask_idx * 2 + 1].permute(0,2,1)

            comlex_filter_ss = comlex_filter_ss.view(batch, freq_bins, 2 * self.filter_size * self.filter_size, time_bins)
            comlex_filter_ss = comlex_filter_ss.view(batch, freq_bins, 2, self.filter_size * self.filter_size,  time_bins)
            comlex_filter_ss = comlex_filter_ss.view(batch, freq_bins, 2, self.filter_size, self.filter_size, time_bins)
            comlex_filter_ss = comlex_filter_ss.permute(0, 2, 5, 1, 3, 4)
            covariance_ss = get_covariance_v2(spectrograms, comlex_filter_ss, self.filter_size)
            covariance_ss = self.covariance_ln_ss(covariance_ss)
            covariance_ss = covariance_ss.unsqueeze(dim=4)

            comlex_filter_nn = comlex_filter_nn.view(batch, freq_bins, 2 * self.filter_size * self.filter_size, time_bins)
            comlex_filter_nn = comlex_filter_nn.view(batch, freq_bins, 2, self.filter_size * self.filter_size,  time_bins)
            comlex_filter_nn = comlex_filter_nn.view(batch, freq_bins, 2, self.filter_size, self.filter_size, time_bins)
            comlex_filter_nn = comlex_filter_nn.permute(0, 2, 5, 1, 3, 4)
            covariance_nn = get_covariance_v2(spectrograms, comlex_filter_nn, self.filter_size)
            covariance_nn = self.covariance_ln_nn(covariance_nn)
            covariance_nn = covariance_nn.unsqueeze(dim=4)

            # TODO
            covariance_set = torch.cat((covariance_ss, covariance_nn), dim=4)  # B,2,T,F,2,6,6

            covariance_set = covariance_set.permute(2, 0, 3, 1, 4, 5, 6)
            # (time bins, batch, freq bins, 2, 2, channels, channels)
            covariance_set = covariance_set.reshape(time_bins, batch, freq_bins, 2, 2, channels *channels)
            covariance_set = covariance_set.reshape(time_bins, batch, freq_bins, 2, 2* channels *channels)
            covariance_set = covariance_set.reshape(time_bins, batch, freq_bins, 2* 2* channels *channels)

            covariance_set_doa = covariance_set
            covariance_set_doa = covariance_set_doa.permute(1,3,0,2)
            convariance_set_doa_list.append(covariance_set_doa) # (B, 144, T, F)

            covariance_set = covariance_set.reshape(time_bins, batch * freq_bins, 2* 2* channels *channels)
            covariance_set = covariance_set.permute(1,0,2) # [B*F, T, 144]

            convariance_set_bss_list.append(covariance_set) # [B*F, T, 144]
        
        '''Branch_1'''      
        # DoA_1
        # as1_freq = self.conv1_doa(convariance_set_doa_list[0]) # [B, 210*n_avb_mics, T, F]
        # as1_loc_emb = as1_freq.permute(0, 3, 2, 1) # [B, F, T, 210*n_avb_mics] 
        # as1_freq = as1_freq.reshape(batch, 210, self.n_avb_mics, time_bins, freq_bins) # [B, 210, n_avb_mics, T, F]
        # as1_freq = as1_freq.reshape(batch, 210, self.n_avb_mics*time_bins, freq_bins) # [B, 210, T*n_avb_mics, F]
        # as1_freq_emb = as1_freq # [B, 210, T*n_avb_mics, F]
        # as1_freq = as1_freq.reshape(batch, 210, time_bins, self.n_avb_mics, freq_bins)
        # as1_freq = as1_freq.reshape(batch*self.n_avb_mics, 210, time_bins, freq_bins)
        # as1_freq = as1_freq.permute(0, 3, 2, 1) # [B*n_avb_mics, F, T, 210]
        # as1_freq_emb = self.conv_emb_doa(as1_freq_emb) # [B, 1, T*n_avb_mics, F]
        # as1_freq_emb = as1_freq_emb.squeeze(dim=1) # [B, T*n_avb_mics, F]
        # as1_freq_emb = as1_freq_emb.reshape(batch, time_bins, freq_bins, self.n_avb_mics) # [B, T, F, n_avb_mics]
        # as1_freq_emb = as1_freq_emb.reshape(batch * freq_bins, time_bins, self.n_avb_mics) # [B*F, T, n_avb_mics]

        ###
        as1_freq = self.conv1_doa(convariance_set_doa_list[0]) # [B, 210*n_avb_mics, T, F]
        as1_freq = as1_freq.permute(0, 3, 2, 1) # [B, F, T, 210*n_avb_mics]
        as1_freq_emb = as1_freq.reshape(batch * freq_bins, time_bins, 210*self.n_avb_mics) # [B*F, T, 210*n_avb_mics]
        as1_freq = as1_freq.reshape(batch, freq_bins, time_bins, 210, self.n_avb_mics)
        as1_freq = as1_freq.reshape(batch*self.n_avb_mics, freq_bins, time_bins, 210) # [B*n_avb_mics, F, T, 210]

        as_1 = self.conv2_doa(as1_freq) # (B*n_avb_mics, 1, T, 210)
        as_1 = as_1.squeeze(dim=1) # [B*n_avb_mics, T, 210]
        azis_1, _ = self.gru_doa(as_1) # [B*n_avb_mics, T, 210]
        azis_1 = azis_1.reshape(batch, self.n_avb_mics, time_bins, 210) # [B, n_avb_mics, T, 210]
        azis_1 = azis_1.permute(0, 2, 1, 3) # [B, T, n_mics=n_avb_mics, 210]
        doas_1 = torch.max(azis_1, dim=3)[1]
        xy1 = self.triangulation(doas_1) # [B, T, 2]
        xy1_emb = torch.repeat_interleave(xy1.unsqueeze(dim=1),repeats=257,dim=1) # [B, F, T, 2]
        xy1_emb = xy1_emb.reshape(batch*freq_bins, time_bins, 2) # [B*F, T, 2]

        # BSS_1
        convariance_set_with_sps_1 = torch.cat((convariance_set_bss_list[0], as1_freq_emb, xy1_emb), dim=2)
        # convariance_set_with_sps_1 = torch.cat((convariance_set_bss_list[0], as1_freq_emb), dim=2)
        covariance_set_bss_1 = self.linear_bss_1(convariance_set_with_sps_1) # [B*F, T, 300]
        gru_output_w1, _ = self.gru_bss_1(covariance_set_bss_1) # [B*F, T, 300]
        gru_output_w1 = gru_output_w1.reshape(batch, freq_bins, time_bins, 300)
        gru_output_w1 = gru_output_w1.permute(2, 0, 1, 3)
        w1 = self.linear_w1(gru_output_w1) # (time_bins, batch, freq_bins, channels * 2)
        w1 = w1.reshape(time_bins, batch, freq_bins, channels, 2) # (time_bins, batch, freq_bins, channels, 2)
        beamformer_1 = w1.permute(1, 0, 4, 2, 3) # (batch, time_bins, 2, freq_bins, channels)
        beamformer_1 = beamformer_1.unsqueeze(dim=5) # (batch, time_bins, 2, freq_bins, channels, 1)

        '''Branch_2'''      
        # DoA_2
        # as2_freq = self.conv1_doa(convariance_set_doa_list[1]) # [B, 210*n_avb_mics, T, F]
        # as2_freq = as2_freq.reshape(batch, 210, self.n_avb_mics, time_bins, freq_bins)
        # as2_freq = as2_freq.reshape(batch, 210, self.n_avb_mics*time_bins, freq_bins) # [B, 210, T*n_avb_mics, F]
        # as2_freq_emb = as2_freq # [B, 210, T*n_avb_mics, F]
        # as2_freq = as2_freq.reshape(batch, 210, time_bins, self.n_avb_mics, freq_bins)
        # as2_freq = as2_freq.reshape(batch*self.n_avb_mics, 210, time_bins, freq_bins)
        # as2_freq = as2_freq.permute(0, 3, 2, 1) # [B*n_avb_mics, F, T, 210]        
        # as2_freq_emb = self.conv_emb_doa(as2_freq_emb) # [B, 1, T*n_avb_mics, F]
        # as2_freq_emb = as2_freq_emb.squeeze(dim=1) # [B, T*n_avb_mics, F]
        # as2_freq_emb = as2_freq_emb.reshape(batch, time_bins, freq_bins, self.n_avb_mics) # [B, T, F, n_avb_mics]
        # as2_freq_emb = as2_freq_emb.reshape(batch * freq_bins, time_bins, self.n_avb_mics) # [B*F, T, n_avb_mics]

        ###
        as2_freq = self.conv1_doa(convariance_set_doa_list[1]) # [B, 210*n_avb_mics, T, F]
        as2_freq = as2_freq.permute(0, 3, 2, 1) # [B, F, T, 210*n_avb_mics]
        as2_freq_emb = as2_freq.reshape(batch * freq_bins, time_bins, 210*self.n_avb_mics) # [B*F, T, 210*n_avb_mics]
        as2_freq = as2_freq.reshape(batch, freq_bins, time_bins, 210, self.n_avb_mics)
        as2_freq = as2_freq.reshape(batch*self.n_avb_mics, freq_bins, time_bins, 210) # [B*n_avb_mics, F, T, 210]

        as_2 = self.conv2_doa(as2_freq) # (B*n_avb_mics, 1, T, 210)
        as_2 = as_2.squeeze(dim=1) # [B*n_avb_mics, T, 210]
        azis_2, _ = self.gru_doa(as_2) # [B*n_avb_mics, T, 210]
        azis_2 = azis_2.reshape(batch, self.n_avb_mics, time_bins, 210) # [B, n_avb_mics, T, 210]
        azis_2 = azis_2.permute(0, 2, 1, 3) # [B, T, n_mics=n_avb_mics, 210]
        doas_2 = torch.max(azis_2, dim=3)[1]
        xy2 = self.triangulation(doas_2) # [B, T, 2]
        xy2_emb = torch.repeat_interleave(xy2.unsqueeze(dim=1),repeats=257,dim=1) # [B, F, T, 2]
        xy2_emb = xy2_emb.reshape(batch*freq_bins, time_bins, 2) # [B*F, T, 2]

        # BSS_2
        convariance_set_with_sps_2 = torch.cat((convariance_set_bss_list[1], as2_freq_emb, xy2_emb), dim=2)
        # convariance_set_with_sps_2 = torch.cat((convariance_set_bss_list[1], as2_freq_emb), dim=2)
        covariance_set_bss_2 = self.linear_bss_2(convariance_set_with_sps_2)
        gru_output_w2, _ = self.gru_bss_2(covariance_set_bss_2)
        gru_output_w2 = gru_output_w2.reshape(batch, freq_bins, time_bins, 300)
        gru_output_w2 = gru_output_w2.permute(2, 0, 1, 3)
        w2 = self.linear_w2(gru_output_w2) # (time_bins, batch, freq_bins, channels * 2)
        w2 = w2.reshape(time_bins, batch, freq_bins, channels, 2) # (time_bins, batch, freq_bins, channels, 2)
        beamformer_2 = w2.permute(1, 0, 4, 2, 3) # (batch, time_bins, 2, freq_bins, channels)
        beamformer_2 = beamformer_2.unsqueeze(dim=5) # (batch, time_bins, 2, freq_bins,channels, 1)

        '''Beamformer Separation'''
        spec = torch.unsqueeze(spectrograms, 4) # (batch, channels, 2*freq_bins,time_bins, 1)
        # (batch, channels, 2, freq_bins, time bins, 1)
        spec = spec.reshape(batch, channels, 2, freq_bins, time_bins, 1)
        spec = spec.permute(0, 4, 2, 3, 1, 5) # (batch, time bins, 2, freq_bins, channels, 1)
        spec_real = spec[:, :, 0, :, :, :]# (batch,  time bins, freq bins, channels, 1)
        spec_imag = spec[:, :, 1, :, :, :] # (batch,time bins, freq bins,channels, 1)

        beamformer_list = []
        extract_signal_list = []
        beamformer_list.append(beamformer_1)
        beamformer_list.append(beamformer_2)

        for beamformer in beamformer_list:
            beamformer_real = beamformer[:, :, 0, ...]  # (batch,time bins, freq_bins,channels, 1)
            beamformer_imag = beamformer[:, :, 1, ...]  # (batch, time bins, freq bins,channels, 1)
            beamformer_real = beamformer_real.permute(0, 1, 2, 4, 3)    # (batch, time_ bins, freq_bins,1, channels)
            beamformer_imag = beamformer_imag.permute(0, 1, 2, 4, 3)   #(batch, time_bins, freqg_bins, 1, channels)

            enhancement_real = torch.matmul(beamformer_real, spec_real) - torch.matmul(beamformer_imag,  spec_imag)
            enhancement_imag = torch.matmul(beamformer_real, spec_imag) + torch.matmul(beamformer_imag,  spec_real)#( batch, time bins, freq bins, 1, 1)

            enhancement_real = enhancement_real.squeeze(dim=-1) # (batch, time_bins, freq_bins, 1)
            enhancement_imag = enhancement_imag.squeeze(dim=-1) # (batch, time_bins,freq_bins, 1)

            enhancement_real = enhancement_real.permute(0, 1, 3, 2)
            enhancement_imag = enhancement_imag.permute(0, 1, 3, 2)

            enhancement = torch.cat((enhancement_real, enhancement_imag), dim=2)  # batch, time_bins, 2, freq_bins
            enhancement = enhancement. reshape(batch, time_bins, 2 * freq_bins) # batch, time bins,  2 * freq bins
            enhancement = enhancement.permute(0, 2, 1)  # batch, 2 * freq bins, time_bins
            '''do iSTFT'''
            extract_signal = self.istftConv(enhancement)  # batch, 1, time_samples
            extract_signal_list.append(extract_signal)

        extract_signal_1 = extract_signal_list[0] # [B, 1, T]
        extract_signal_2 = extract_signal_list[1]

        return azis_1, azis_2, extract_signal_1, extract_signal_2
        # return azis_1, azis_2, xy1, xy2, extract_signal_1, extract_signal_2

if __name__ == '__main__':
    # e1 = np.random.randn(4, 32000)
    # e2 = np.random.randn(4, 32000)
    # c1 = np.random.randn(4, 32000)
    # c2 = np.random.randn(4, 32000)
    # print(f'e1.shape {e1.shape} e2.shape {e2.shape}')
    # print(f'c1.shape {c1.shape} c2.shape {c2.shape}')
    # print(pit_sisdr_numpy(e1, e2, c1, c2))
    inputs = torch.randn((4, 6, 64000))
    model = Tree()
    model.to(torch.device('cpu'))
    sps_1, sps_2, es_sig_1, es_sig_2 = model(inputs)
    print(f'sps_1.shape {sps_1.shape} sps_2.shape {sps_2.shape}', flush=True)
    print(f'es_sig_1.shape {es_sig_1.shape} es_sig_2.shape {es_sig_2.shape}', flush=True)



