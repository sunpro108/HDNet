import torch.nn as nn
import torch.fft as fft
import torch


class MaskWeightedMSE(nn.Module):
    def __init__(self, min_area=100):
        super(MaskWeightedMSE, self).__init__()
        self.min_area = min_area

    def forward(self, pred, label, mask):
        loss = (pred - label) ** 2
        reduce_dims = (1, 2, 3)
        delimeter = pred.size(1) * torch.clamp_min(torch.sum(mask, dim=reduce_dims), self.min_area)
        loss = torch.sum(loss, dim=reduce_dims) / delimeter 
        loss = torch.sum(loss) / pred.size(0)
        return loss

def calc_fft(image):
    '''image is tensor, N*C*H*W'''
    '''pytorch old version with pytorch <= 1.7.0
    # fft = torch.rfft(image, 2, onesided=False)
    # fft_mag = torch.log(1 + torch.sqrt(fft[..., 0] ** 2 + fft[..., 1] ** 2 + 1e-8))
    '''
    freq = fft.fftn(image, dim=(2, 3))
    freq = fft.fftshift(freq, dim=(2,3))
    fft_mag = torch.log(1 +torch.abs(freq) + 1e-8)
    return fft_mag


class MaskedFftLoss(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, fake_image, real_image, mask=None):
        """
        param:
        fake_image: b 3 h w, 
        real_image: b 3 h w

        """
        criterion_L1 = nn.L1Loss()
        fake_image_gray = fake_image[:,0]*0.299 + fake_image[:,1]*0.587 + fake_image[:,2]*0.114
        real_image_gray = real_image[:,0]*0.299 + real_image[:,1]*0.587 + real_image[:,2]*0.114

        if mask is not None:
            fake_image_gray = mask * fake_image_gray
            real_image_gray = mask * real_image_gray

        fake_fft = calc_fft(fake_image_gray)
        real_fft = calc_fft(real_image_gray)
        loss = criterion_L1(fake_fft, real_fft)
        return loss
