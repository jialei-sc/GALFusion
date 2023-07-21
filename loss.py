import torch.nn as nn
from utils import SSIM, TV_Loss
from option import args


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIM()
        self.tv_loss = TV_Loss()

    def forward(self,input,fused):
        mse = self.mse_loss(input,fused)
        ssim= 1 - self.ssim_loss(input,fused)
        tv = self.tv_loss(input,fused)
        fused_loss = args.lamda_mse * mse + args.lamda_ssim * ssim + args.lamda_tv * tv

        return fused_loss


class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.content_loss = ContentLoss()

    def forward(self, oe, ue, f_img):
        # fusion
        fusion_loss_oe = self.content_loss(oe,f_img)
        fusion_loss_ue = self.content_loss(ue,f_img)
        fusion_loss = fusion_loss_ue + fusion_loss_oe

        return fusion_loss

