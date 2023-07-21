from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
import attention
from loss import FusionLoss
from option import args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# shallow feature extraction
class Feature_Extraction(nn.Module):
    def __init__(self):
        super(Feature_Extraction, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 128, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self,oe,ue):
        x_cat = torch.cat((oe,ue),1)
        x = self.conv(x_cat)
        return x


# local adaptive learning network
class Position_Attention(nn.Module):
    def __init__(self):
        super(Position_Attention, self).__init__()
        self.PA_module =nn.Sequential(OrderedDict([
            ('conv0', nn.Sequential(nn.Conv2d(128, 8, 3, 1, 1), nn.LeakyReLU(0.2))),
            ('GAP', nn.AdaptiveAvgPool3d((1, None, None))),
            ('MaxPool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),

            ('conv1', nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1), nn.LeakyReLU(0.2))),
            ('AvgPool', nn.AvgPool2d(kernel_size=2, stride=2, padding=0)),

            ('conv2', nn.Sequential(nn.Conv2d(8, 16, 3, 1, 1), nn.LeakyReLU(0.2))),
            ('subpixel_conv1', nn.PixelShuffle(2)),

            ('conv_k1', nn.Sequential(nn.Conv2d(12, 4, 1), nn.LeakyReLU(0.2))),
            ('subpixel_conv2', nn.PixelShuffle(2)),

            ('sigmoid', nn.Sigmoid())
        ]))

    def forward(self, f0):
        f0 = self.PA_module.conv0(f0)
        f = self.PA_module.GAP(f0)
        f = self.PA_module.MaxPool(f)
        f_conv1 = self.PA_module.conv1(f)
        f = self.PA_module.AvgPool(f_conv1)
        f = self.PA_module.conv2(f)
        f = self.PA_module.subpixel_conv1(f)
        f = F.interpolate(f, f_conv1.shape[2:])
        f = torch.cat((f_conv1, f), dim = 1)
        f = self.PA_module.conv_k1(f)
        f = self.PA_module.subpixel_conv2(f)
        f = F.interpolate(f, f0.shape[2:])
        w = self.PA_module.sigmoid(f)
        return w


class Fusion_Network(nn.Module):
    def __init__(self):
        super(Fusion_Network, self).__init__()
        n_feats = args.n_feats
        self.fea_extr = Feature_Extraction()
        self.enlca = attention.ENLCA(
            channel=n_feats, reduction=2,
            res_scale=args.res_scale)
        self.position_attention = Position_Attention()
        self.fusion_layer1 = nn.Sequential(
                nn.Conv2d(384, 256, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(256, 128, 3, 1, 1))
        self.fusion_layer2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1))

    def forward(self,oe,ue):
        fea_extr = self.fea_extr(oe,ue)

        # collaborative aggregation module
        global_seq = self.enlca(fea_extr)
        local_map = self.position_attention(fea_extr)
        local_seq = local_map * fea_extr

        # fusion
        fusion_cat = torch.cat((fea_extr,local_seq,global_seq),1)
        fused_m1 = self.fusion_layer1(fusion_cat)
        fused_m2 = fused_m1 + fea_extr
        fused_m = torch.tanh(self.fusion_layer2(fused_m2))
        # fused_m = fused_m / 2 + 0.5
        # fused_oe = oe * fused_m
        # fused_ue = ue * fused_m
        return fused_m, fea_extr


# recursive refinement module
class EnhanceNetwork(nn.Module):
    def __init__(self, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=65, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=kernel_size,stride=1,padding=padding),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=65, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=65, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )


    def forward(self, input):
        b, c, h, w = input.shape
        # out = torch.zeros([b,1,h,w],dtype=float).to(device)
        out = torch.zeros([b, 1, h, w]).to(device)

        x = torch.cat((input,out),1)
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.conv3(x)

        x = torch.cat((input,out),1)
        x = self.conv4(x)
        x = self.conv5(x)
        out = self.conv6(x)

        x = torch.cat((input, out), 1)
        x = self.conv7(x)
        x = self.conv8(x)
        out = self.conv9(x)
        return out


class MEF_Network(nn.Module):
    def __init__(self):
        super(MEF_Network, self).__init__()
        self.f_net = Fusion_Network()
        self.e_net = EnhanceNetwork(channels=64)
        self.conv_i1 = nn.Sequential(
            nn.Conv2d(128,64,3,1,1),
            nn.ReLU(),)
        self._fusionloss = FusionLoss()

    def forward(self,oe,ue):
        f_m, f_ext = self.f_net(oe,ue)
        t = self.conv_i1(f_ext)
        map = self.e_net(t)
        map = map / 2 + 0.5
        refined_out = map * f_m

        return refined_out

    def _floss(self,oe,ue):
        f_img = self(oe,ue)
        fusion_loss = self._fusionloss(oe,ue,f_img)

        return fusion_loss




if __name__ == '__main__':
    ue = torch.rand(4, 1, 128, 128)#.to(device)
    oe = torch.rand(4, 1, 128, 128)#.to(device)
    mef = MEF_Network()
    a = mef(oe, ue)
    print(a.shape)
