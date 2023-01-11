import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1, padding_mode='reflect'),
        )

    def forward(self, x):
        return x + self.block(x)





class Trans_mid(nn.Module):
    def __init__(self, num_residual_blocks=5):
        super(Trans_mid, self).__init__()

        model = [nn.Conv2d(9, 32, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 16, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Conv2d(16, 3, 3, padding=1, padding_mode='reflect')]

        self.model = nn.Sequential(*model)

    def forward(self, c_low, sed_low, cr_mid):
        c_mid = F.interpolate(c_low, [256, 256])
        sed_mid = F.interpolate(sed_low, [256, 256])
        x = torch.cat([c_mid, sed_mid, cr_mid], dim=1)
        out = self.model(x)
        return out

class Trans_high(nn.Module):
    def __init__(self, num_residual_blocks=3, num_high=3):
        super(Trans_high, self).__init__()

        self.num_high = num_high

        model = [nn.Conv2d(9, 64, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 1, 3, padding=1, padding_mode='reflect')]

        self.model = nn.Sequential(*model)


    def forward(self, sedr_mid, sed_mid, cr_high):
        sedr_high = F.interpolate(sedr_mid, [512, 512])
        sed_high = F.interpolate(sed_mid, [512, 512])
        x = torch.cat([cr_high, sedr_high, sed_high], dim=1)
        out = self.model(x)
        return out


path_e4_1 = r'F:\pyproject\lap_cin\pretrain_vgg\wct_pretrain_models\vgg_normalised_conv4_1.pth'

class vgg4(nn.Module):
    def __init__(self):
        super(vgg4, self).__init__()
        self.e4 = vgg_normalised_conv4_1.cuda()
        self.e4.load_state_dict(torch.load(path_e4_1))
        self.capture_layers = [1, 3, 6, 8, 11, 13, 15, 22, 29]

    def forward(self, x):
        r_dict = {}
        for i in range(len(self.e4)):
            x = self.e4[i](x)
            if i == 3:
                r_dict['r1_1'] = x
            elif i== 6:
                r_dict['r1_2'] = x
            elif i == 10:
                r_dict['r2_1'] = x
            elif i == 13:
                r_dict['r2_2'] = x
            elif i == 17:
                r_dict['r3_1'] = x
            elif i == 23:
                r_dict['r3_3'] = x
            elif i == 30:
                r_dict['r4_1'] = x

        return r_dict


vgg_normalised_conv4_1 = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),
)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)