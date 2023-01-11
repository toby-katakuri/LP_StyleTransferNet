import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




#### with edge fusion
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(nn.ReflectionPad2d([1, 1, 1, 1]),
                                        nn.Conv2d(dim, dim, (3, 3)),
                                        nn.ReLU(),
                                        nn.ReflectionPad2d([1, 1, 1, 1]),
                                        nn.Conv2d(dim, dim, (3, 3)))

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = 1

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = 1
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class Trans_mid_encoder_edge(nn.Module):
    def __init__(self, input_dim=3, isEdge=False):
        super(Trans_mid_encoder_edge, self).__init__()
        self.isEdge = isEdge
        DownBlock1 = []
        DownBlock2 = []
        DownBlock3 = []
        DownBlock1 += [
            nn.ReflectionPad2d([1, 1, 1, 1]),
            nn.Conv2d(input_dim, 16, (3, 3)),
            nn.ReLU()
        ]
        DownBlock1 += [
            nn.ReflectionPad2d([1, 1, 1, 1]),
            nn.Conv2d(16, 16, (3, 3), stride=2),
            nn.ReLU()
        ]

        DownBlock2 += [
            nn.ReflectionPad2d([1, 1, 1, 1]),
            nn.Conv2d(16, 32, (3, 3), stride=2),
            nn.ReLU()
        ]

        self.DownBlock1 = nn.Sequential(*DownBlock1)
        self.DownBlock2 = nn.Sequential(*DownBlock2)


        self.pam1 = _Module(16)
        self.pam2 = PAM_Module(32)





    def forward(self, edge):

        results = []
        out = self.DownBlock1(edge)
        result1 = self.pam1(out)
        results.append(result1)
        out = self.DownBlock2(out)
        result2 = self.pam2(out)
        results.append(result2)

        return results



class Trans_mid_encoder_image(nn.Module):
    def __init__(self, input_dim=9):
        super(Trans_mid_encoder_image, self).__init__()
        model1 = [nn.Conv2d(9, 32, 3, padding=1, padding_mode='reflect'),
                 nn.InstanceNorm2d(32),
                 nn.LeakyReLU(),
                 nn.Conv2d(32, 64, 3, padding=1, padding_mode='reflect'),
                 nn.LeakyReLU()]

        for _ in range(3):
            model1 += [ResidualBlock(64)]

        model2 = [nn.Conv2d(96, 16, 3, padding=1, padding_mode='reflect'),
                  nn.LeakyReLU(),
                  nn.Conv2d(16, 3, 3, padding=1, padding_mode='reflect')]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)

        self.c1 = nn.Conv2d(3, 32, 3, padding=1, padding_mode='reflect')
        self.pam = CAM_Module(32)

    def forward(self, c_low, sed_low, cr_mid, c_edge):
        c_mid = F.interpolate(c_low, [256, 256])
        sed_mid = F.interpolate(sed_low, [256, 256])
        x = torch.cat([c_mid, sed_mid, cr_mid], dim=1)
        #x = torch.cat([c_mid, cr_mid], dim=1)
        out1 = self.model1(x)
        out2 = self.pam(self.c1(c_edge))
        out = torch.cat([out1, out2], dim=1)
        out = self.model2(out)
        return out




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


class Trans_high(nn.Module):
    def __init__(self, num_residual_blocks=3, num_high=3):
        super(Trans_high, self).__init__()

        self.num_high = num_high

        model = [nn.Conv2d(9, 32, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(32)]

        model += [nn.Conv2d(32, 1, 3, padding=1, padding_mode='reflect')]

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