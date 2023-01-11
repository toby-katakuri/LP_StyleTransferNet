import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
from enhance_modules import Trans_mid, Trans_high, vgg4, NLayerDiscriminator
from loss_function import CalcStyleLoss, CalcContentLoss, CalcStyleEmdLoss, CalcContentReltLoss, GANLoss
from modules import Transform, vgg, decoder
from loss_function import calc_mean_std, mean_variance_norm
import torch.nn.functional as F

class EnhanceNet(nn.Module):
    def __init__(self, vgg_path='vgg_normalised.pth'):
        super(EnhanceNet, self).__init__()

        self.net_mid = Trans_mid().cuda()
        self.net_high = Trans_high().cuda()
        #
        # self.net_mid.load_state_dict(torch.load(r'D:\PycharmProjects\lap_cin\saved_paras_enhance\001_3w1\net_mid.pt'))
        # self.net_high.load_state_dict(torch.load(r'D:\PycharmProjects\lap_cin\saved_paras_enhance\001_3w1\net_high.pt'))

        #
        # # Gan
        # self.discri = NLayerDiscriminator().cuda()
        # self.optimizer_d = torch.optim.Adam(self.discri.parameters(), betas=[0.5, 0.99], lr=1e-4)
        ###

        self.e4 = vgg4().cuda()

        self.optimizer = torch.optim.Adam([{'params': self.net_mid.parameters()}, {'params': self.net_high.parameters()}], lr=1e-4)

        # loss

        self.calc_content_loss = CalcContentLoss()
        self.calc_style_loss = CalcStyleLoss()
        self.calc_style_emd_loss = CalcStyleEmdLoss()
        self.calc_content_relt_loss = CalcContentReltLoss()

        self.gan_criterion = GANLoss(gan_mode='vanilla')

        self.content_layers = ['r1_1', 'r2_1', 'r3_1', 'r4_1']
        self.style_layers = ['r1_1', 'r2_1', 'r3_1', 'r4_1']

        self.loss_c_l = []
        self.loss_s_l = []


    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_input(self, c_pyr, c, s, sed):
        self.c_low = c_pyr[-1].cuda()
        self.cr_mid = c_pyr[1].cuda()
        self.cr_high = c_pyr[0].cuda()
        #
        self.c = c.cuda()
        self.s = s.cuda()
        self.sed_low = sed.cuda()

    def forward_net(self):
        sedr_mid = self.net_mid(self.c_low, self.sed_low, self.cr_mid)
        sed_mid = F.interpolate(self.sed_low, [256, 256]) + sedr_mid
        sedr_high = self.net_high(sedr_mid, sed_mid, self.cr_high)
        self.sed_high = sedr_high + F.interpolate(sed_mid, [512, 512])


    # without GAN
    def calculate_loss(self):
        self.se_f = self.e4(self.sed_high)
        self.c_f = self.e4(self.c)
        self.s_f = self.e4(self.s)

        self.loss_c = 0
        for layer in self.content_layers:
            self.loss_c += self.calc_content_loss(self.se_f[layer], self.c_f[layer], norm=True)

        self.loss_s = 0
        for layer in self.style_layers:
            self.loss_s += self.calc_style_loss(self.se_f[layer], self.s_f[layer])
        #
        # self.loss_style_remd1 = self.calc_style_emd_loss(
        #     F.interpolate(self.output_features['r2_1'], scale_factor=0.5),
        #     F.interpolate(self.s_features['r2_1'], scale_factor=0.5))

        self.loss_style_remd2 = self.calc_style_emd_loss(self.se_f['r3_1'], self.s_f['r3_1']) + \
                                self.calc_style_emd_loss(self.se_f['r4_1'], self.s_f['r4_1'])

        self.loss_content_relt = self.calc_content_relt_loss(self.se_f['r3_1'], self.c_f['r3_1']) + \
                                 self.calc_content_relt_loss(self.se_f['r4_1'], self.c_f['r4_1'])


        # self.loss_tv = calc_tv_loss(self.final_output)

        # calu
        self.loss_ct = 1 * self.loss_c + 15 * self.loss_content_relt
        self.loss_st = 80 * self.loss_style_remd2 + 50 * self.loss_s

        self.loss_G = 1.0 * self.loss_ct + 1.0 * self.loss_st
        # self.loss_G = self.loss_c_f + 10000 * self.loss_s_f
        self.loss_G.backward()


    def train_iter(self):
        self.optimizer.zero_grad()
        self.forward_net()
        self.calculate_loss()
        self.optimizer.step()

        self.loss_c_l.append(self.loss_ct.item())
        self.loss_s_l.append(self.loss_st.item())

    # WITH gan
    def backward_G(self):
        self.se_f = self.e4(self.sed_high)
        self.c_f = self.e4(self.c)
        self.s_f = self.e4(self.s)


        # G_loss
        self.loss_c = 0
        for layer in self.content_layers:
            self.loss_c += self.calc_content_loss(self.se_f[layer], self.c_f[layer], norm=True)

        self.loss_s = 0
        for layer in self.style_layers:
            self.loss_s += self.calc_style_loss(self.se_f[layer], self.s_f[layer])
        #
        # self.loss_style_remd1 = self.calc_style_emd_loss(
        #     F.interpolate(self.output_features['r2_1'], scale_factor=0.5),
        #     F.interpolate(self.s_features['r2_1'], scale_factor=0.5))

        self.loss_style_remd2 = self.calc_style_emd_loss(self.se_f['r3_1'], self.s_f['r3_1']) + \
                                self.calc_style_emd_loss(self.se_f['r4_1'], self.s_f['r4_1'])

        self.loss_content_relt = self.calc_content_relt_loss(self.se_f['r3_1'], self.c_f['r3_1']) + \
                                 self.calc_content_relt_loss(self.se_f['r4_1'], self.c_f['r4_1'])

        # self.loss_tv = calc_tv_loss(self.final_output)

        # GAN LOSS
        pred_fake = self.discri(self.sed_high)
        self.loss_gan_g = self.gan_criterion(pred_fake, True)

        # calu
        self.loss_G_1 = 1 * self.loss_c + 10 * self.loss_style_remd2 + 16 * self.loss_content_relt + 30 * self.loss_s
        self.loss_G = self.loss_G_1 + 10 * self.loss_gan_g

        self.loss_G.backward()

    def backward_D(self):
        pred_fake = self.discri(self.sed_high.detach())
        self.loss_gan_d_fake = self.gan_criterion(pred_fake, False)
        pred_real = self.discri(self.s)
        self.loss_gan_d_real = self.gan_criterion(pred_real, True)
        self.loss_D = (self.loss_gan_d_fake + self.loss_gan_d_real) / 0.5

        self.loss_D.backward()


    def train_iter_with_gan(self, batch_id):
        self.forward_net()

        # compute D

        self.set_requires_grad(self.discri, True)
        self.optimizer_d.zero_grad()
        self.backward_D()
        self.optimizer_d.step()

        # d

        self.set_requires_grad(self.discri, False)
        self.optimizer.zero_grad()
        self.backward_G()
        self.optimizer.step()




    def show_loss(self):
        return self.loss_G.item()

    def show_pic(self,path,batch_id):
        if batch_id == 1:
            torchvision.utils.save_image(self.s, path + '/{}_s.jpg'.format(batch_id))
        else:
            torchvision.utils.save_image(self.c, path + '/{}_c.jpg'.format(batch_id))
            torchvision.utils.save_image(self.sed_high, path + '/{}_output.jpg'.format(batch_id))

    def save_para(self, path):
        torch.save(self.net_mid.state_dict(), path + '/net_mid.pt')
        torch.save(self.net_high.state_dict(), path + '/net_high.pt')


    def save_loss(self, path):
        torch.save(self.loss_c_l, path + r'/loss_c.pt')
        torch.save(self.loss_s_l, path + r'/loss_s.pt')
        self.loss_c_l = []
        self.loss_s_l = []


class EnhanceNet_Test(nn.Module):
    def __init__(self, path):
        super(EnhanceNet_Test, self).__init__()

        self.net_mid = Trans_mid().cuda()
        self.net_high = Trans_high().cuda()

        self.net_mid.load_state_dict(torch.load(path + r'\net_mid.pt'))
        self.net_high.load_state_dict(torch.load(path + r'\net_high.pt'))


    def set_input(self, c_pyr, c, s, sed):
        self.c_low = c_pyr[-1].cuda()
        self.cr_mid = c_pyr[1].cuda()
        self.cr_high = c_pyr[0].cuda()
        #
        self.c = c.cuda()
        self.s = s.cuda()
        self.sed_low = sed.cuda()

    def forward_net(self):
        sedr_mid = self.net_mid(self.c_low, self.sed_low, self.cr_mid)
        sed_mid = F.interpolate(self.sed_low, [256, 256]) + sedr_mid
        sedr_high = self.net_high(sedr_mid, sed_mid, self.cr_high)
        self.sed_high = sedr_high + F.interpolate(sed_mid, [512, 512])


    def test_image(self, test_path, batch_id):
        self.forward_net()
        torchvision.utils.save_image(self.c, test_path + '/{}_c.jpg'.format(batch_id))
        torchvision.utils.save_image(self.sed_high, test_path + '/{}_se.jpg'.format(batch_id))







