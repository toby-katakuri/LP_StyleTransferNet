import torch
import torch.nn as nn
import torchvision.utils

from modules import Transform, vgg, decoder
from loss_function import calc_mean_std, mean_variance_norm


vgg = vgg
decoder = decoder

class BaseNet(nn.Module):
    def __init__(self, vgg_path='vgg_normalised.pth'):
        super(BaseNet, self).__init__()

        # vgg
        vgg.load_state_dict(torch.load(vgg_path))
        encoder = nn.Sequential(*list(vgg.children())[:44])
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        # transform
        self.transform = Transform(in_planes=512)
        # decoder
        self.decoder = decoder

        self.mse_loss = nn.MSELoss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False


        # optimizer
        self.optimizer = torch.optim.Adam([{'params': self.decoder.parameters()}, {'params': self.transform.parameters()}], lr=5e-5)

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target, norm=False):
        if (norm == False):
            return self.mse_loss(input, target)
        else:
            return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def net_forward(self, content, style):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        stylized = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
        g_t = self.decoder(stylized)
        self.stylized = g_t
        g_t_feats = self.encode_with_intermediate(g_t)
        loss_c = self.calc_content_loss(g_t_feats[3], content_feats[3], norm=True) + self.calc_content_loss(
            g_t_feats[4], content_feats[4], norm=True)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        """IDENTITY LOSSES"""
        Icc = self.decoder(self.transform(content_feats[3], content_feats[3], content_feats[4], content_feats[4]))
        Iss = self.decoder(self.transform(style_feats[3], style_feats[3], style_feats[4], style_feats[4]))
        l_identity1 = self.calc_content_loss(Icc, content) + self.calc_content_loss(Iss, style)
        Fcc = self.encode_with_intermediate(Icc)
        Fss = self.encode_with_intermediate(Iss)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(Fss[0], style_feats[0])
        for i in range(1, 5):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + self.calc_content_loss(Fss[i],
                                                                                                     style_feats[i])
        return loss_c, loss_s, l_identity1, l_identity2


    def train_iter(self, content, style):
        self.content = content
        self.style = style
        l_c, l_s, l_i1, l_i2 = self.net_forward(self.content, self.style)
        self.l_c = 1.0 * l_c
        self.l_s = 1.5 * l_s
        self.loss = self.l_c + self.l_s + l_i1 + l_i2

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def test_image(self,content, style, save_path,batch_id, para_path):
        self.transform_test = Transform(in_planes=512).cuda()
        self.transform_test.load_state_dict(torch.load(para_path + '/transform.pt'))
        # decoder
        self.decoder_test = decoder.cuda()
        self.decoder_test.load_state_dict(torch.load(para_path + '/decoder.pt'))
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        stylized_t = self.transform_test(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
        stylized_test = self.decoder_test(stylized_t)


        torchvision.utils.save_image(stylized_test, save_path + '/{}.jpg'.format(str(batch_id).zfill(5)))

    def test_image_time(self,content, style, para_path):
        self.transform_test = Transform(in_planes=512).cuda()
        self.transform_test.load_state_dict(torch.load(para_path + '/transform.pt'))
        # decoder
        self.decoder_test = decoder.cuda()
        self.decoder_test.load_state_dict(torch.load(para_path + '/decoder.pt'))
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        stylized_t = self.transform_test(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
        stylized_test = self.decoder_test(stylized_t)


        return stylized_test

    def show_loss(self):
        return self.loss.item()
    def show_pic(self,path,batch_id):
        torchvision.utils.save_image(self.content, path + '/{}_c.jpg'.format(batch_id))
        torchvision.utils.save_image(self.stylized, path + '/{}_output.jpg'.format(batch_id))

    def save_para(self, path):
        torch.save(self.decoder.state_dict(), path + '/decoder.pt')
        torch.save(self.transform.state_dict(), path + '/transform.pt')

class BaseNet_Test(nn.Module):
    def __init__(self, vgg_path='vgg_normalised.pth',para_path=r'F:\pyproject\lap_cin\saved_paras\001_3'):
        super(BaseNet_Test, self).__init__()

        # vgg
        vgg.load_state_dict(torch.load(vgg_path))
        encoder = nn.Sequential(*list(vgg.children())[:44])
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1



        self.transform_test = Transform(in_planes=512).cuda()
        self.transform_test.load_state_dict(torch.load(para_path + '/transform.pt'))
        # decoder
        self.decoder_test = decoder.cuda()
        self.decoder_test.load_state_dict(torch.load(para_path + '/decoder.pt'))

        self.mse_loss = nn.MSELoss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False



    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target, norm=False):
        if (norm == False):
            return self.mse_loss(input, target)
        else:
            return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def net_forward(self, content, style):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        stylized = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
        g_t = self.decoder(stylized)
        self.stylized = g_t
        g_t_feats = self.encode_with_intermediate(g_t)
        loss_c = self.calc_content_loss(g_t_feats[3], content_feats[3], norm=True) + self.calc_content_loss(
            g_t_feats[4], content_feats[4], norm=True)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        """IDENTITY LOSSES"""
        Icc = self.decoder(self.transform(content_feats[3], content_feats[3], content_feats[4], content_feats[4]))
        Iss = self.decoder(self.transform(style_feats[3], style_feats[3], style_feats[4], style_feats[4]))
        l_identity1 = self.calc_content_loss(Icc, content) + self.calc_content_loss(Iss, style)
        Fcc = self.encode_with_intermediate(Icc)
        Fss = self.encode_with_intermediate(Iss)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(Fss[0], style_feats[0])
        for i in range(1, 5):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + self.calc_content_loss(Fss[i],
                                                                                                     style_feats[i])
        return loss_c, loss_s, l_identity1, l_identity2


    def train_iter(self, content, style):
        self.content = content
        self.style = style
        l_c, l_s, l_i1, l_i2 = self.net_forward(self.content, self.style)
        self.l_c = 1.0 * l_c
        self.l_s = 1.5 * l_s
        self.loss = self.l_c + self.l_s + l_i1 + l_i2

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def test_image_time(self,content, style):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        stylized_t = self.transform_test(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
        stylized_test = self.decoder_test(stylized_t)


        return stylized_test

    def show_loss(self):
        return self.loss.item()
    def show_pic(self,path,batch_id):
        torchvision.utils.save_image(self.content, path + '/{}_c.jpg'.format(batch_id))
        torchvision.utils.save_image(self.stylized, path + '/{}_output.jpg'.format(batch_id))

    def save_para(self, path):
        torch.save(self.decoder.state_dict(), path + '/decoder.pt')
        torch.save(self.transform.state_dict(), path + '/transform.pt')

