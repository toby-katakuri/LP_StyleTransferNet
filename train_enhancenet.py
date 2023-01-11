from make_dataset import EnhanceNet_Dataset
from BaseNet import BaseNet
import torchvision.utils
import torch.utils.data as data
from EnhanceNet import EnhanceNet
from tqdm import tqdm
from utils import make_laplace_pyramid, generate_path
import os


### option ###
style = '026'
number = '1'
new_number = 'for_loss_fig_edge'
### path ###
root_c =  r'F:\datasets\chinese_content_512'
root_s = os.path.join(r'F:\pyproject\lap_cin\data\style_images', style + '.jpg')
root_sed = os.path.join(r'F:\datasets\lap_cin\basenet', style + '_' + number)
check_image = generate_path(r'F:\pyproject\lap_cin\check_images_enhance', style + '_' + number + new_number)

saved_para = generate_path(r'F:\pyproject\lap_cin\saved_paras_enhance', style + '_' + number + new_number)

saved_loss = generate_path(r'F:\pyproject\lap_cin\saved_loss_wosanet', style + '_' + number + new_number)
epochs = 10
###  ###



dataset = EnhanceNet_Dataset(root_c, root_s, root_sed)
data_loader = data.DataLoader(dataset, batch_size=1, shuffle=False)

net = EnhanceNet().cuda()
net.train()
batch_id = 0
loss_g = 0
for epoch in range(epochs):
    for data in tqdm(data_loader):
        batch_id += 1
        content_pyr = make_laplace_pyramid(data['content'], 2)
        net.set_input(content_pyr, data['content'], data['style'], data['stylized'])
        net.train_iter()
        #net.train_iter_with_gan(batch_id)

        # loss_g += net.show_loss()
        #
        # if batch_id == 1 or batch_id % 100 == 0:
        #     net.show_pic(check_image, batch_id)
        #
        #     print(loss_g/batch_id)

net.save_loss(saved_loss)

