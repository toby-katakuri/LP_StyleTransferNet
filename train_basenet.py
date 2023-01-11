from make_dataset import BaseNet_Dataset
from BaseNet import BaseNet
import torchvision.utils
import torch.utils.data as data
from BaseNet import BaseNet
from tqdm import tqdm
from utils import make_laplace_pyramid, generate_path
import os


### option ###
style = '017'
number = '3withbase'
### path ###
root_c = r'F:\datasets\chinese_content_512'
root_s = os.path.join(r'F:\pyproject\lap_cin\data\style_images', style + '.jpg')
check_image = generate_path(r'F:\pyproject\lap_cin\check_images', style + '_' + number)
saved_para = generate_path(r'F:\pyproject\lap_cin\saved_paras', style + '_' + number)
epochs = 10
###  ###



dataset = BaseNet_Dataset(root_c, root_s)
data_loader = data.DataLoader(dataset, batch_size=2, shuffle=False)

net = BaseNet().cuda()
net.train()
batch_id = 0
loss_g = 0
for epoch in range(epochs):
    print(epoch)
    for data in tqdm(data_loader):
        batch_id += 1
        #content_pyr = make_laplace_pyramid(data['content'], 2)
        #style_pyr = make_laplace_pyramid(data['style'], 2)
        content_pyr = data['content']
        style_pyr = data['style']
        #net.train_iter(content_pyr[-1].cuda(), style_pyr[-1].cuda())
        net.train_iter(content_pyr.cuda(), style_pyr.cuda())
        loss_g += net.show_loss()

        if batch_id % 100 == 0:
            net.show_pic(check_image, batch_id)

            print(loss_g/batch_id)

net.save_para(saved_para)