from make_dataset import BaseNet_Dataset
from BaseNet import BaseNet
import torchvision.utils
import torch.utils.data as data
from BaseNet import BaseNet
from tqdm import tqdm
from utils import make_laplace_pyramid, generate_path
import os


### option ###
style = '002'
number = '3withbase'
### path ###
root_c = r'F:\datasets\chinese_content_512'
root_s = os.path.join(r'F:\pyproject\lap_cin\data\style_images', style + '.jpg')
para_path = generate_path(r'F:\pyproject\lap_cin\saved_paras', style + '_' + number)
generate_image = generate_path(r'F:\datasets\lap_cin\basenet', style + '_' + number)


###  ###

import torch.nn.functional as F

dataset = BaseNet_Dataset(root_c, root_s)
data_loader = data.DataLoader(dataset, batch_size=1, shuffle=False)

net = BaseNet().cuda()
net.eval()

batch_id = 0
for data in tqdm(data_loader):
    batch_id += 1
    content = F.interpolate(data['content'], [512, 512], mode='bicubic', align_corners=False)
    style = F.interpolate(data['style'], [512, 512], mode='bicubic', align_corners=False)
    net.test_image(content.cuda(), style.cuda(), generate_image, batch_id, para_path)


