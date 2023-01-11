from make_dataset import EnhanceNet_Dataset
from BaseNet import BaseNet
import torchvision.utils
import torch.utils.data as data
from EnhanceNet import EnhanceNet_Test
from tqdm import tqdm
from utils import make_laplace_pyramid, generate_path
import os


### option ###
style = '026'
number = '1'
new_number = 'wosanet2'
### path ###
root_c = r'F:\datasets\chinese_content_512'

root_s = os.path.join(r'F:\pyproject\lap_cin\data\style_images', style + '.jpg')
root_sed = os.path.join(r'F:\datasets\lap_cin\basenet', style + '_' + number)
test_images = generate_path(r'F:\pyproject\lap_cin\test_images', style + '_' + number + new_number)

saved_para = generate_path(r'F:\pyproject\lap_cin\saved_paras_enhance', style + '_' + number + new_number)
epochs = 1
###  ###



dataset = EnhanceNet_Dataset(root_c, root_s, root_sed)
data_loader = data.DataLoader(dataset, batch_size=1, shuffle=False)

net = EnhanceNet_Test(saved_para).cuda()
net.eval()
batch_id = 0
loss_g = 0
for epoch in range(epochs):
    print(epoch)
    for data in tqdm(data_loader):
        batch_id += 1
        content_pyr = make_laplace_pyramid(data['content'], 2)
        net.set_input(content_pyr, data['content'], data['style'], data['stylized'])
        net.test_image(test_images, batch_id)




