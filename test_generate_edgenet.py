from make_dataset import EdgeEnhanceNet_Dataset
from BaseNet import BaseNet
import torchvision.utils
import torch.utils.data as data
from EdgeEnhanceNet import EdgeEnhanceNet_Test
from tqdm import tqdm
from utils import make_laplace_pyramid, generate_path
import os
import time

### option ###
style = '001'
number = '3'
new_number = 'for_loss_all'
### path ###
root_c = r'F:\datasets\chinese_content_512'
root_cedge = r'F:\datasets\chinese_content_512_edge'
root_s = os.path.join(r'F:\pyproject\lap_cin\data\style_images', style + '.jpg')
root_sed = os.path.join(r'F:\datasets\lap_cin\basenet', style + '_' + number)
test_images = generate_path(r'F:\pyproject\lap_cin\test_images', style + '_' + number + new_number)

saved_para = generate_path(r'F:\pyproject\lap_cin\saved_paras_enhance', style + '_' + number + new_number)
epochs = 1
###  ###



dataset = EdgeEnhanceNet_Dataset(root_c, root_s, root_sed, root_cedge)
data_loader = data.DataLoader(dataset, batch_size=1, shuffle=False)

net = EdgeEnhanceNet_Test(saved_para).cuda()
net.eval()
batch_id = 0
loss_g = 0
for epoch in range(epochs):
    print(epoch)
    a = time.time()
    for data in tqdm(data_loader):
        batch_id += 1
        content_pyr = make_laplace_pyramid(data['content'], 2)
        net.set_input(content_pyr, data['content'], data['style'], data['stylized'], data['content_edge'])
        net.test_image(test_images, batch_id)

        if batch_id == 200:
            break

    b = time.time()
    print((b-a)/200)
        # if batch_id == 673:
        #     content_pyr = make_laplace_pyramid(data['content'], 2)
        #     net.set_input(content_pyr, data['content'], data['style'], data['stylized'], data['content_edge'])
        #     net.test_1image(batch_id)




