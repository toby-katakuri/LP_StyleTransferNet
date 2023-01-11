from make_dataset import EdgeEnhanceNet_Dataset
from BaseNet import BaseNet
import torchvision.utils
import torch.utils.data as data
from EdgeEnhanceNet import EdgeEnhanceNet
from tqdm import tqdm
from utils import make_laplace_pyramid, generate_path
import os


### option ###
style = '001'
number = '3'
new_number = 'for_normal'
### path ###
root_c = r'F:\datasets\chinese_content_512'
root_cedge = r'F:\datasets\chinese_content_512_edge'

root_s = os.path.join(r'F:\pyproject\lap_cin\data\style_images', style + '.jpg')
#root_sed = r'F:\datasets\chinese_content_512'
root_sed = os.path.join(r'F:\datasets\lap_cin\basenet', style + '_' + number)
check_image = generate_path(r'F:\pyproject\lap_cin\check_images_enhance', style + '_' + number + new_number)

saved_para = generate_path(r'F:\pyproject\lap_cin\saved_paras_enhance', style + '_' + number + new_number)

saved_loss = generate_path(r'F:\pyproject\lap_cin\saved_loss_wosanet', style + '_' + number + new_number)

epochs = 10
###  ###



dataset = EdgeEnhanceNet_Dataset(root_c, root_s, root_sed, root_cedge)
data_loader = data.DataLoader(dataset, batch_size=1, shuffle=False)

net = EdgeEnhanceNet().cuda()
net.train()
batch_id = 0
loss_g = 0
for epoch in range(epochs):
    batch_id2 = 0
    print(epoch)
    for data in tqdm(data_loader):
        batch_id += 1
        batch_id2 +=1
        content_pyr = make_laplace_pyramid(data['content'], 2)
        net.set_input(content_pyr, data['content'], data['style'], data['stylized'], data['content_edge'])
        net.train_iter()
        #net.train_iter_with_gan(batch_id)

        loss_g += net.show_loss()

        if batch_id == 1 or batch_id % 50 == 0:
            net.show_pic(check_image, batch_id)

            print(loss_g/batch_id)

        if batch_id2 % 513==0 or batch_id2 % 238 == 0:
            net.show_pic(check_image, batch_id)




#net.save_loss(saved_loss)

net.save_para(saved_para)
