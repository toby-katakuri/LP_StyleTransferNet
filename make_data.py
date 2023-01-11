import torchvision
from PIL import Image
from make_dataset import make_iamgelist
from tqdm import tqdm



root = r'E:\my_datasets\chinese_content_all'
image_list = make_iamgelist(root)
trans_raw = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

trans_512 = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop([512, 512],scale=(1.0, 1.0), ratio=(1.0, 1.0))])

trans_256 = torchvision.transforms.Compose([torchvision.transforms.Resize([256, 256])])
num = 0
for img in tqdm(image_list):
    num += 1
    image = Image.open(img).convert('RGB')
    image = trans_raw(image)
    image = trans_512(image)
    torchvision.utils.save_image(image, r'E:\my_datasets\chinese_content_512' + '\{}.jpg'.format(str(num).zfill(5)))
