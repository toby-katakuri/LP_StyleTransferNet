import torchvision
from skimage.filters import sobel, roberts, scharr,  try_all_threshold, threshold_yen, threshold_otsu
from skimage.color import rgb2gray
import matplotlib
from PIL import Image
import skimage
import matplotlib.pyplot as plt


c = torchvision.transforms.Compose([torchvision.transforms.ToTensor()
                                   ])

for i in range(1, 2451):
    name = r'F:\datasets\chinese_content_512' + '\{}.jpg'.format(str(i).zfill(5))
    image = skimage.io.imread(name)
    image = rgb2gray(image)

    image = sobel(image)


    image = c(image)
    name2 = r'F:\datasets\chinese_content_512_newedge' + '\{}.jpg'.format(str(i).zfill(5))
    torchvision.utils.save_image(image, name2)