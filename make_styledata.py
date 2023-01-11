from PIL import Image
import torchvision




s = r'F:\pyproject\lap_cin\data\style_images\026.jpeg'

trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.RandomResizedCrop(512, (1.5, 1.5))])

image = Image.open(s)
image = trans(image)
torchvision.utils.save_image(image, 'new_style_imaages/5.jpg')