import torch.utils.data as data
import torch
import torchvision.utils
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np



def make_iamgelist(path):
    image_list = []
    for l in os.listdir(path):
        image_list.append(os.path.join(path,l))
    return sorted(image_list)


def get_transform(resize=None):
    transform = []
    if resize:
        transform += [transforms.Resize(resize)]
    transform += [transforms.ToTensor()]

    return transforms.Compose(transform)


class FirstNet_Dataset(data.Dataset):
    def __init__(self, root_c, root_s, resize=None):
        super(FirstNet_Dataset, self).__init__()

        self.root_c = root_c
        self.root_s = root_s
        self.resize = resize
        # content
        self.content_paths = make_iamgelist(root_c)
        # style
        self.style_paths = make_iamgelist(root_s)

    def __getitem__(self, index):
        # content
        c_path = self.content_paths[index]
        c = Image.open(c_path).convert('RGB')
        c_tensor = get_transform(self.resize)(c)

        # style
        s_path = np.random.choice(self.style_paths,replace=False)
        s = Image.open(s_path).convert('RGB')
        s_tensor = get_transform(self.resize)(s)


        data_dict = {'content': c_tensor, 'style': s_tensor}
        return data_dict

    def __len__(self):
        return len(self.content_paths)


class BaseNet_Dataset(data.Dataset):
    def __init__(self, root_c, root_s, resize=None):
        super(BaseNet_Dataset, self).__init__()

        self.root_c = root_c
        self.resize = resize
        # content
        self.content_paths = make_iamgelist(root_c)
        # style
        s = Image.open(root_s).convert('RGB')
        self.s_tensor = get_transform([512, 512])(s)


    def __getitem__(self, index):
        # content
        c_path = self.content_paths[index]
        c = Image.open(c_path).convert('RGB')
        c_tensor = get_transform(self.resize)(c)


        data_dict = {'content': c_tensor, 'style': self.s_tensor}
        return data_dict

    def __len__(self):
        return len(self.content_paths)


class EnhanceNet_Dataset(data.Dataset):
    def __init__(self, root_c, root_s, root_sed, resize=None):
        super(EnhanceNet_Dataset, self).__init__()

        self.root_c = root_c

        self.resize = resize
        # content
        self.content_paths = make_iamgelist(root_c)
        # style
        s = Image.open(root_s).convert('RGB')
        self.s_tensor = get_transform([512, 512])(s)

        self.root_sed = root_sed
        self.stylized_paths = make_iamgelist(root_sed)


    def __getitem__(self, index):
        # content
        c_path = self.content_paths[index]
        c = Image.open(c_path).convert('RGB')
        c_tensor = get_transform(self.resize)(c)

        #
        sed_path = self.stylized_paths[index]
        sed = Image.open(sed_path).convert('RGB')
        sed_tensor = get_transform(self.resize)(sed)

        data_dict = {'content': c_tensor, 'style': self.s_tensor, 'stylized': sed_tensor}
        return data_dict

    def __len__(self):
        return len(self.content_paths)


class EdgeEnhanceNet_Dataset(data.Dataset):
    def __init__(self, root_c, root_s, root_sed, root_cedge, resize=None):
        super(EdgeEnhanceNet_Dataset, self).__init__()

        self.root_c = root_c

        self.resize = resize
        # content
        self.content_paths = make_iamgelist(root_c)
        # style
        s = Image.open(root_s).convert('RGB')
        self.s_tensor = get_transform([512, 512])(s)
        # stylized
        self.root_sed = root_sed
        self.stylized_paths = make_iamgelist(root_sed)
        # edge
        self.root_cedge = root_cedge
        self.cedge_paths = make_iamgelist(root_cedge)


    def __getitem__(self, index):
        # content
        c_path = self.content_paths[index]
        c = Image.open(c_path).convert('RGB')
        c_tensor = get_transform(self.resize)(c)

        #
        sed_path = self.stylized_paths[index]
        sed = Image.open(sed_path).convert('RGB')
        sed_tensor = get_transform(self.resize)(sed)

        cedge_path = self.cedge_paths[index]
        cedge = Image.open(cedge_path).convert('RGB')
        cedge_tensor = get_transform(self.resize)(cedge)

        data_dict = {'content': c_tensor, 'style': self.s_tensor, 'stylized': sed_tensor, 'content_edge': cedge_tensor}
        return data_dict

    def __len__(self):
        return len(self.content_paths)


