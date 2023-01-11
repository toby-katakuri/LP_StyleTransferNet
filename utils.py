import torch.nn.functional as F
import os


def tensor_resample(tensor, dst_size, mode='bicubic'):
    return F.interpolate(tensor, dst_size, mode=mode, align_corners=False)

def laplacian(x):
    # x - upsample(downsample(x))
    return x - tensor_resample(tensor_resample(x, [x.shape[2] // 2, x.shape[3] // 2]), [x.shape[2], x.shape[3]])



def make_laplace_pyramid(x, levels):
    pyramid = []
    current = x
    for i in range(levels):
        pyramid.append(laplacian(current))
        current = tensor_resample(current, (max(current.shape[2] // 2, 1), max(current.shape[3] // 2, 1)))
    pyramid.append(current)
    return pyramid


def generate_path(path_name, model_name):
    path = os.path.join(path_name, model_name)
    if not os.path.exists(path):
        os.mkdir(path)

    return path