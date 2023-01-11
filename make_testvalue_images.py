from utils import make_laplace_pyramid, generate_path
import os
from PIL import Image
from tqdm import tqdm


# generate all models dir
def generate_all_model_path(styles=8):
    root = r'F:\pyproject\lap_cin\all_model_images'
    model_lists = ['ours', 'Gatys', 'fast', 'Adain', 'WCT', 'SANet']

    for s in range(styles):
        root_styles = generate_path(root, str(s))
        for m in model_lists:

            m_path = generate_path(root_styles, m)



#
def get_ours_images(style=r'\0', target=r'\001_3e2'):
    root_style = r'F:\pyproject\lap_cin\test_images' + target

    root_out = r'F:\pyproject\lap_cin\all_model_images' + style + r'\ours'


    for i in tqdm(range(1, 2451)):

        image_name = str(i) + '_' + 'se.jpg'
        image_name = os.path.join(root_style, image_name)
        image = Image.open(image_name)

        new_image_name = '{}.jpg'.format(str(i).zfill(5))
        new_image_name = os.path.join(root_out, new_image_name)
        image.save(new_image_name)


#get_ours_images(r'\9', r'\026_1e1')







