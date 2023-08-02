import os

from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from matplotlib import pyplot as plt
import mmcv
from collections import Counter
from PIL import Image
import numpy as np
from tqdm import tqdm


config_file = r"local_configs\segformer\B2\segformer.b2.512x512.ade.160k.py"
checkpoint_file = r"F:\002Segformer\SegFormer-master\tools\rs128\sim_dif_alt_bld.pth"
# checkpoint_file = r"F:\002Segformer\SegFormer-master\tools\output_weights\02wheat\wheat_sim2real.pth"


model = init_segmentor(config_file, checkpoint_file, device='cuda:0')


#### format  must be .png  ####
img_root = r"F:\000DLdataset\2023danyang_rice\0718\rgb/"
save_mask_root = r"F:\000DLdataset\2023danyang_rice\0718\test/"

if not os.path.exists(save_mask_root):
    os.mkdir(save_mask_root)
img_names = os.listdir(img_root)
for img_name in tqdm(img_names):
    # test a single image
    img = img_root + img_name
    result = inference_segmentor(model, img)[0]
    img = Image.fromarray(np.uint8(result*255))
    img.save(save_mask_root + img_name)
