import torch
import glob
from img_prepro import img_segmentation
from model import autoencoder
from utils import get_key
import pickle
import numpy as np


with open("background_d.pkl","rb") as f:
    background_d = pickle.load(f)

key = get_key()

model = autoencoder(n_class=len(key))#.cuda()
source_num_epochs = 13400
model.load_state_dict(torch.load(f'./conv_ae_{str(source_num_epochs-1).zfill(6)}.pth',map_location=torch.device('cpu')))

def _detect(imgarrs):
    if type(imgarrs)==list:
        imgarrs = torch.stack([ torch.FloatTensor(imgarr) for imgarr in imgarrs], 0)
    imgs = imgarrs.unsqueeze(1)#.cuda()
    rc_imgs, pred_labels = model(imgs)
    pred_labels = pred_labels.cpu().data.numpy()
    pred_labels = np.argmax(pred_labels, 1)
    return "".join([key[l]  for l in pred_labels])

def detect(fname="tmp.png"):
    img_i_list = img_segmentation(fname, background_d, show_main=0)
    result_txt = _detect(img_i_list )
    return result_txt

