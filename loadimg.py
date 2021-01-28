from PIL import Image
import requests
from io import BytesIO
import numpy as np
import base64


def write_img(file_url, imgstr):
    d = base64.b64decode(imgstr)
    with open(file_url , 'wb') as f:
        f.write(d)
    

def load_save_img(img_file_pth):
    if img_file_pth[:4]=="http":
        response = requests.get(img_file_pth)
        img_rgb = Image.open(BytesIO(response.content))
    else:
        img_rgb = Image.open(img_file_pth)   
    img_rgb.save("tmp.png")
    return np.array(img_rgb)
    
    
