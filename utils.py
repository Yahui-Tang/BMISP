import numpy as np
import logging
import torch
from torchvision import transforms
import cv2
import os


def from_tensor_to_image(tensor, device='cuda'):
    """ converts tensor to image """
    tensor = torch.squeeze(tensor, dim=0)
    if device == 'cpu':
        image = tensor.data.numpy()
    else:
        image = tensor.cpu().data.numpy()
    # CHW to HWC
    image = image.transpose((1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def outOfGamutClipping(I):
    """ Clips out-of-gamut pixels. """
    I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
    I[I < 0] = 0  # any pixel is below 0, clip it to 0
    return I


def loading_img_as_numpy(path, flag=None, normalize=True):
    img_data = cv2.imread(path, flag)
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    if normalize:
        img_data = cv2.normalize(img_data.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    else:
        if flag == -1:
            img_data = img_data.astype('float32') / 65535.
        else:
            img_data = img_data.astype('float32') / 255.
    return img_data


def save_img(filename, save_img, dtype):
    save_img = from_tensor_to_image(save_img, device='cpu')
    save_img = outOfGamutClipping(save_img)
    if dtype=='uint8':
        save_img = save_img * 255
        cv2.imwrite(filename, save_img.astype(np.uint8))
    elif dtype=='uint16':
        save_img = save_img * 65535
        cv2.imwrite(filename, save_img.astype(np.uint16))


def load_img(img_path, depth='8bit', cut=True):
    trans = transforms.ToTensor()
    if depth == '16bit':
        flag = -1
    else:
        flag = None
    img_data = loading_img_as_numpy(img_path, flag, normalize=False)
    if cut:
        H, W, C = np.shape(img_data)
        h_shift = H % 4
        w_shift = W % 4
        img_data = img_data[:H - h_shift, :W - w_shift, :]
        if h_shift + w_shift > 0:
            logging.info(img_path + ' has been cut')
    img_data_tensor = trans(img_data).unsqueeze(0)
    return img_data_tensor


def get_dir_filename_list(dir, type='.png'):
    files = []
    for _, _, filenames in os.walk(dir):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() == type.lower():
                files.append(filename)
    files.sort()
    return files

