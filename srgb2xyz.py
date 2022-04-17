import torch
import argparse
from utils import save_img, load_img
from utils import get_dir_filename_list
from model.whiting_net import whiting_Net


def srgb2xyz():
    parser = argparse.ArgumentParser(description="sRGB2XYZ Option")
    parser.add_argument('--input_img_dir', type=str, default='./input', help='input srgb img dir')
    parser.add_argument('--img_type', type=str, default='.jpg', help='extension of input img')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/srgb2xyz.pth', help='checkpoint path')
    parser.add_argument('--output_dir', type=str, default='./output/xyz', help='output unprocessing xyz img dir')

    args = parser.parse_args()

    with torch.no_grad():
        net = whiting_Net()
        checkpoint = torch.load(args.checkpoint_path)
        weight = checkpoint['weight']
        net.load_state_dict(weight)
        net.cuda()

        img_name_list = get_dir_filename_list(args.input_img_dir, type=args.img_type)

        for img_name in img_name_list:
            img_input_path = args.input_img_dir + '/' + img_name
            img_out_path = args.output_dir + '/' + img_name[:-4] + '.png'
            input = load_img(img_input_path).cuda()
            out_img, _ = net(input)

            out_img = out_img.detach().cpu()
            print(img_name)
            save_img(img_out_path, out_img, dtype='uint16')


if __name__ == '__main__':
    srgb2xyz()