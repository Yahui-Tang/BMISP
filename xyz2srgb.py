import argparse
from utils import get_dir_filename_list, load_img, save_img
import numpy as np
import torch
from model.coloring_net import ColoringNet


def xyz2srgb():
    with torch.no_grad():
        parser = argparse.ArgumentParser(description="sRGB2XYZ Option")
        parser.add_argument('--srgb_img_dir', type=str, default='./input', help='input srgb img dir')
        parser.add_argument('--srgb_img_type', type=str, default='.jpg', help='extension of input srgb img')

        parser.add_argument('--rec_xyz_img_dir', type=str, default='./output/xyz', help='input reconstructed xyz img dir')
        parser.add_argument('--rec_xyz_img_type', type=str, default='.png', help='extension of the reconstructed xyz img')

        parser.add_argument('--rendered_xyz_img_dir', type=str, default='./output/xyz', help='input rendered xyz img dir')
        parser.add_argument('--rendered_xyz_type', type=str, default='.png', help='extension of the rendered xyz img')

        parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/xyz2srgb.pth', help='checkpoint path')
        parser.add_argument('--output_dir', type=str, default='./output/srgb', help='output dir')

        args = parser.parse_args()

        net = ColoringNet()
        checkpoint = torch.load(args.checkpoint_path)
        weight = checkpoint['weight']
        net.load_state_dict(weight)
        net.cuda()

        img_name_list = get_dir_filename_list(args.srgb_img_dir, type=args.srgb_img_type)

        for img_name in img_name_list:
            srgb_input_path = args.srgb_img_dir + '/' + img_name
            rec_xyz_input_path = args.rec_xyz_img_dir + '/' + img_name[:-4] + args.rec_xyz_img_type
            rendered_xyz_img_path = args.rendered_xyz_img_dir + '/' + img_name[:-4] + args.rendered_xyz_type
            img_out_path = args.output_dir + '/' + img_name[:-4] + '.png'

            srgb_input = load_img(srgb_input_path).cuda()
            rec_xyz_input = load_img(rec_xyz_input_path, depth='16bit').cuda()
            rendered_xyz_input = load_img(rendered_xyz_img_path, depth='16bit').cuda()

            out_img = net(srgb_input, rec_xyz_input, rendered_xyz_input)

            out_img = out_img.detach().cpu()
            print(img_name)
            save_img(img_out_path, out_img, dtype='uint16')


if __name__ == '__main__':
    xyz2srgb()