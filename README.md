# RMISP: Bidirectional Mapping of Image Signal
Processing Pipeline
After being processed by the image signal processing (ISP) pipeline in digital cameras, the sRGB images are nonlinear, and thus are not suitable for the computer vision tasks which work best in a linear color space. Therefore, mapping nonlinear sRGB images back to a linear color space is a highly valuable task. To achieve an accurate mapping, this paper proposes a framework based on convolutional neural networks, which models the ISP pipeline in both reverse and forward directions. In particular, for the reverse mapping, a U-net structure is applied to extract features from a given sRGB image, and the extracted features are utilized to estimate the linear and nonlinear transformations in the ISP pipeline. For the forward mapping, the
original sRGB image is used as a guidance to embed the camerastyle information. To incorporate the encoded prior knowledge, affine transformations are employed to modulate the features. Experiments demonstrate that the proposed framework is able to achieve the state-of-the-art performance.
# Requirements
python3 and pytorch 1.8. Tested on Ubuntu 16.04 and Windows 10.
# Using the pretrained model
 - srgb2xyz: python3 srgb2xyz.py --input_img_dir (input srgb image directory) --output_dir(output unprocessing xyz image dir)
 - xyz2srgb: python3 srgb2xyz.py --srgb_img_dir (input srgb image directory) --rec_xyz_img_dir(input reconstructed xyz image dir) --rendered_xyz_img_dir(input rendered xyz image dir) --output_dir(output unprocessing xyz image dir)
# Training on your own dataset
Currently, the paper is under review, the training code will be released after the paper is accepted.

