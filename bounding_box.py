import argparse
import os
from cp_dataset import CPDataset, CPDataLoader
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
import numpy as np
from utils import *

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_ids', default="0")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=2)
    parser.add_argument('--fp16', action='store_true', help='use amp')
    # Cuda availability
    parser.add_argument('--cuda',default=True, help='cuda or cpu')

    parser.add_argument("--dataroot", default="./data/")
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--data_list", default="test_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=768)
    parser.add_argument("--fine_height", type=int, default=1024)
    parser.add_argument("--radius", type=int, default=20)
    parser.add_argument("--grid_size", type=int, default=5)

    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--tocg_checkpoint', type=str, default='./model/mtviton.pth', help='condition generator checkpoint')
    parser.add_argument('--gen_checkpoint', type=str, default='', help='gen checkpoint')
    parser.add_argument('--dis_checkpoint', type=str, default='', help='dis checkpoint')

    parser.add_argument("--tensorboard_count", type=int, default=100)
    parser.add_argument("--display_count", type=int, default=100)
    parser.add_argument("--save_count", type=int, default=10000)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    
    # test
    parser.add_argument("--lpips_count", type=int, default=1000)
    parser.add_argument("--test_datasetting", default="paired")
    parser.add_argument("--test_dataroot", default="./data/")
    parser.add_argument("--test_data_list", default="test_pairs.txt")

    # Hyper-parameters
    parser.add_argument('--G_lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--D_lr', type=float, default=0.0004, help='initial learning rate for adam')

    # SEAN-related hyper-parameters
    parser.add_argument('--GMM_const', type=float, default=None, help='constraint for GMM module')
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of input label classes without unknown class')
    parser.add_argument('--gen_semantic_nc', type=int, default=7, help='# of input label classes without unknown class')
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance', help='instance normalization or batch normalization')
    parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most',
                    help='If \'more\', add upsampling layer between the two middle resnet blocks. '
                            'If \'most\', also add one more (upsampling + resnet) layer at the end of the generator.')
    parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

    parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
    parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
    parser.add_argument('--lambda_l1', type=float, default=1.0, help='weight for feature matching loss')
    parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
    parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
    
    # D
    parser.add_argument('--n_layers_D', type=int, default=3, help='# layers in each discriminator')
    parser.add_argument('--netD_subarch', type=str, default='n_layer', help='architecture of each discriminator')
    parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to be used in multiscale')
    
    # Training
    parser.add_argument('--GT', action='store_true')
    parser.add_argument('--occlusion', action='store_true')
    # tocg
    # network
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
    parser.add_argument("--clothmask_composition", type=str, choices=['no_composition', 'detach', 'warp_grad'], default='warp_grad')
    # visualize
    parser.add_argument("--num_test_visualize", type=int, default=3)

    opt = parser.parse_args()

    # set gpu ids
    # str_ids = opt.gpu_ids.split(',')
    # opt.gpu_ids = []
    # for str_id in str_ids:
    #     id = int(str_id)
    #     if id >= 0:
    #         opt.gpu_ids.append(id)
    # if len(opt.gpu_ids) > 0:
    #     torch.cuda.set_device(opt.gpu_ids[0])

    assert len(opt.gpu_ids) == 0 or opt.batch_size % len(opt.gpu_ids) == 0, \
        "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
        % (opt.batch_size, len(opt.gpu_ids))

    return opt


def get_bounding_box(segmentation):
    # Get the height and width of the segmentation
    height, width = segmentation.shape
    
    # Get the row and column indices where the segmentation is nonzero
    rows, cols = torch.nonzero(segmentation, as_tuple=True)
    if len(rows) == 0:
        return torch.zeros(1, 4)
    
    # Calculate the bounding box
    top = rows.min().item()
    left = cols.min().item()
    height = rows.max().item() - top + 1
    width = cols.max().item() - left + 1
    
    return torch.tensor([top, left, height, width])

def masks_to_boxes(masks):
    """
    Convert binary masks to bounding boxes.
    Args:
        masks: torch.Tensor of shape (N, H, W) representing N binary masks of height H and width W.
    Returns:
        boxes: torch.Tensor of shape (N, 4) representing N bounding boxes in (top, left, height, width) format.
    """
    N, H, W = masks.shape
    boxes = torch.zeros((N, 4))
    for i in range(N):
        boxes[i] = get_bounding_box(masks[i])
    return boxes

def add_bounding_box(bbox, seg, color):
    top, left, height, width = bbox
    seg_with_box = seg.clone()  # Make a copy of the segmentation tensor
    
    # Add the bounding box to the segmentation tensor
    seg_with_box[top:top+height, left:left+1] = color  # Left edge
    seg_with_box[top:top+height, left+width-1:left+width] = color  # Right edge
    seg_with_box[top:top+1, left:left+width] = color  # Top edge
    seg_with_box[top+height-1:top+height, left:left+width] = color  # Bottom edge
    
    return seg_with_box

def add_bounding_box_img(bbox, img, color):
    top, left, height, width = bbox
    img_with_box = img.clone()  # Make a copy of the segmentation tensor
    
    # Add the bounding box to the segmentation tensor
    img_with_box[:, top:top+height, left:left+1] = color  # Left edge
    img_with_box[:, top:top+height, left+width-1:left+width] = color  # Right edge
    img_with_box[:, top:top+1, left:left+width] = color  # Top edge
    img_with_box[:, top+height-1:top+height, left:left+width] = color  # Bottom edge
    
    return img_with_box

def generate_parse(parse_GT, opt):
    fake_parse = parse_GT.argmax(dim=1)[:, None]

    old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt.fine_height, opt.fine_width).zero_()
    old_parse.scatter_(1, fake_parse, 1.0)

    labels = {
        0:  ['background',  [0]],
        1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
        2:  ['upper',       [3]],
        3:  ['hair',        [1]],
        4:  ['left_arm',    [5]],
        5:  ['right_arm',   [6]],
        6:  ['noise',       [12]]
    }
    parse = torch.FloatTensor(fake_parse.size(0), 7, opt.fine_height, opt.fine_width).zero_()
    for i in range(len(labels)):
        for label in labels[i][1]:
            parse[:, i] += old_parse[:, label]

    return parse

def split_body_parts(parse):
    batch_size, num_masks, _, _ = parse.shape
    boxes = torch.FloatTensor(batch_size, num_masks, 4).zero_()
    for batch in range(batch_size):
        masks = parse[batch, :, :, :]
        boxes[batch, :, :] = masks_to_boxes(masks)

    return boxes

def main():
    print("check opt")
    opt = get_opt()
    print(opt)
    print("Start to train!")

    train_dataset = CPDataset(opt)
    train_loader = CPDataLoader(opt, train_dataset)
    c=0
    labels = [ {"width": 0, "height": 0},    {"width": 0, "height": 0},    {"width": 0, "height": 0},{"width": 0, "height": 0},{"width": 0, "height": 0},{"width": 0, "height": 0},{"width": 0, "height": 0}]
    for inputs in train_loader.data_loader: 
            img = inputs['image']
            parse_GT = inputs['parse']
            parse = generate_parse(parse_GT, opt)
            shape = parse.shape
            masks = parse[0, :, :, :]
            boxes = masks_to_boxes(masks)
            seg = parse[0, 0, :, :]
            for i in range(boxes.shape[0]):
                seg = add_bounding_box(boxes[i].type(torch.int64), seg, 1)
                # print(split_body_parts(seg))
                img[0] = add_bounding_box_img(boxes[i].type(torch.int64), img[0], 0)
            parse[0, 0, :, :] = seg
            # save_image(img[0] / 2 + 0.5, f'./output/test/image_new{c}.png')
            boxes = split_body_parts(parse)
            for i in range(len(boxes[0])):
                print(boxes[0][i][2].item())
                if(boxes[0][i][2].item()>labels[i]["width"]):
                    labels[i]["width"]=boxes[0][i][2].item()
                if(boxes[0][i][3].item()>labels[i]["height"]):
                    labels[i]["height"]=boxes[0][i][3].item()
            c=c+1
    print(labels)

    # inputs = train_loader.next_batch()
    # img = inputs['image']
    # parse_GT = inputs['parse']

    # parse = generate_parse(parse_GT, opt)
    # shape = parse.shape
    # c=0


    # save_image(img[0] / 2 + 0.5, f'./output/torch/image.png')
    # save_image(visualize_segmap(parse.cpu(), batch=0), f'./output/torch/tensor.png')

    # for channel in range(shape[1]):
    #     x = parse[0, channel, :, :]
    #     save_image(x, f'./output/torch/tensor{channel}.png')
    
    # masks = parse[0, :, :, :]
    # print(masks.shape)
    # boxes = masks_to_boxes(masks)
    # print("box")
    # print(boxes)

    # seg = parse[0, 0, :, :]
    # for i in range(boxes.shape[0]):
    #     seg = add_bounding_box(boxes[i].type(torch.int64), seg, 1)
    #     img[0] = add_bounding_box_img(boxes[i].type(torch.int64), img[0], 0)
    # parse[0, 0, :, :] = seg

    # save_image(visualize_segmap(parse.cpu(), batch=0), f'./output/torch/tensor_new.png')
    # save_image(img[0] / 2 + 0.5, f'./output/torch/image_new.png')

    # boxes = split_body_parts(parse)
    # print(boxes.shape)
    # print(boxes)

if __name__ == "__main__":
    main()