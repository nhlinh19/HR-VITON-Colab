import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import make_grid as make_image_grid

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from cp_dataset_test import CPDatasetTest
from networks import ConditionGenerator, VGGLoss, load_checkpoint, save_checkpoint, make_grid
from network_generator import SPADEGenerator, MultiscaleDiscriminator, GANLoss

from sync_batchnorm import DataParallelWithCallback
from tensorboardX import SummaryWriter
from utils import create_network, visualize_segmap
import sys
from tqdm import tqdm

import numpy as np
from torch.utils.data import Subset
from torchvision.transforms import transforms
import eval_models as models
import torchgeometry as tgm
from bounding_box import find_bouding_box, crop_boxes_batch

from PIL import Image
from torchvision.models.inception import inception_v3
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim

def remove_overlap(seg_out, warped_cm):
    
    assert len(warped_cm.shape) == 4
    
    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True) * warped_cm
    return warped_cm

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--gpu_ids', default="0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('--fp16', action='store_true', help='use amp')
    # Cuda availability
    parser.add_argument('--cuda',default=True, help='cuda or cpu')

    parser.add_argument("--dataroot", default="./data/")
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--data_list", default="test_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=20)
    parser.add_argument("--grid_size", type=int, default=5)

    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--tocg_checkpoint', type=str, default='./model/mtviton.pth', help='condition generator checkpoint')
    parser.add_argument('--gen_checkpoint', type=str, default='', help='gen checkpoint')
    parser.add_argument('--dis_checkpoint', type=str, default='', help='dis checkpoint')

    parser.add_argument("--tensorboard_count", type=int, default=10)
    parser.add_argument("--display_count", type=int, default=100)
    parser.add_argument("--save_count", type=int, default=100)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    
    # test
    parser.add_argument("--lpips_count", type=int, default=100)
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
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer') # 64
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer') # 64
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='more', # most
                    help='If \'more\', add upsampling layer between the two middle resnet blocks. '
                            'If \'most\', also add one more (upsampling + resnet) layer at the end of the generator.')
    parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

    parser.add_argument('--no_L1_loss', action='store_true', help='if specified, do *not* use L1 loss')
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

    # bounding box
    parser.add_argument("--bbox_max_size", default=[[0.3520238681102362, 0.3265102116141732], [0.4945308655265748, 0.5477464730971129], [0.29374154158464566, 0.46564089156824146], [0.33411509596456695, 0.15164528994422571], [0.35114246278297245, 0.14647924868766404]])
    parser.add_argument("--add_body_loss", default=False) 
    parser.add_argument("--num_evaluate", default=50)

    opt = parser.parse_args()

    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    assert len(opt.gpu_ids) == 0 or opt.batch_size % len(opt.gpu_ids) == 0, \
        "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
        % (opt.batch_size, len(opt.gpu_ids))

    return opt

def train(opt, train_loader, test_loader, test_vis_loader, board, tocg, generator, discriminator, model):
    """
        Train Generator
    """
    # bbox_max_size
    bbox_max_size = torch.tensor(opt.bbox_max_size) * torch.tensor([opt.fine_height, opt.fine_width])
    bbox_max_size = torch.round(bbox_max_size).type(torch.int64)
    print(bbox_max_size)

    # discriminator_label
    if opt.add_body_loss:
        discriminator_body_parts = []
        for i in range(5):
            discriminator_body_parts.append(create_network(MultiscaleDiscriminator, opt, 0))
            discriminator_body_parts[i].train()

    # Model
    if not opt.GT:
        print("--Cuda for tocg")
        tocg.cuda()
        tocg.eval()
        print("--Eval for tocg")
    generator.train()
    discriminator.train()
    model.eval()
    print("--Model")
    # criterion
    if opt.fp16:
        criterionGAN = GANLoss('hinge', tensor=torch.cuda.HalfTensor)
    else:
        criterionGAN = GANLoss('hinge', tensor=torch.cuda.FloatTensor)
    criterionL1 = nn.L1Loss()
    criterionFeat = nn.L1Loss()
    print("--Finish L1 loss")
    criterionVGG = VGGLoss(opt)
    print("--Finish loss")
    # optimizer
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=opt.G_lr, betas=(0, 0.9))
    scheduler_gen = torch.optim.lr_scheduler.LambdaLR(optimizer_gen, lr_lambda=lambda step: 1.0 -
            max(0, step * 1000 + opt.load_step - opt.keep_step) / float(opt.decay_step + 1))

    optimizer_dis_parameters = list(discriminator.parameters())
    if opt.add_body_loss:
        for i in range(5):
            optimizer_dis_parameters += list(discriminator_body_parts[i].parameters())
    optimizer_dis = torch.optim.Adam(optimizer_dis_parameters, lr=opt.D_lr, betas=(0, 0.9))
    
    scheduler_dis = torch.optim.lr_scheduler.LambdaLR(optimizer_dis, lr_lambda=lambda step: 1.0 -
            max(0, step * 1000 + opt.load_step - opt.keep_step) / float(opt.decay_step + 1))
    print("--Finish optimizer")
    if opt.fp16:
        if not opt.GT:
            from apex import amp
            [tocg, generator, discriminator], [optimizer_gen, optimizer_dis] = amp.initialize(
                [tocg, generator, discriminator], [optimizer_gen, optimizer_dis], opt_level='O1', num_losses=2)
        else:
            from apex import amp
            [generator, discriminator], [optimizer_gen, optimizer_dis] = amp.initialize(
                [generator, discriminator], [optimizer_gen, optimizer_dis], opt_level='O1', num_losses=2)
    print("--check parallel")
    if len(opt.gpu_ids) > 0:
        print("--start parallel")
        if not opt.GT:
            tocg = DataParallelWithCallback(tocg, device_ids=opt.gpu_ids)
        generator = DataParallelWithCallback(generator, device_ids=opt.gpu_ids)
        discriminator = DataParallelWithCallback(discriminator, device_ids=opt.gpu_ids)
        criterionGAN = DataParallelWithCallback(criterionGAN, device_ids=opt.gpu_ids)
        criterionL1 = DataParallelWithCallback(criterionL1, device_ids=opt.gpu_ids)
        criterionFeat = DataParallelWithCallback(criterionFeat, device_ids=opt.gpu_ids)
        criterionVGG = DataParallelWithCallback(criterionVGG, device_ids=opt.gpu_ids)
        
    upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss = gauss.cuda()
    print("--Start training")
    for step in tqdm(range(opt.load_step, opt.keep_step + opt.decay_step)):
        iter_start_time = time.time()
        print("start load data")
        inputs = train_loader.next_batch()
        print("done load data")
        # input
        agnostic = inputs['agnostic'].cuda()
        # print(f"--agnostic: {agnostic.device}")
        parse_GT = inputs['parse'].cuda()
        # print(f"--parse: {agnostic.device}")
        pose = inputs['densepose'].cuda()
        # print(f"--densepose: {agnostic.device}")
        parse_cloth = inputs['parse_cloth'].cuda()
        parse_agnostic = inputs['parse_agnostic'].cuda()
        pcm = inputs['pcm'].cuda()
        cm = inputs['cloth_mask']['paired'].cuda()
        c_paired = inputs['cloth']['paired'].cuda()
        # print(f"--c_paired: {c_paired.device}")
        # target
        im = inputs['image'].cuda()
        print("end load data")
        with torch.no_grad():
            if not opt.GT:
                print("tocg")
                # Warping Cloth
                # down
                pre_clothes_mask_down = F.interpolate(cm, size=(256, 192), mode='nearest')
                input_parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='nearest')
                clothes_down = F.interpolate(c_paired, size=(256, 192), mode='bilinear')
                densepose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
                
                # multi-task inputs
                input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
                input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1)

                # forward
                flow_list, fake_segmap, _, warped_clothmask_paired = tocg(opt, input1, input2)
                
                # warped cloth mask one hot 
                warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
                
                if opt.clothmask_composition != 'no_composition':
                    if opt.clothmask_composition == 'detach':
                        cloth_mask = torch.ones_like(fake_segmap)
                        cloth_mask[:,3:4, :, :] = warped_cm_onehot
                        fake_segmap = fake_segmap * cloth_mask
                        
                    if opt.clothmask_composition == 'warp_grad':
                        cloth_mask = torch.ones_like(fake_segmap)
                        cloth_mask[:,3:4, :, :] = warped_clothmask_paired
                        fake_segmap = fake_segmap * cloth_mask
                        
                # warped cloth
                N, _, iH, iW = c_paired.shape
                grid = make_grid(N, iH, iW,opt)
                flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
                flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)], 3)
                warped_grid = grid + flow_norm
                warped_cloth_paired = F.grid_sample(c_paired, warped_grid, padding_mode='border').detach()
                warped_clothmask = F.grid_sample(cm, warped_grid, padding_mode='border')


                # flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), scale_factor=2, mode=upsample).permute(0, 2, 3, 1)
                # flow_norm = torch.cat([flow[:, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((iH/2 - 1.0) / 2.0)], 3)
                # warped_input1 = F.grid_sample(input1, flow_norm + grid, padding_mode='border')
                
                # x = self.out_layer(torch.cat([x, input2, warped_input1], 1))

                # warped_c = warped_input1[:, :-1, :, :]
                # warped_cm = warped_input1[:, -1:, :, :]

                # make generator input parse map
                fake_parse_gauss = gauss(F.interpolate(fake_segmap, size=(iH, iW), mode='bilinear'))
                fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]

                # occlusion
                if opt.occlusion:
                    warped_clothmask = remove_overlap(F.softmax(fake_parse_gauss, dim=1), warped_clothmask)
                    warped_cloth_paired = warped_cloth_paired * warped_clothmask + torch.ones_like(warped_cloth_paired) * (1-warped_clothmask)
                    warped_cloth_paired = warped_cloth_paired.detach()
                # region_mask = parse[:, 2:3] - warped_cm
                # region_mask[region_mask < 0.0] = 0.0
                # parse_rn = torch.cat((parse, region_mask), dim=1)
                # parse_rn[:, 2:3] -= region_mask
            else:
                # parse pre-process
                fake_parse = parse_GT.argmax(dim=1)[:, None]
                warped_cloth_paired = parse_cloth
                
            old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt.fine_height, opt.fine_width).zero_().cuda()
            old_parse.scatter_(1, fake_parse, 1.0)

            # parse 
            labels = {
                0:  ['background',  [0]],
                1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                2:  ['upper',       [3]],
                3:  ['hair',        [1]],
                4:  ['left_arm',    [5]],
                5:  ['right_arm',   [6]],
                6:  ['noise',       [12]]
            }
            parse = torch.FloatTensor(fake_parse.size(0), 7, opt.fine_height, opt.fine_width).zero_().cuda()
            for i in range(len(labels)):
                for label in labels[i][1]:
                    parse[:, i] += old_parse[:, label]
                    
            parse = parse.detach() # [batch, num_labels=5, height, width])

            # body_parts_parse
            body_parts_labels = {
                0:  ['face_hair',   [1, 2]],
                1:  ['upper',       [3]],
                2:  ['bottom',      [4, 7, 8, 9, 10, 11]],
                3:  ['left_arm',    [5]],
                4:  ['right_arm',   [6]],
            }

            if opt.add_body_loss:
                body_parts_parse = torch.FloatTensor(fake_parse.size(0), 5, opt.fine_height, opt.fine_width).zero_().cuda()
                for i in range(len(body_parts_labels)):
                    for label in body_parts_labels[i][1]:
                        body_parts_parse[:, i] += old_parse[:, label]

                body_parts_parse.detach()
            print("--Start generater")
        # --------------------------------------------------------------------------------------------------------------
        #                                              Train the generator
        # --------------------------------------------------------------------------------------------------------------
        # print("agnostic shape: ", agnostic.shape) # [batch, 3, height, width]
        # print("pose shape: ", pose.shape) # [batch, 3, height, width]
        # print("warped_cloth_paired shape: ", warped_cloth_paired.shape) # [batch, 3, height, width]

        output_paired = generator(torch.cat((agnostic, pose, warped_cloth_paired), dim=1), parse)
        # print("output_paired shape: ", output_paired.shape) # [batch, 3, height, width]
        # print("parse shape: ", parse.shape) # [batch, 3, height, width]
        # print("im shape: ", im.shape) # [batch, 3, height, width]
        print("--Done generate")
        fake_concat = torch.cat((parse, output_paired), dim=1) # [batch, 3 + num_labels, height, width]
        real_concat = torch.cat((parse, im), dim=1) # [batch, 3 + num_labels, height, width]
        pred = discriminator(torch.cat((fake_concat, real_concat), dim=0))
        # print(f"--Predict shape: {len(pred), len(pred[0])}")
        # for i in range(len(pred)):
        #     for j in range(len(pred[i])):
        #         print(f"--Predict[{i}][{j}] shape: {pred[i][j].shape}")

        # body_parts
        if opt.add_body_loss:
            body_parts_fake = crop_boxes_batch(output_paired, body_parts_parse, bbox_max_size)
            body_parts_real = crop_boxes_batch(im, body_parts_parse, bbox_max_size)
            body_parts_pred = []
            # body_parts_pred_fake = [] # [5][2, 4] tensor of feature layer [2*batch, output_channel, height_feature, width_feature]
            # body_parts_pred_real = []
            for i in range(5):
                # body_parts_pred_fake.append(
                #     discriminator_body_parts[i](body_parts_fake[i])
                # )
                # body_parts_pred_real.append(
                #     discriminator_body_parts[i](body_parts_real[i])
                # )
                body_parts_pred.append(discriminator_body_parts[i](torch.cat((body_parts_fake[i], body_parts_real[i]), dim=0)))

        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            pred_fake = []
            pred_real = []
            for p in pred:
                pred_fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                pred_real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            pred_fake = pred[:pred.size(0) // 2]
            pred_real = pred[pred.size(0) // 2:]

        if opt.add_body_loss:
            body_parts_pred_fake = [] # [5][2, 4] tensor of feature layer [batch, output_channel, height_feature, width_feature]
            body_parts_pred_real = [] # [5][2, 4] tensor of feature layer [batch, output_channel, height_feature, width_feature]
            for i in range(5):
                if type(body_parts_pred[i]) == list:
                    pred_fake_body = []
                    pred_real_body = []
                    for p in body_parts_pred[i]:
                        pred_fake_body.append([tensor[:tensor.size(0) // 2] for tensor in p])
                        pred_real_body.append([tensor[tensor.size(0) // 2:] for tensor in p])
                else:
                    pred_fake_body = body_parts_pred[:body_parts_pred.size(0) // 2]
                    pred_real_body = body_parts_pred[body_parts_pred.size(0) // 2:]

                body_parts_pred_fake.append(pred_fake_body)
                body_parts_pred_real.append(pred_real_body)

        G_losses = {}
        G_losses['GAN'] = criterionGAN(pred_fake, True, for_discriminator=False)

        if opt.add_body_loss:
            G_body_parts_losses = {}
            for i in range(5):
                G_body_parts_losses[f'{i}'] = criterionGAN(body_parts_pred_fake[i], True, for_discriminator=False)
            G_losses['GAN_Body_Parts'] = sum(G_body_parts_losses.values()).mean()


        if not opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = torch.cuda.FloatTensor(len(opt.gpu_ids)).zero_()
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not opt.no_vgg_loss:
            G_losses['VGG'] = criterionVGG(output_paired, im) * opt.lambda_vgg

        if not opt.no_L1_loss:
            G_losses['L1'] = criterionL1(output_paired, im) * opt.lambda_l1
            G_L1_body_losses = {} 
            for k in range(5):
                G_L1_body_losses[f'{k}'] = 0
                for i in range(len(body_parts_pred_fake[0])):
                    for j in range(len(body_parts_pred_fake[0][i])):
                        G_L1_body_losses[f'{k}'] += criterionL1(body_parts_pred_fake[k][i][j], body_parts_pred_real[k][i][j]) 
                G_L1_body_losses[f'{k}'] *= opt.lambda_l1
            G_losses['L1'] += sum(G_L1_body_losses.values()).mean()

        loss_gen = sum(G_losses.values()).mean()
        print(G_losses)
        print("Loss gen:", loss_gen)
        # print("Loss GAN:", G_losses['GAN'])
        # print("Loss GAN_Feat:", G_losses['GAN_Feat'])
        # print("Loss VGG:", G_losses['VGG'])
        # print("Loss GAN_Body_Parts:", G_losses['GAN_Body_Parts'])

        optimizer_gen.zero_grad()
        if opt.fp16:
            with amp.scale_loss(loss_gen, optimizer_gen, loss_id=0) as loss_gen_scaled:
                loss_gen_scaled.backward()
        else:
            loss_gen.backward()
        optimizer_gen.step()
        print("Finish loss gen")

        # --------------------------------------------------------------------------------------------------------------
        #                                            Train the discriminator
        # --------------------------------------------------------------------------------------------------------------
        with torch.no_grad():
            output = generator(torch.cat((agnostic, pose, warped_cloth_paired), dim=1), parse)
            output = output.detach()
            output.requires_grad_()

        fake_concat = torch.cat((parse, output), dim=1)
        real_concat = torch.cat((parse, im), dim=1)
        pred = discriminator(torch.cat((fake_concat, real_concat), dim=0))

        if opt.add_body_loss:
            body_parts_fake = crop_boxes_batch(output, body_parts_parse, bbox_max_size) # [5][batch, 3, height_body_part, width_body_part]
            body_parts_real = crop_boxes_batch(im, body_parts_parse, bbox_max_size) # [5][batch, 3, height_body_part, width_body_part]
            body_parts_pred = []
            # body_parts_pred_fake = [] # [5][2, 4] tensor of feature layer [2*batch, output_channel, height_feature, width_feature]
            # body_parts_pred_real = []
            for i in range(5):
                # body_parts_pred_fake.append(
                #     discriminator_body_parts[i](body_parts_fake[i])
                # )
                # body_parts_pred_real.append(
                #     discriminator_body_parts[i](body_parts_real[i])
                # )
                body_parts_pred.append(discriminator_body_parts[i](torch.cat((body_parts_fake[i], body_parts_real[i]), dim=0)))

        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            pred_fake = []
            pred_real = []
            for p in pred:
                pred_fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                pred_real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            pred_fake = pred[:pred.size(0) // 2]
            pred_real = pred[pred.size(0) // 2:]

        if opt.add_body_loss:
            body_parts_pred_fake = [] # [5][2, 4] tensor of feature layer [batch, output_channel, height_feature, width_feature]
            body_parts_pred_real = [] # [5][2, 4] tensor of feature layer [batch, output_channel, height_feature, width_feature]
            for i in range(5):
                if type(body_parts_pred[i]) == list:
                    pred_fake_body = []
                    pred_real_body = []
                    for p in body_parts_pred[i]:
                        pred_fake_body.append([tensor[:tensor.size(0) // 2] for tensor in p])
                        pred_real_body.append([tensor[tensor.size(0) // 2:] for tensor in p])
                else:
                    pred_fake_body = body_parts_pred[:body_parts_pred.size(0) // 2]
                    pred_real_body = body_parts_pred[body_parts_pred.size(0) // 2:]

                body_parts_pred_fake.append(pred_fake_body)
                body_parts_pred_real.append(pred_real_body)

        
        # for i in range(len(pred_fake)):
        #     for j in range(len(pred_fake[i])):
        #         print(f"--pred_fake[{i}][{j}] shape: {pred_fake[i][j].shape}")

        # for i in range(len(body_parts_pred_fake[0])):
        #     for j in range(len(body_parts_pred_fake[0][i])):
        #         print(f"--pred_fake[{i}][{j}] shape: {body_parts_pred_fake[0][i][j].shape}")
        
        # for i in range(len(pred_real)):
        #     for j in range(len(pred_real[i])):
        #         print(f"--pred_real[{i}][{j}] shape: {pred_real[i][j].shape}")

        D_losses = {}
        D_losses['D_Fake'] = criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses['D_Real'] = criterionGAN(pred_real, True, for_discriminator=True)

        if opt.add_body_loss:
            D_Fake_body_parts_losses = {}
            D_Real_body_parts_losses = {}
            for i in range(5):
                D_Fake_body_parts_losses[f'{i}'] = criterionGAN(body_parts_pred_fake[i], False, for_discriminator=True)
                D_Real_body_parts_losses[f'{i}'] = criterionGAN(body_parts_pred_real[i], True, for_discriminator=True)
            D_losses['D_Fake_body_parts_losses'] = sum(D_Fake_body_parts_losses.values()).mean()
            D_losses['D_Real_body_parts_losses'] = sum(D_Real_body_parts_losses.values()).mean()
            # loss_body = sum(D_losses.values()).mean()
        
        if not opt.no_L1_loss:
            D_losses['L1'] = criterionL1(output, im) * opt.lambda_l1
            D_L1_body_losses = {} 
            for k in range(5):
                D_L1_body_losses[f'{k}'] = 0
                for i in range(len(body_parts_pred_fake[0])):
                    for j in range(len(body_parts_pred_fake[0][i])):
                        D_L1_body_losses[f'{k}'] += criterionL1(body_parts_pred_fake[k][i][j], body_parts_pred_real[k][i][j]) 
                D_L1_body_losses[f'{k}'] *= opt.lambda_l1
            D_losses['L1'] += sum(D_L1_body_losses.values()).mean()

        loss_dis = sum(D_losses.values()).mean()
        print(D_losses)
        print("Loss loss_dis", loss_dis)
        # print("Loss loss_body", loss_body)
        # print("Loss D_Fake", D_losses['D_Fake'])
        # print("Loss D_Real", D_losses['D_Real'])
        # print("Loss D_Fake_body_parts_losses", D_losses['D_Fake_body_parts_losses'])
        # print("Loss D_Real_body_parts_losses", D_losses['D_Real_body_parts_losses'])

        optimizer_dis.zero_grad()
        if opt.fp16:
            with amp.scale_loss(loss_dis, optimizer_dis, loss_id=1) as loss_dis_scaled:
                loss_dis_scaled.backward()
        else:
            loss_dis.backward()
            # loss_body.backward()
        optimizer_dis.step()
        # --------------------------------------------------------------------------------------------------------------
        #                                            recording
        # --------------------------------------------------------------------------------------------------------------
        if (step + 1) % opt.tensorboard_count == 0:
            i = 0
            grid = make_image_grid([(c_paired[0].cpu() / 2 + 0.5), (cm[0].cpu()).expand(3, -1, -1), ((pose.cpu()[0]+1)/2), visualize_segmap(parse_agnostic.cpu(), batch=i),
                                    (warped_cloth_paired[i].cpu() / 2 + 0.5), (agnostic[i].cpu() / 2 + 0.5), (pose[i].cpu() / 2 + 0.5), visualize_segmap(fake_parse_gauss.cpu(), batch=i),
                                    (output[i].cpu() / 2 + 0.5), (im[i].cpu() / 2 + 0.5)],
                                    nrow=4)
            board.add_images('train_images', grid.unsqueeze(0), step + 1)
            board.add_scalar('Loss/gen', loss_gen.item(), step + 1)
            board.add_scalar('Loss/gen/adv', G_losses['GAN'].mean().item(), step + 1)
            board.add_scalar('Loss/gen/l1', G_losses['L1'].mean().item(), step + 1)
            board.add_scalar('Loss/gen/feat', G_losses['GAN_Feat'].mean().item(), step + 1)
            board.add_scalar('Loss/gen/vgg', G_losses['VGG'].mean().item(), step + 1)
            board.add_scalar('Loss/dis', loss_dis.item(), step + 1)
            board.add_scalar('Loss/dis/adv_fake', D_losses['D_Fake'].mean().item(), step + 1)
            board.add_scalar('Loss/dis/adv_real', D_losses['D_Real'].mean().item(), step + 1)
            board.add_scalar('Loss/dis/l1', D_losses['L1'].mean().item(), step + 1)
            if opt.add_body_loss:
                board.add_scalar('Loss/gen/adv_body', G_losses['GAN_Body_Parts'].mean().item(), step + 1)
                board.add_scalar('Loss/dis/adv_fake_body', D_losses['D_Fake_body_parts_losses'].mean().item(), step + 1)
                board.add_scalar('Loss/dis/adv_real_body', D_losses['D_Real_body_parts_losses'].mean().item(), step + 1)
            
            # unpaired visualize
            generator.eval()
            
            inputs = test_vis_loader.next_batch()
            # input
            agnostic = inputs['agnostic'].cuda()
            parse_GT = inputs['parse'].cuda()
            pose = inputs['densepose'].cuda()
            parse_cloth = inputs['parse_cloth'].cuda()
            parse_agnostic = inputs['parse_agnostic'].cuda()
            pcm = inputs['pcm'].cuda()
            cm = inputs['cloth_mask']['unpaired'].cuda()
            c_paired = inputs['cloth']['unpaired'].cuda()
            
            # target
            im = inputs['image'].cuda()
                        
            with torch.no_grad():
                if not opt.GT:
                    # Warping Cloth
                    # down
                    pre_clothes_mask_down = F.interpolate(cm, size=(256, 192), mode='nearest')
                    input_parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='nearest')
                    clothes_down = F.interpolate(c_paired, size=(256, 192), mode='bilinear')
                    densepose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
                    
                    # multi-task inputs
                    input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
                    input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1)

                    # forward
                    flow_list, fake_segmap, _, warped_clothmask_paired = tocg(opt, input1, input2)
                    
                    # warped cloth mask one hot 
                    warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(float)).cuda()
                    
                    if opt.clothmask_composition != 'no_composition':
                        if opt.clothmask_composition == 'detach':
                            cloth_mask = torch.ones_like(fake_segmap)
                            cloth_mask[:,3:4, :, :] = warped_cm_onehot
                            fake_segmap = fake_segmap * cloth_mask
                            
                        if opt.clothmask_composition == 'warp_grad':
                            cloth_mask = torch.ones_like(fake_segmap)
                            cloth_mask[:,3:4, :, :] = warped_clothmask_paired
                            fake_segmap = fake_segmap * cloth_mask
                            
                    # warped cloth
                    N, _, iH, iW = c_paired.shape
                    grid = make_grid(N, iH, iW,opt)
                    flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
                    flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)], 3)
                    warped_grid = grid + flow_norm
                    warped_cloth_paired = F.grid_sample(c_paired, warped_grid, padding_mode='border').detach()
                    warped_clothmask = F.grid_sample(cm, warped_grid, padding_mode='border')

                    # make generator input parse map
                    fake_parse_gauss = gauss(F.interpolate(fake_segmap, size=(iH, iW), mode='bilinear'))
                    fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]

                    if opt.occlusion:
                        warped_clothmask = remove_overlap(F.softmax(fake_parse_gauss, dim=1), warped_clothmask)
                        warped_cloth_paired = warped_cloth_paired * warped_clothmask + torch.ones_like(warped_cloth_paired) * (1-warped_clothmask)
                        warped_cloth_paired = warped_cloth_paired.detach()

                else:
                    # parse pre-process
                    fake_parse = parse_GT.argmax(dim=1)[:, None]
                    warped_cloth_paired = parse_cloth
                    
                old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt.fine_height, opt.fine_width).zero_().cuda()
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
                parse = torch.FloatTensor(fake_parse.size(0), 7, opt.fine_height, opt.fine_width).zero_().cuda()
                for i in range(len(labels)):
                    for label in labels[i][1]:
                        parse[:, i] += old_parse[:, label]
                        
                parse = parse.detach()
            
                output = generator(torch.cat((agnostic, pose, warped_cloth_paired), dim=1), parse)
                
                for i in range(opt.num_test_visualize):
                    grid = make_image_grid([(c_paired[i].cpu() / 2 + 0.5), (cm[i].cpu()).expand(3, -1, -1), ((pose.cpu()[i]+1)/2), visualize_segmap(parse_agnostic.cpu(), batch=i),
                        (warped_cloth_paired[i].cpu() / 2 + 0.5), (agnostic[i].cpu() / 2 + 0.5), (pose[i].cpu() / 2 + 0.5), visualize_segmap(fake_parse_gauss.cpu(), batch=i),
                        (output[i].cpu() / 2 + 0.5), (im[i].cpu() / 2 + 0.5)],
                        nrow=4)
                    board.add_images(f'test_images/{i}', grid.unsqueeze(0), step + 1)
                
            generator.train()

        if (step + 1) % opt.lpips_count == 0:
            generator.eval()
            T1 = transforms.ToTensor()
            T2 = transforms.Compose([transforms.Resize((128, 128))])
            T3 = transforms.Compose([transforms.Resize((299, 299)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                 std=(0.5, 0.5, 0.5))])

            splits = 1 # Hyper-parameter for IS score

            inception_model = inception_v3(pretrained=True, transform_input=False).type(torch.cuda.FloatTensor)
            inception_model.eval()

            num_img = opt.num_evaluate
            lpips_list = []
            avg_ssim, avg_mse, avg_distance = 0.0, 0.0, 0.0
            preds = np.zeros((100, 1000))
            transform = transforms.ToPILImage()

            # Calculate Inception Score
            split_scores = [] # Now compute the mean kl-divergence
            
            with torch.no_grad():
                print("Evaluate")
                for i in tqdm(range(num_img)):
                    inputs = test_loader.next_batch()
                    # input
                    agnostic = inputs['agnostic'].cuda()
                    parse_GT = inputs['parse'].cuda()
                    pose = inputs['densepose'].cuda()
                    parse_cloth = inputs['parse_cloth'].cuda()
                    parse_agnostic = inputs['parse_agnostic'].cuda()
                    pcm = inputs['pcm'].cuda()
                    cm = inputs['cloth_mask']['paired'].cuda()
                    c_paired = inputs['cloth']['paired'].cuda()
                    
                    # target
                    im = inputs['image'].cuda()
                                
                    with torch.no_grad():
                        if not opt.GT:
                            # Warping Cloth
                            # down
                            pre_clothes_mask_down = F.interpolate(cm, size=(256, 192), mode='nearest')
                            input_parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='nearest')
                            clothes_down = F.interpolate(c_paired, size=(256, 192), mode='bilinear')
                            densepose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
                            
                            # multi-task inputs
                            input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
                            input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1)

                            # forward
                            flow_list, fake_segmap, _, warped_clothmask_paired = tocg(opt, input1, input2)
                            
                            # warped cloth mask one hot 
                            warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
                            
                            if opt.clothmask_composition != 'no_composition':
                                if opt.clothmask_composition == 'detach':
                                    cloth_mask = torch.ones_like(fake_segmap)
                                    cloth_mask[:,3:4, :, :] = warped_cm_onehot
                                    fake_segmap = fake_segmap * cloth_mask
                                    
                                if opt.clothmask_composition == 'warp_grad':
                                    cloth_mask = torch.ones_like(fake_segmap)
                                    cloth_mask[:,3:4, :, :] = warped_clothmask_paired
                                    fake_segmap = fake_segmap * cloth_mask
                                    
                            # warped cloth
                            N, _, iH, iW = c_paired.shape
                            flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
                            flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)], 3)
                            
                            grid = make_grid(N, iH, iW,opt)
                            warped_grid = grid + flow_norm
                            warped_cloth_paired = F.grid_sample(c_paired, warped_grid, padding_mode='border').detach()
                            warped_clothmask = F.grid_sample(cm, warped_grid, padding_mode='border')

                            # make generator input parse map
                            fake_parse_gauss = gauss(F.interpolate(fake_segmap, size=(iH, iW), mode='bilinear'))
                            fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]

                            if opt.occlusion:
                                warped_clothmask = remove_overlap(F.softmax(fake_parse_gauss, dim=1), warped_clothmask)
                                warped_cloth_paired = warped_cloth_paired * warped_clothmask + torch.ones_like(warped_cloth_paired) * (1-warped_clothmask)
                                warped_cloth_paired = warped_cloth_paired.detach()

                        else:
                            # parse pre-process
                            fake_parse = parse_GT.argmax(dim=1)[:, None]
                            warped_cloth_paired = parse_cloth
                            
                        old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt.fine_height, opt.fine_width).zero_().cuda()
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
                        parse = torch.FloatTensor(fake_parse.size(0), 7, opt.fine_height, opt.fine_width).zero_().cuda()
                        for i in range(len(labels)):
                            for label in labels[i][1]:
                                parse[:, i] += old_parse[:, label]
                                
                        parse = parse.detach()
                    
                    output_paired = generator(torch.cat((agnostic, pose, warped_cloth_paired), dim=1), parse)
                    
                    gt_img = transform(im[0])
                    pred_img = transform(output_paired[0])

                    # Calculate SSIM
                    gt_np = np.asarray(gt_img.convert('L'))
                    assert gt_img.size == pred_img.size, f"{gt_img.size} vs {pred_img.size}"
                    pred_np = np.asarray(pred_img.convert('L'))
                    avg_ssim += ssim(gt_np, pred_np, data_range=255, gaussian_weights=True, use_sample_covariance=False)

                    # Calculate Inception model prediction
                    pred_img_IS = T3(pred_img).unsqueeze(0).cuda()
                    preds[i] = F.softmax(inception_model(pred_img_IS)).data.cpu().numpy()

                    gt_img_MSE = T1(gt_img).unsqueeze(0).cuda()
                    pred_img_MSE = T1(pred_img).unsqueeze(0).cuda()
                    avg_mse += F.mse_loss(gt_img_MSE, pred_img_MSE)

                    # Caculate Lpips
                    avg_distance += model.forward(T2(im), T2(output_paired))

                print("Calculate Inception Score...")
                for k in range(splits):
                    part = preds[k * (num_img // splits): (k+1) * (num_img // splits), :]
                    py = np.mean(part, axis=0)
                    scores = []
                    for i in range(part.shape[0]):
                        pyx = part[i, :]
                        scores.append(entropy(pyx + 1e-10, py + 1e-10))
                    split_scores.append(np.exp(np.mean(scores)))

                IS_mean, IS_std = np.mean(split_scores), np.std(split_scores)

            avg_ssim /= num_img
            avg_mse = avg_mse / num_img
            avg_distance = avg_distance / num_img
                    
            print(f"SSIM : {avg_ssim} / MSE : {avg_mse} / LPIPS : {avg_distance}")
            print(f"IS_mean : {IS_mean} / IS_std : {IS_std}")
            board.add_scalar('test/SSIM', avg_ssim, step + 1)
            board.add_scalar('test/MSE', avg_mse, step + 1)
            board.add_scalar('test/LPIPS', avg_distance, step + 1)
            board.add_scalar('test/IS_mean', IS_mean, step + 1)
            board.add_scalar('test/IS_std', IS_std, step + 1)
                
            generator.train()

        if (step + 1) % opt.display_count == 0:
            t = time.time() - iter_start_time
            print("step: %8d, time: %.3f, G_loss: %.4f, G_adv_loss: %.4f, D_loss: %.4f, D_fake_loss: %.4f, D_real_loss: %.4f"
                  % (step + 1, t, loss_gen.item(), G_losses['GAN'].mean().item(), loss_dis.item(),
                     D_losses['D_Fake'].mean().item(), D_losses['D_Real'].mean().item()), flush=True)

        if (step + 1) % opt.save_count == 0:
            save_checkpoint(generator.module, os.path.join(opt.checkpoint_dir, opt.name, 'gen_step_%06d.pth' % (step + 1)),opt)
            save_checkpoint(discriminator.module, os.path.join(opt.checkpoint_dir, opt.name, 'dis_step_%06d.pth' % (step + 1)),opt)

        if (step + 1) % 1000 == 0:
            scheduler_gen.step()
            scheduler_dis.step()


def main():
    opt = get_opt()
    print(opt)
    print("Start to train %s!" % opt.name)

    # create dataset
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)
    
    # test dataloader
    opt.batch_size = 1
    opt.dataroot = opt.test_dataroot
    opt.datamode = 'test'
    opt.data_list = opt.test_data_list
    test_dataset = CPDatasetTest(opt)
    test_dataset = Subset(test_dataset, np.arange(500))
    test_loader = CPDataLoader(opt, test_dataset)
    print("done_test_loader")
    # test vis loader
    opt.batch_size = opt.num_test_visualize
    test_vis_dataset = CPDatasetTest(opt)
    test_vis_loader = CPDataLoader(opt, test_vis_dataset)
    print("done_test_vis_loader")
    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))
    
    # warping-seg Model
    tocg = None
    
    if not opt.GT:
        input1_nc = 4  # cloth + cloth-mask
        input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
        tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=13, ngf=96, norm_layer=nn.BatchNorm2d)
        # Load Checkpoint
        print("--Load tocg check point")
        load_checkpoint(tocg, opt.tocg_checkpoint, opt)
    print("done_load_checkpoint")
    # Generator model
    generator = SPADEGenerator(opt, 3+3+3)
    generator.print_network()
    if len(opt.gpu_ids) > 0:
        print("cuda is available")
        assert(torch.cuda.is_available())
        generator.cuda()
    generator.init_weights(opt.init_type, opt.init_variance)
    discriminator = create_network(MultiscaleDiscriminator, opt, opt.gen_semantic_nc)
    print("done_Generator model")
    # lpips
    model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=True)

    # Load Checkpoint
    if not opt.gen_checkpoint == '' and os.path.exists(opt.gen_checkpoint):
        load_checkpoint(generator, opt.gen_checkpoint, opt)
        load_checkpoint(discriminator, opt.dis_checkpoint, opt)
    print("done_load_checkpoint")
    # Train
    train(opt, train_loader, test_loader, test_vis_loader, board, tocg, generator, discriminator, model)

    # Save Checkpoint
    save_checkpoint(generator, os.path.join(opt.checkpoint_dir, opt.name, 'gen_model_final.pth'),opt)
    save_checkpoint(discriminator, os.path.join(opt.checkpoint_dir, opt.name, 'dis_model_final.pth'),opt)

    print("Finished training %s!" % opt.name)


if __name__ == "__main__":
    main()
