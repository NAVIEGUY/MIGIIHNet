import sys
import os
import torch
import torch.nn
import torch.optim
from torchvision.utils import save_image
import torchvision
import torch.nn.functional as F
import math
import numpy as np
import tqdm
from model import *
import config as c

from dataset import trainloader, testloader
from ssim import SSIM
import modules.Unet_common as common
import warnings
from skimage.metrics import structural_similarity as ssim
warnings.filterwarnings("ignore")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def ssim_image(image1, image2):
    image1 = image1.astype(np.uint8)
    image2 = image2.astype(np.uint8)
    image1 = image1 / 255.0
    image2 = image2 / 255.0
    image1 = 65.481 * image1[:, :, 0] + 128.553 * image1[:, :, 1] + 24.966 * image1[:, :, 2] + 16
    image2 = 65.481 * image2[:, :, 0] + 128.553 * image2[:, :, 1] + 24.966 * image2[:, :, 2] + 16
    image1 = image1 / 255.0
    image2 = image2 / 255.0
    # 只计算Y通道的值
   # print(image1.dtype)
    image1 = np.expand_dims(image1, axis=2)
    image2 = np.expand_dims(image2, axis=2)
   # print(image1.shape)
    #print(image1.shape)
    ssim_val = ssim(image1, image2, win_size=11, gaussian_weights=True, multichannel=True, data_range=1.0, K1=0.01, K2=0.03, sigma=1.5)


    return ssim_val

# def ssim_loss(rev_input, ori_input):
#     SSIM_loss = SSIM(window_size=11).to(device)
#     #rev_input = rev_input
#     #ori_input = ori_input
#     loss = SSIM_loss(rev_input, ori_input)
#     return loss


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def gauss_noise(shape):
    noise = torch.zeros(shape).to(device)
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).to(device)

    return noise


def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(rev_input, input)
    return loss.to(device)

def reconstruction_z_loss(z1, z):
    loss_fn = torch.nn.L1Loss(reduce=True, size_average=False)
    loss = loss_fn(z1, z)
    return loss.to(device)

def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)



# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def load(name, net, optim):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


def init_net2(mod):
    for key, param in mod.named_parameters():
        if param.requires_grad:
            param.data = 0.1 * torch.randn(param.data.shape).to(device)
#####################
# Model initialize: #
#####################
net1 = Model()
net2 = MC_model()
#SSIM_loss = SSIM(window_size=11).to(device)
net1 = net1.to(device)
net2 = net2.to(device)
init_model(net1)
init_net2(net2)

#net1 = torch.nn.DataParallel(net1, device_ids=c.device_ids)
#net2 = torch.nn.DataParallel(net2, device_ids=c.device_ids)

# para1 = get_parameter_number(net1)
#
# print(para1)
#
# params_trainable1 = (list(filter(lambda p: p.requires_grad, net1.parameters())))

optim1 = torch.optim.Adam(net1.parameters(), lr=c.lr1, betas=c.betas1, eps=1e-6, weight_decay=c.weight_decay)
optim2 = torch.optim.Adam(net2.parameters(), lr=c.lr2, betas=c.betas2, eps=1e-6, weight_decay=c.weight_decay)

weight_scheduler1 = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)
weight_scheduler2 = torch.optim.lr_scheduler.StepLR(optim2, c.weight_step, gamma=c.gamma)


dwt = common.DWT()
iwt = common.IWT()

file_write = open('evalue.txt', mode='w+')
file_write.write("SSIM_stego" + "\t\t\t" + "psnr_stego" + "\t\t\t" + "SSIM_secret" + "\t\t\t" + "psnr_secret" + "\n")
for i_epoch in range(c.epochs):
    #################
    #     train:    #
    #################
    # vgg_loss = VGGLoss(3, 1, False)
    # vgg_loss.to(device)
    for i_batch, data in enumerate(trainloader):
        # data preparation
        data = data.to(device)
        cover = data[:data.shape[0] // 2]  # channels = 3
        secret = data[data.shape[0] // 2: 2 * (data.shape[0] // 2)]
        cover_dwt = dwt(cover)
        secret_dwt = dwt(secret)
        input = torch.cat((cover_dwt, secret_dwt), 1)  # channels = 24

        #################
        #    forward1:   #
        #################
        output = net1(input)  # channels = 24
        output_steg = output.narrow(1, 0, 4 * c.channels_in)
        output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
        steg_img = iwt(output_steg)
        #output_z= iwt(output_z)





        #################
        #   backward1:   #
        #################
        output_z_guass = gauss_noise(output_z.shape)  # channels = 12
        #output_z_restruct = output_z_guass
        output_z_restruct = net2(output_steg) * output_z_guass
        #output_z_restruct_dwt = dwt(output_z_restruct)
        output_rev = torch.cat((output_steg, output_z_restruct), 1)  # channels = 24

        output_image = net1(output_rev, rev=True)

        secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
        secret_rev = iwt(secret_rev)

        #################
        #     loss:     #
        #################
        #print(steg_img.shape)
        g_loss = guide_loss(steg_img, cover)
        r_loss = reconstruction_loss(secret_rev, secret)
        steg_low = output_steg.narrow(1, 0, c.channels_in)
        cover_low = cover_dwt.narrow(1, 0, c.channels_in)
        l_loss = low_frequency_loss(steg_low, cover_low)
        z_loss = reconstruction_z_loss(output_z_restruct, output_z)

        total_loss = g_loss + 3 * r_loss + l_loss + z_loss
        
        total_loss.backward()
        optim1.step()
        optim2.step()
        optim1.zero_grad()
        optim2.zero_grad()
        #print('epoch{}/{}:g_loss:{}, r_secret_loss:{}, r_frequency_loss:{}, z_loss:{}'.format(i_epoch, i_batch, g_loss.item(), r_loss.item(), l_loss.item(), z_loss.item()))
    weight_scheduler1.step()
    weight_scheduler2.step()


    #################
    #     val:    #
    #################
    if i_epoch % c.val_freq == 0:
        torch.save(net1.state_dict(), os.path.join(c.MODEL_PATH, 'net1_{}.pth'.format(i_epoch)))
        torch.save(net2.state_dict(), os.path.join(c.MODEL_PATH, 'net2_{}.pth'.format(i_epoch)))
        with torch.no_grad():
        #    net1.eval()
        #    net2.eval()
            average_stego_ssim = 0
            average_stego_psnr = 0
            average_secret_ssim = 0
            average_secret_psnr = 0

            for i, x in enumerate(testloader):
                x = x.to(device)
                cover = x[:x.shape[0] // 2]  # channels = 3
                secret = x[x.shape[0] // 2: 2 * x.shape[0] // 2]
                cover_dwt = dwt(cover)
                secret_dwt = dwt(secret)



                input = torch.cat((cover_dwt, secret_dwt), 1)  # channels = 24

                #################
                #    forward1:   #
                #################
                output = net1(input)  # channels = 24
                output_steg = output.narrow(1, 0, c.channels_in * 4)  # channels = 12
                output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
                steg_img = iwt(output_steg)
                #output_z_iwt = iwt(output_z)

                #################0
                #   backward1:   #
                #################
                output_z_guass = gauss_noise(output_z.shape)  # channels = 12
                output_z_restruct = net2(output_steg) * output_z_guass
                #output_z_restruct = dwt(output_z_restruct)
                output_rev = torch.cat((output_steg, output_z_restruct), 1)  # channels = 24
                output_image = net1(output_rev, rev=True)

                secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
                secret_rev = iwt(secret_rev)

                secret_rev = secret_rev.detach()
                secret_1_255 = secret.detach()
                cover_255 = cover.detach()
                steg_1_255 = steg_img.detach()
                                
                if(i<6):
                    stego = torch.cat([cover_255, steg_1_255], dim=0)
                    root_output_stego = os.path.join(c.save_image, "sample{}_stego_{}.jpg".format(i, i_epoch))
                    save_image(stego, root_output_stego, nrow=c.val_batch_size // 2)

                    rev_se = torch.cat([secret_1_255, secret_rev], dim=0)
                    root_output_secret = os.path.join(c.save_image, "sample{}_secret_{}.jpg".format(i, i_epoch))
                    save_image(rev_se, root_output_secret, nrow=c.val_batch_size // 2)
                    
                   

                ssim_stego = ssim_image(np.array(steg_1_255.squeeze(0).permute(1, 2, 0).cpu() * 255),
                                        np.array(cover_255.squeeze(0).permute(1, 2, 0).cpu() * 255))
                psnr_stego = psnr(np.array(steg_1_255.cpu()) * 255, np.array(cover_255.cpu()) * 255)
                ssim_secret = ssim_image(np.array(secret_1_255.squeeze(0).permute(1, 2, 0).cpu() * 255),
                                         np.array(secret_rev.squeeze(0).permute(1, 2, 0).cpu() * 255))
                psnr_secret = psnr(np.array(secret_1_255.cpu()) * 255, np.array(secret_rev.cpu()) * 255)

                average_secret_ssim += ssim_secret
                average_secret_psnr += psnr_secret
                average_stego_ssim += ssim_stego
                average_stego_psnr += psnr_stego

            file_write.write(str(average_stego_ssim / 50) + "\t\t\t" + str(average_stego_psnr / 50) + "\t\t\t" +
                             str(average_secret_ssim / 50) + "\t\t\t" + str(average_secret_psnr / 50) + "\n")
file_write.close()


