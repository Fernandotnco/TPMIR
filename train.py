from data_loader import ImageDataset, DatasetSplit
from ST_CGAN import Generator, Discriminator
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torch.autograd import Variable
from collections import OrderedDict
from torchvision import models
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import time
import torch
import os

torch.manual_seed(44)
# choose your device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def get_parser():
    parser = argparse.ArgumentParser(
        prog='ST-CGAN: Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal',
        usage='python3 main.py',
        description='This module demonstrates shadow detection and removal using ST-CGAN.',
        add_help=True)

    parser.add_argument('-e', '--epoch', type=int, default=10000, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('-l', '--load', type=str, default=None, help='the number of checkpoints')
    parser.add_argument('-hor', '--hold_out_ratio', type=float, default=0.8, help='training-validation ratio')
    parser.add_argument('-s', '--image_size', type=int, default=286)
    parser.add_argument('-cs', '--crop_size', type=int, default=256)
    parser.add_argument('-lr', '--lr', type=float, default=2e-4)

    return parser

def fix_model_state_dict(state_dict):
    '''
    remove 'module.' of dataparallel
    '''
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict

def set_requires_grad(nets, requires_grad=False):
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def unnormalize(x):
    x = x.transpose(1, 3)
    #mean, std
    x = x * torch.Tensor((0.5, )) + torch.Tensor((0.5, ))
    x = x.transpose(1, 3)
    return x

def evaluate(G1, dataset, device, filename):
    img, prevImg = zip(*[dataset[i] for i in range(8)])
    img = torch.stack(img)
    prevImg = torch.stack(prevImg)

    with torch.no_grad():
        newCompass = G1(prevImg.to(device))
        newCompass = newCompass.to(torch.device('cpu'))
        concat = torch.cat([prevImg, newCompass, prevImg, img], dim=3)

    print(newCompass[6])
    plt.imshow(concat[6][0,:,:])
    plt.show()
    plt.imshow(newCompass[6][0,:,:])
    plt.show()
    plt.imshow(prevImg[6][0,:,:])
    plt.show()
    plt.imshow(img[6][0,:,:])
    plt.show()

    save_image(concat, filename+'_Generated.jpg')

def plot_log(data, save_model_name='model'):
    plt.cla()
    plt.plot(data['G'], label='G_loss ')
    plt.plot(data['D'], label='D_loss ')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.savefig('./logs/'+save_model_name+'.png')

def check_dir():
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    if not os.path.exists('./result'):
        os.mkdir('./result')

def train_model(G1, D1, dataloader, val_dataset, num_epochs, parser, save_model_name='model'):

    check_dir()
    print("TRAINING")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    G1.to(device)
    D1.to(device)

    """use GPU in parallel"""
    if device == 'cuda':
        G1 = torch.nn.DataParallel(G1)
        D1 = torch.nn.DataParallel(D1)
        print("parallel mode")

    print("device:{}".format(device))

    lr = parser.lr
    beta1, beta2 = 0.5, 0.999

    optimizerG = torch.optim.Adam([{'params': G1.parameters()}],
                                  lr=lr,
                                  betas=(beta1, beta2))
    optimizerD = torch.optim.Adam([{'params': D1.parameters()}],
                                  lr=lr,
                                  betas=(beta1, beta2))

    criterionGAN = nn.BCEWithLogitsLoss().to(device)

    torch.backends.cudnn.benchmark = True

    mini_batch_size = parser.batch_size
    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    lambda_dict = {'lambda1':5, 'lambda2':0.1, 'lambda3':0.1}

    iteration = 1
    g_losses = []
    d_losses = []

    for epoch in range(num_epochs+1):
        print(epoch)

        G1.train()
        D1.train()
        t_epoch_start = time.time()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        print('-----------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('(train)')

        for images, prevImgs in tqdm(dataloader):
            # if size of minibatch is 1, an error would be occured.
            if images.size()[0] == 1:
                continue

            images = images.to(device)
            prevImgs = prevImgs.to(device)

            mini_batch_size = images.size()[0]

            # Train Discriminator
            set_requires_grad([D1], True)  # enable backprop$
            optimizerD.zero_grad()

            # for D1
            newCompass = G1(prevImgs)
            fake1 = torch.cat([prevImgs, newCompass], dim=3)
            real1 = torch.cat([prevImgs, images], dim=3)
            D_input_1 = torch.cat([fake1.detach(), real1], dim = 1)
            D_input_2 = torch.cat([real1, fake1.detach()], dim = 1)
            out_1_D1 = D1(D_input_1)
            out_2_D1 = D1(D_input_2)

            # L_CGAN1
            zeros = torch.tensor(np.zeros((newCompass.shape[0], 1, 1, 1)))
            ones = torch.tensor(np.ones((newCompass.shape[0], 1, 1, 1)))

            label_1_D1 = torch.cat([zeros, ones], dim = 1)
            label_2_D1 = torch.cat([ones, zeros], dim = 1)

            loss_1_D1 = criterionGAN(out_1_D1, label_1_D1)
            loss_2_D1 = criterionGAN(out_2_D1, label_2_D1)
            
            D_L_CGAN1 = loss_1_D1 + loss_2_D1

            # total
            D_loss = D_L_CGAN1
            D_loss.backward()
            optimizerD.step()

            # Train Generator
            set_requires_grad([D1, D1], False)
            optimizerG.zero_grad()

            # L_CGAN1
            fake1 = torch.cat([prevImgs, newCompass], dim=3)
            real1 = torch.cat([prevImgs, images], dim=3)
            D_input_1 = torch.cat([fake1.detach(), real1], dim = 1)
            D_input_2 = torch.cat([real1, fake1.detach()], dim = 1)
            out_1_D1 = D1(D_input_1)
            out_2_D1 = D1(D_input_2)

            

            loss_1_G1 = criterionGAN(out_1_D1, label_2_D1)
            loss_2_G1 = criterionGAN(out_2_D1, label_1_D1)
            G_L_CGAN1 = loss_1_G1 + loss_2_G1


            #total
            G_loss = G_L_CGAN1
            G_loss.requires_grad = True
            G_loss.backward()
            optimizerG.step()

            epoch_d_loss += D_loss.item()
            epoch_g_loss += G_loss.item()

        t_epoch_finish = time.time()
        print('-----------')
        print('epoch {} || Epoch_D_Loss:{:.4f} || Epoch_G_Loss:{:.4f}'.format(epoch, epoch_d_loss/batch_size, epoch_g_loss/batch_size))
        print('timer: {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))

        d_losses += [epoch_d_loss/batch_size]
        g_losses += [epoch_g_loss/batch_size]
        t_epoch_start = time.time()
        plot_log({'G':g_losses, 'D':d_losses}, save_model_name)

        if(epoch%1 == 0):
            if parser.load is not None:
                torch.save(G1.state_dict(), 'checkpoints/'+save_model_name+'_G1_'+str(epoch + int(parser.load) + 1)+'.pth')
                torch.save(D1.state_dict(), 'checkpoints/'+save_model_name+'_D1_'+str(epoch+ int(parser.load) + 1)+'.pth')
            else:
                torch.save(G1.state_dict(), 'checkpoints/'+save_model_name+'_G1_'+str(epoch)+'.pth')
                torch.save(D1.state_dict(), 'checkpoints/'+save_model_name+'_D1_'+str(epoch)+'.pth')
            G1.eval()
            evaluate(G1, val_dataset, device, '{:s}/val_{:d}'.format('result', epoch))

    return G1, D1



def main(parser):
    G1 = Generator(input_channels=1, output_channels=1)
    D1 = Discriminator(input_channels=2)

    '''load'''
    if parser.load is not None:
        print('load checkpoint ' + parser.load)

        G1_weights = torch.load('./checkpoints/ST-CGAN_G1_'+parser.load+'.pth')
        G1.load_state_dict(fix_model_state_dict(G1_weights))

        D1_weights = torch.load('./checkpoints/ST-CGAN_D1_'+parser.load+'.pth')
        D1.load_state_dict(fix_model_state_dict(D1_weights))


    train_img_list, val_img_list, test_img_list = DatasetSplit('dbMetadata.json')

    mean = (0.5,)
    std = (0.5,)
    size = parser.image_size
    crop_size = parser.crop_size
    batch_size = parser.batch_size
    num_epochs = parser.epoch

    train_dataset = ImageDataset(img_list=train_img_list, dir = 'dataset')
    val_dataset = ImageDataset(img_list=val_img_list, dir = 'dataset')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #num_workers=4

    G1_update, D1_update = train_model(G1, D1, dataloader=train_dataloader,
                                                            val_dataset=val_dataset, num_epochs=num_epochs,
                                                            parser=parser, save_model_name='ST-CGAN')

if __name__ == "__main__":
    parser = get_parser().parse_args()
    main(parser)
