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
import cv2

from numpy.random import default_rng

torch.manual_seed(44)
# choose your device
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--load_dir', type=str, default='./')
    parser.add_argument('--disc_epochs', type=int, default='2')

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

def evaluate(G1,G2, dataset, device, filename):
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

    #save_image(concat[6][0,:,:], filename+'_Generated.jpg')

def plot_log(data, save_model_name='model', load_dir = '.'):
    plt.cla()
    plt.plot(data['G1'], label='G1_loss ')
    plt.plot(data['G2'], label='G2_loss ')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.savefig(load_dir + 'logs/'+save_model_name+'.png')

def check_dir(load_dir):
    if not os.path.exists(load_dir + 'logs'):
        os.mkdir(load_dir + 'logs')
    if not os.path.exists(load_dir + 'checkpoints'):
        os.mkdir(load_dir + 'checkpoints')
    if not os.path.exists(load_dir + 'result'):
        os.mkdir(load_dir + 'result')

def train_model(G1, D1, dataloader, val_dataset, num_epochs, parser, save_model_name='model'):

    check_dir(parser.load_dir)
    print("TRAINING")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    G1.to(device)
    #G2.to(device)
    D1.to(device)

    """use GPU in parallel"""
    '''if device == 'cuda':
        G1 = torch.nn.DataParallel(G1)
        #G2 = torch.nn.DataParallel(G2)
        D1 = torch.nn.DataParallel(D1)
        print("parallel mode")'''

    print("device:{}".format(device))

    lr = parser.lr
    beta1, beta2 = 0.9, 0.999

    optimizerG1 = torch.optim.Adam([{'params': G1.parameters()}],
                                  lr=lr*2,
                                  betas=(beta1, beta2))
    '''optimizerG2 = torch.optim.Adam([{'params': G2.parameters()}],
                                  lr=lr * 15,
                                  betas=(beta1, beta2))'''
    optimizerD = torch.optim.Adam([{'params': D1.parameters()}],
                                  lr=lr,
                                  betas=(beta1, beta2))

    criterionGAN = nn.L1Loss().to(device)

    torch.backends.cudnn.benchmark = True

    mini_batch_size = parser.batch_size
    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    lambda_dict = {'lambda1':5, 'lambda2':0.1, 'lambda3':0.1}

    iteration = 1
    g1_losses = []
    #g2_losses = []
    d_losses = []

    good_G1s = []
    #good_G2s = []

    
    blackImg = torch.zeros((parser.batch_size, 1, 88, 64)).to(device)
    zeros = torch.zeros((parser.batch_size, 1)).to(device)
    ones = torch.ones((parser.batch_size, 1)).to(device)

    count = 0
    for epoch in range(num_epochs+1):
        print(epoch)

        G1.train()
        #G2.train()
        D1.train()
        t_epoch_start = time.time()

        epoch_g1_loss = 0.0
        #epoch_g2_loss = 0.0
        epoch_d_loss = 0.0

        good_G1 = 0
        #good_G2 = 0


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

            rg = default_rng()

            # Train Discriminator

            # for D1
            newCompass1 = G1(prevImgs)
            #newCompass2 = G2(prevImgs)

            c = rg.choice([0,1], (newCompass1.shape[0],1))
            c_diff = (c-1)*-1
            l = np.array(range(newCompass1.shape[0]))
            l = np.reshape(l,(newCompass1.shape[0], 1))

            fake1 = torch.cat([prevImgs, newCompass1], dim=3)
            #fake2 = torch.cat([prevImgs, newCompass2], dim=3)


            real1 = torch.cat([prevImgs, images], dim=3)
            aux = torch.cat([fake1, real1], dim = 1)

            D_input_G1 = torch.cat([aux[l,c,:,:], aux[l,c_diff,:,:]], dim = 1)



            #aux = torch.cat([fake2.detach(), real1], dim = 1)
            '''D_input_G2 = torch.cat([real1, torch.flip(real1, dims = 0)], dim = 1)
            plt.imshow(D2_input_G1[0][1].detach().cpu())
            plt.show()'''
            


            out_1_D1 = D1(D_input_G1)
            #out_2_D1 = D1(D_input_G2)


            # L_CGAN1

            aux = torch.cat([zeros[0:newCompass1.shape[0]], ones[0:newCompass1.shape[0]]], axis = 1)

            labels = torch.cat([aux[l,c], aux[l,c_diff]], axis = 1)
            inv_labels = torch.cat([aux[l,c_diff], aux[l,c]], axis = 1)
            loss_1_D1 = criterionGAN(out_1_D1, labels).to(device)
            #loss_2_D1 = criterionGAN(out_2_D1, labels).to(device)

            '''print(out_1_D1)
            print(labels)
            print(loss_1_D1)

            print(out_2_D1)
            print(labels)
            print(loss_2_D1)'''

            '''if(loss_2_D1 > 0.4):
              cv2.imwrite("goodImg_G2.png", np.array(newCompass2[0][0,:,:].cpu())* 254)
              cv2.imwrite("goodImg2._G2.png", np.array(newCompass2[1][0,:,:].cpu())* 254)
              good_G2 +=1
              print(out_2_D1)
              print(labels)'''
            
            
            D_L_CGAN1 = loss_1_D1

            # total
            D_loss = D_L_CGAN1
            '''if(epoch % parser.disc_epochs == 0):
                D_loss.backward(retain_graph=True)
                optimizerD.step()'''

            # Train Generator
            #optimizerG2.zero_grad()

            # L_CGAN1

            #G1

            '''fakeG1 = torch.cat([prevImgs, newCompass1], dim=3)
            fakeG2 = torch.cat([prevImgs, newCompass2], dim=3)
            real1 = torch.cat([prevImgs, images], dim=3)

            D_input_1_G1 = torch.cat([fakeG1.detach(), real1], dim = 1).to(device)
            D_input_2_G1 = torch.cat([real1, fakeG1.detach()], dim = 1).to(device)

            D_input_1_G2 = torch.cat([fakeG2.detach(), real1], dim = 1).to(device)
            D_input_2_G2 = torch.cat([real1, fakeG2.detach()], dim = 1).to(device)'''

            '''D_input_3 = torch.cat([fake2, fake1], dim = 1).to(device)

            aux = torch.cat([fake1, fake2], dim = 1)
            D_input_3 = torch.cat([aux[l,c,:,:], aux[l,c_diff,:,:]], dim = 1)

            out_D1_G1 = D1(D_input_G1)

            out_D1_G2 = D1(D_input_G2)
  

            out_3_D1 = D1(D_input_3)'''
            '''print(out_3_D1)
            print(np.sum(np.array(newCompass1[0][0].cpu())))
            cv2.imwrite("test.png", np.array(D_input_3[0][0,:,:].cpu())* 2)'''

            #out_D1_G1 = D1(D_input_G1)


            #loss_1_G1 = criterionGAN(out_D1_G1, inv_labels)
            #loss_2_G1 = criterionGAN(out_3_D1, inv_labels)

            loss_1_G1 = criterionGAN(out_1_D1, inv_labels)


            G_L_CGAN1 = loss_1_G1

            '''loss_1_G2 = criterionGAN(out_D1_G2, inv_labels).to(device)
            loss_2_G2 = criterionGAN(out_3_D1, labels)

            G_L_CGAN2 = loss_1_G2*3 + loss_2_G2'''


            #total
            G_loss_G1 = G_L_CGAN1
            if((count + epoch)%2 == 0):
                set_requires_grad([D1], False)
                optimizerG1.zero_grad()
                #a = list(G1.parameters())[0].clone()
                G_loss_G1.backward()
                optimizerG1.step()
            else:
                set_requires_grad([D1], True)  # enable backprop$
                optimizerD.zero_grad()
                D_loss.backward(retain_graph=False)
                optimizerD.step()
                good_G1 +=1
            #b = list(G1.parameters())[0].clone()
            #print(a==b)
            count += 1


            '''optimizerG2.zero_grad()

            G_loss_G2 = G_L_CGAN2
            G_loss_G2.requires_grad = True
            G_loss_G2.backward()
            optimizerG2.step()'''

            if(loss_1_D1 > 0.4):
              cv2.imwrite("goodImg_G1.png", np.array(newCompass1[0][0,:,:].detach().cpu())* 254)
              cv2.imwrite("goodImg2_G1.png", np.array(newCompass1[1][0,:,:].detach().cpu())* 254)
              '''print(out_1_D1)
              print(labels)'''

            epoch_d_loss += D_loss.item()
            epoch_g1_loss += G_loss_G1.item()
            #epoch_g2_loss += G_loss_G2.item()

        t_epoch_finish = time.time()
        print('-----------')
        #print('epoch {} || Epoch_D_Loss:{:.4f} || Epoch_G1_Loss:{:.4f} || Good_G1: {} || Epoch_G2_Loss:{:.4f} || Good_G2: {} '.format(epoch, epoch_d_loss/batch_size, epoch_g1_loss/batch_size, good_G1,epoch_g2_loss/batch_size, good_G2))
        print('epoch {} || Epoch_D_Loss:{:.4f} || Epoch_G1_Loss:{:.4f} || Good_G1: {}'.format(epoch, epoch_d_loss/batch_size, epoch_g1_loss/batch_size, good_G1))
        print('timer: {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))

        d_losses += [epoch_d_loss/batch_size]
        g1_losses += [epoch_g1_loss/batch_size]
        #g2_losses += [epoch_g2_loss/batch_size]
        good_G1s.append(good_G1)
        #good_G2s.append(good_G2)
        t_epoch_start = time.time()
        #plot_log({'G1':g1_losses, 'G2':g2_losses, 'D':d_losses}, save_model_name)
        #plot_log({'G1': good_G1s, 'G2': good_G2s}, save_model_name, parser.load_dir)

        if(epoch%10 == 0):
            if parser.load is not None:
                torch.save(G1.state_dict(), parser.load_dir + 'checkpoints/'+save_model_name+'_G1_'+str(epoch + int(parser.load) + 1)+'.pth')
                #torch.save(G2.state_dict(), parser.load_dir + 'checkpoints/'+save_model_name+'_G2_'+str(epoch + int(parser.load) + 1)+'.pth')
                torch.save(D1.state_dict(), parser.load_dir + 'checkpoints/'+save_model_name+'_D1_'+str(epoch+ int(parser.load) + 1)+'.pth')
            else:
                torch.save(G1.state_dict(), parser.load_dir + 'checkpoints/'+save_model_name+'_G1_'+str(epoch)+'.pth')
                #torch.save(G2.state_dict(), parser.load_dir + 'checkpoints/'+save_model_name+'_G2_'+str(epoch)+'.pth')
                torch.save(D1.state_dict(), parser.load_dir + 'checkpoints/'+save_model_name+'_D1_'+str(epoch)+'.pth')
            G1.eval()
            #G2.eval()
            #evaluate(G1, G2, val_dataset, device, '{:s}/val_{:d}'.format('result', epoch))

    return G1, D1



def main(parser):
    G1 = Generator(input_channels=1, output_channels=1)
    #G2 = Generator(input_channels=1, output_channels=1)
    D1 = Discriminator(input_channels=2)

    '''load'''
    if parser.load is not None:
        print('load checkpoint ' + parser.load)

        G1_weights = torch.load(parser.load_dir + 'checkpoints/ST-CGAN_G1_'+parser.load+'.pth')
        G1.load_state_dict(fix_model_state_dict(G1_weights))

        '''G2_weights = torch.load(parser.load_dir + 'checkpoints/ST-CGAN_G2_'+parser.load+'.pth')
        G2.load_state_dict(fix_model_state_dict(G2_weights))'''

        D1_weights = torch.load(parser.load_dir + 'checkpoints/ST-CGAN_D1_'+parser.load+'.pth')
        D1.load_state_dict(fix_model_state_dict(D1_weights))


    train_img_list, val_img_list, test_img_list = DatasetSplit('dbMetadata.json', dataset_dir = parser.dataset_dir, val_rate = 0.1, test_rate = 0.05)

    mean = (0.5,)
    std = (0.5,)
    size = parser.image_size
    crop_size = parser.crop_size
    batch_size = parser.batch_size
    num_epochs = parser.epoch

    train_dataset = ImageDataset(img_list=train_img_list, dir = parser.dataset_dir)
    val_dataset = ImageDataset(img_list=val_img_list, dir = parser.dataset_dir)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #num_workers=4

    G1_update, D1_update = train_model(G1, D1, dataloader=train_dataloader,
                                                            val_dataset=val_dataset, num_epochs=num_epochs,
                                                            parser=parser, save_model_name='ST-CGAN')

if __name__ == "__main__":
    parser = get_parser().parse_args()
    main(parser)
    rg = default_rng()
    zeros = torch.tensor(np.zeros((32, 1)))
    ones = torch.tensor(np.ones((32, 1)))

    labels = torch.cat([zeros, ones], axis = 1)
    labels2 = torch.cat([ones, zeros], axis = 1)
    print(labels)
    print(rg.choice([0,1], (32,1)))
    c = (rg.choice([0,1], (32,1)))
    l = np.array(range(32))
    l = np.reshape(l,(32, 1))

    l1 = labels[(l,c)]
    l2 = labels2[(l,c)]
    labels = torch.cat([l1, l2], axis = 1)
    print(labels)
