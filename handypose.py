# -*-coding:UTF-8-*-
import argparse
import time
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import numpy as np
import cv2
import math
sys.path.append("..")
from utils.utils import adjust_learning_rate as adjust_learning_rate
from utils.utils import save_checkpoint      as save_checkpoint
from utils.utils import printAccuracies      as printAccuracies
from utils.utils import guassian_kernel      as guassian_kernel
from utils.utils import get_parameters       as get_parameters
from utils       import Mytransforms         as  Mytransforms 
from utils.utils import getDataloader        as getDataloader
from utils.utils import getOutImages         as getOutImages
from utils.utils import AverageMeter         as AverageMeter
from utils.utils import draw_paint           as draw_paint
from utils       import evaluate             as evaluate
from utils.utils import get_kpts             as get_kpts
from utils.utils import get_model_summary

from model.handypose import handypose

from tqdm import tqdm

import torch.nn.functional as F
from collections import OrderedDict
from torchsummary import summary

from PIL import Image


class Trainer(object):
    def __init__(self, args):
        self.args         = args
        self.train_dir    = args.train_dir
        self.val_dir      = args.val_dir
        self.test_dir      = args.test_dir
        self.model_arch   = args.model_arch
        self.dataset      = args.dataset


        self.workers      = 1
        self.weight_decay = 0.0005
        self.momentum     = 0.9
        self.batch_size   = 8
        self.lr           = 0.0001
        self.gamma        = 0.333
        self.step_size    = 13275
        self.sigma        = 3
        self.stride       = 8

        cudnn.benchmark   = True

        if self.dataset   ==  "LSP":
            self.numClasses  = 14
        elif self.dataset == "MPII":
            self.numClasses  = 16
        elif self.dataset == "CMUHand":
            self.numClasses  = 21
        elif self.dataset == "MPII_NZSLHand":
            self.numClasses  = 21
        elif self.dataset == "VeRi":
            self.numClasses  = 20

        self.train_loader, self.val_loader, self.test_loader = getDataloader(self.dataset, self.train_dir,\
            self.val_dir, self.test_dir, self.sigma, self.stride, self.workers, self.batch_size)
        #dataiter = iter(self.train_loader)
        #quit()
            
        model = handypose(self.dataset, num_classes=self.numClasses,backbone='resnet',output_stride=16,sync_bn=True,freeze_bn=False, stride=self.stride)
        #print(model)
        #quit()
        #dump_input = torch.rand((1, 3, 368, 368))
        #print(get_model_summary(model, dump_input))
        #quit()

        self.model       = model.cuda()

        self.criterion   = nn.MSELoss().cuda()

        self.optimizer   = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.best_model  = 12345678.9

        self.iters       = 0

        if self.args.pretrained is not None:
            print("Loading pretrained model file")
            checkpoint = torch.load(self.args.pretrained)
            p = checkpoint['state_dict']

            state_dict = self.model.state_dict()
            model_dict = {}

            for k,v in p.items():
                if k in state_dict and state_dict[k].size() == checkpoint['state_dict'][k].size():
                    model_dict[k] = v
                else :
                    print("Skipped this layer!!")
                    #print(k)
                    #print(v.shape)

            state_dict.update(model_dict)
            self.model.load_state_dict(state_dict)                      
            
            
            
        self.isBest = 0
        self.bestPCK01  = 0
        self.bestPCK02  = 0
        self.bestPCK03  = 0
        self.bestPCK04  = 0
        self.bestPCK05  = 0
        self.bestPCK06  = 0
        self.bestPCKh = 0
        #print(self.model)
        #quit()



    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        print("Epoch " + str(epoch) + ':') 
        tbar = tqdm(self.train_loader)

        for i, (input, heatmap, w) in enumerate(tbar):
            learning_rate = adjust_learning_rate(self.optimizer, self.iters, self.lr, policy='step',
                                                 gamma=self.gamma, step_size=self.step_size)

            input_var     =     input.cuda()
            heatmap_var   =    heatmap.cuda()
            #limbs_var     =   limbsmap.cuda()

            self.optimizer.zero_grad()

            heat = self.model(input_var)

            loss_heat   = self.criterion(heat,  heatmap_var)

            loss = loss_heat

            train_loss += loss_heat.item()

            loss.backward()
            self.optimizer.step()

            tbar.set_description('Train loss: %.6f' % (train_loss / ((i + 1)*self.batch_size)))

            self.iters += 1

            if i == 10000:
            	break

    def validation(self, epoch):
        self.model.eval()
        tbar = tqdm(self.val_loader, desc='\r')
        val_loss = 0.0
        
        AP    = np.zeros(self.numClasses+1)
        PCK01   = np.zeros(self.numClasses)
        PCK02   = np.zeros(self.numClasses)
        PCK03   = np.zeros(self.numClasses)
        PCK04   = np.zeros(self.numClasses)
        PCK05   = np.zeros(self.numClasses)
        PCK06   = np.zeros(self.numClasses)
        PCKh  = np.zeros(self.numClasses+1)
        count01 = np.zeros(self.numClasses+1)
        count02 = np.zeros(self.numClasses+1)
        count03 = np.zeros(self.numClasses+1)
        count04 = np.zeros(self.numClasses+1)
        count05 = np.zeros(self.numClasses+1)
        count06 = np.zeros(self.numClasses+1)

        cnt = 0
        for i, (input, heatmap, w) in enumerate(tbar):

            cnt += 1

            input_var     =      input.cuda()
            heatmap_var   =    heatmap.cuda()
            w = w.item()
            #print(np.shape(input_var))
            #print(np.shape(heatmap_var))
            #print(w)
            #quit()
            #limbs_var     =   limbsmap.cuda()

            self.optimizer.zero_grad()

            heat = self.model(input_var)
            loss_heat   = self.criterion(heat,  heatmap_var)

            loss = loss_heat

            val_loss += loss_heat.item()

            tbar.set_description('Val   loss: %.6f' % (val_loss / ((i + 1)*self.batch_size)))

            acc, acc_PCK01, acc_PCKh, cnt, pred, visible01 = evaluate.accuracy(heat.detach().cpu().numpy(), heatmap_var.detach().cpu().numpy(),0.01,0.005, self.dataset, w)  
            acc, acc_PCK02, acc_PCKh, cnt, pred, visible02 = evaluate.accuracy(heat.detach().cpu().numpy(), heatmap_var.detach().cpu().numpy(),2,0.005, self.dataset, w)
            acc, acc_PCK03, acc_PCKh, cnt, pred, visible03 = evaluate.accuracy(heat.detach().cpu().numpy(), heatmap_var.detach().cpu().numpy(),0.03,0.005, self.dataset, w)
            acc, acc_PCK04, acc_PCKh, cnt, pred, visible04 = evaluate.accuracy(heat.detach().cpu().numpy(), heatmap_var.detach().cpu().numpy(),0.04,0.005, self.dataset, w)
            acc, acc_PCK05, acc_PCKh, cnt, pred, visible05 = evaluate.accuracy(heat.detach().cpu().numpy(), heatmap_var.detach().cpu().numpy(),0.05,0.005, self.dataset, w)
            acc, acc_PCK06, acc_PCKh, cnt, pred, visible06 = evaluate.accuracy(heat.detach().cpu().numpy(), heatmap_var.detach().cpu().numpy(),0.06,0.005, self.dataset, w)
            

            for j in range(self.numClasses):
                if visible01[j] == 1:
                    PCK01[j]    = (PCK01[j] *count01[j] + acc_PCK01[j])  / (count01[j] + 1)
                    count01[j] += 1
            mPCK01    =  PCK01[0:].sum()/(self.numClasses)
            
            for j in range(self.numClasses):
                if visible02[j] == 1:
                    PCK02[j]    = (PCK02[j] *count02[j] + acc_PCK02[j])  / (count02[j] + 1)
                    count02[j] += 1
            mPCK02    =  PCK02[0:].sum()/(self.numClasses)

            for j in range(self.numClasses):
                if visible03[j] == 1:
                    PCK03[j]    = (PCK03[j] *count03[j] + acc_PCK03[j])  / (count03[j] + 1)
                    count03[j] += 1
            mPCK03    =  PCK03[0:].sum()/(self.numClasses)

            for j in range(self.numClasses):
                if visible04[j] == 1:
                    PCK04[j]    = (PCK04[j] *count04[j] + acc_PCK04[j])  / (count04[j] + 1)
                    count04[j] += 1
            mPCK04    =  PCK04[0:].sum()/(self.numClasses)

            for j in range(self.numClasses):
                if visible05[j] == 1:
                    PCK05[j]    = (PCK05[j] *count05[j] + acc_PCK05[j])  / (count05[j] + 1)
                    count05[j] += 1
            mPCK05    =  PCK05[0:].sum()/(self.numClasses)

            for j in range(self.numClasses):
                if visible06[j] == 1:
                    PCK06[j]    = (PCK06[j] *count06[j] + acc_PCK06[j])  / (count06[j] + 1)
                    count06[j] += 1
            mPCK06    =  PCK06[0:].sum()/(self.numClasses)
            
        #torch.save({'state_dict': self.model.state_dict()}, 'Vehicle_model1' + '.pth.tar')

        if mPCK02 > self.bestPCK02:
            self.isBest = mPCK02
            #save_checkpoint({'state_dict': self.model.state_dict()}, self.isBest, self.args.model_name)
            #print("Model saved to "+self.args.model_name)

        if mPCK01 > self.bestPCK01:
            self.bestPCK01 = mPCK01
        if mPCK02 > self.bestPCK02:
            self.bestPCK02 = mPCK02
        if mPCK03 > self.bestPCK03:
            self.bestPCK03 = mPCK03
        if mPCK04 > self.bestPCK04:
            self.bestPCK04 = mPCK04
        if mPCK05 > self.bestPCK05:
            self.bestPCK05 = mPCK05
        if mPCK06 > self.bestPCK06:
            self.bestPCK06 = mPCK06

        print("PCK01 = %2.2f%%; PCK02 = %2.2f%%; PCK03 = %2.2f%%; PCK04 = %2.2f%%; PCK05 = %2.2f%%; PCK06 = %2.2f%%" % (self.bestPCK01*100,self.bestPCK02*100,self.bestPCK03*100,self.bestPCK04*100,self.bestPCK05*100,self.bestPCK06*100))



    def test(self,epoch):
        self.model.eval()
        print("Testing") 

        for idx in range(0,95):
            print(idx,"/",2000)
            img_path = 'Logs/CMUPanoptic/sample'+str(idx)+'.jpg'
            #img_path = 'Logs/sample9.jpg'

            center   = [184, 184]

            img  = np.array(cv2.resize(cv2.imread(img_path),(368,368)), dtype=np.float32)
            img  = img.transpose(2, 0, 1)
            img  = torch.from_numpy(img)
            mean = [128.0, 128.0, 128.0]
            std  = [256.0, 256.0, 256.0]
            for t, m, s in zip(img, mean, std):
                t.sub_(m).div_(s)

            img       = torch.unsqueeze(img, 0)

            self.model.eval()

            input_var   = img.cuda()

            heat = self.model(input_var)

            heat = F.interpolate(heat, size=input_var.size()[2:], mode='bilinear', align_corners=True)

            kpts = get_kpts(heat, img_h=368.0, img_w=368.0)
            draw_paint(img_path, kpts, idx, epoch, self.model_arch, self.dataset)

            heat = heat.detach().cpu().numpy()

            heat = heat[0].transpose(1,2,0)


            for i in range(heat.shape[0]):
                for j in range(heat.shape[1]):
                    for k in range(heat.shape[2]):
                        if heat[i,j,k] < 0:
                            heat[i,j,k] = 0
                        

            im = cv2.resize(cv2.imread(img_path),(368,368))

            heatmap = []
            for i in range(self.numClasses):
                heatmap = cv2.applyColorMap(np.uint8(255*heat[:,:,i]), cv2.COLORMAP_JET)
                im_heat  = cv2.addWeighted(im, 0.6, heatmap, 0.4, 0)
                cv2.imwrite('samples/heat/handypose'+str(i)+'.png', im_heat)
            #cv2.imwrite('samples/heat/handypose'+str(i)+'.png', im_heat)
        
parser = argparse.ArgumentParser()
# Hand arguments - CMU
#parser.add_argument('--pretrained', default='/home/dg9679/bm3768/models/UniPose_Modified/pretrained_weights/UniPose_LSP.tar',type=str)
#parser.add_argument('--pretrained', default='/home/dg9679/bm3768/models/UniPose_Modified/checkpoint/Results/Multi-level_waspv2/MLFWaspv2/CMUHand2D_Resnet_101_best.pth.tar',type=str)
#parser.add_argument('--dataset', type=str, default='CMUHand')
#parser.add_argument('--train_dir', default='/home/dg9679/bm3768/dataset/CMUHand',type=str)
#parser.add_argument('--val_dir', type=str, default='/home/dg9679/bm3768/dataset/CMUHand')
#parser.add_argument('--test_dir', type=str, default='/home/dg9679/bm3768/dataset/CMUHand')
#parser.add_argument('--model_name', default='checkpoint/Results/Multi-level_waspv2/MLFWaspv2/CMUHand2DMLF_Resnet_101', type=str)
#parser.add_argument('--model_name', default='checkpoint/Results/Original_waspv1/CMUHand2D_Resnet_101', type=str)
#parser.add_argument('--model_arch', default='handypose', type=str)

# Hand arguments - MPII + NZSL
#parser.add_argument('--pretrained', default=None,type=str)
#parser.add_argument('--pretrained', default='/home/dg9679/bm3768/models/UniPose_Modified/checkpoint/Results/MPII_NZSL/Multi-level_waspv2/MLFWaspv2/MPII_NZSLHand2D_Resnet_101_best.pth.tar',type=str)
#parser.add_argument('--dataset', type=str, default='MPII_NZSLHand')
#parser.add_argument('--train_dir', default='/home/dg9679/bm3768/dataset/MPII_NZSLHand',type=str)
#parser.add_argument('--val_dir', type=str, default='/home/dg9679/bm3768/dataset/MPII_NZSLHand')
#parser.add_argument('--test_dir', type=str, default='/home/dg9679/bm3768/dataset/MPII_NZSLHand')
#parser.add_argument('--model_name', default='checkpoint/Results/MPII_NZSL/Multi-level_waspv2/MLFWaspv2/MPII_NZSLHand2D_Resnet_101', type=str)
#parser.add_argument('--model_arch', default='handypose', type=str)

# Vehicle arguments
#parser.add_argument('--pretrained', default='/home/dg9679/bm3768/models/UniPose_Modified/pretrained_weights/UniPose_LSP.tar',type=str)
parser.add_argument('--pretrained', default='/home/dg9679/bm3768/models/UniPose_Modified/checkpoint/VeRiModifiedVehicle1_2D_Resnet_101_best.pth.tar',type=str)
parser.add_argument('--dataset', type=str, default='VeRi')
parser.add_argument('--train_dir', default='/home/dg9679/bm3768/VeRi',type=str)
parser.add_argument('--val_dir', type=str, default='/home/dg9679/bm3768/VeRi')
parser.add_argument('--test_dir', type=str, default='/home/dg9679/bm3768/VeRi')
parser.add_argument('--model_name', default='checkpoint/Results/VehiPose/VeRiVehicle_2D_Resnet_101', type=str)
parser.add_argument('--model_arch', default='handypose', type=str)


starter_epoch =    0
epochs        =  80

args = parser.parse_args()

if args.dataset == 'LSP':
    args.train_dir  = '/PATH/TO/LSP/TRAIN'
    args.val_dir    = '/PATH/TO/LSP/VAL'
    args.pretrained = '/PATH/TO/WEIGHTS'
elif args.dataset == 'MPII':
    args.train_dir  = '/PATH/TO/MPIII/TRAIN'
    args.val_dir    = '/PATH/TO/MPIII/VAL'
elif args.dataset == 'CMUHand':
    args.train_dir = '/home/dg9679/bm3768/dataset/CMUHand'
    args.val_dir = '/home/dg9679/bm3768/dataset/CMUHand'
    args.test_dir = '/home/dg9679/bm3768/dataset/CMUHand'
elif args.dataset == 'MPII_NZSLHand':
    args.train_dir = '/home/dg9679/bm3768/dataset/MPII_NZSLHand'
    args.val_dir = '/home/dg9679/bm3768/dataset/MPII_NZSLHand'
    args.test_dir = '/home/dg9679/bm3768/dataset/MPII_NZSLHand'
elif args.dataset == 'VeRi':
    args.train_dir = '/home/dg9679/bm3768/VeRi'
    args.val_dir = '/home/dg9679/bm3768/VeRi'
    args.test_dir = '/home/dg9679/bm3768/VeRi'
    

trainer = Trainer(args)
for epoch in range(starter_epoch, epochs):
    #trainer.training(epoch)
    trainer.validation(epoch)
    #trainer.test(epoch)
    quit()
