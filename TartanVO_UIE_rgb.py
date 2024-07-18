# Software License Agreement (BSD License)
#
# Copyright (c) 2020, Wenshan Wang, Yaoyu Hu,  CMU
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of CMU nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn as nn
import numpy as np
import time

np.set_printoptions(precision=4, suppress=True, threshold=10000)

from Network.UIEVONetRGB import UIEVONetRGB
from Network.UIERGB import PhysicalNNRGB

import torch.utils.tensorboard as tb

tb_logger = tb.SummaryWriter(log_dir="./logs/")



class UIETartanVORGB(object):
    def __init__(self, model_name, uie_model_name):
        # import ipdb;ipdb.set_trace()
        self.vonet = UIEVONetRGB()
        self.uienet = PhysicalNNRGB()

        # load the whole model
        if model_name.endswith('.pkl'):
            modelname = 'models/' + model_name
            # modelname = 'models/wflow_seq7/' + model_name
            self.load_model(self.vonet, modelname)

        
        self.uienet = torch.nn.DataParallel(self.uienet)
        self.load_model_uie(uie_model_name)
        self.uienet = self.uienet.module

        self.vonet.cuda()
        self.uienet.cuda()

        self.test_count = 0
        self.pose_std = np.array([ 0.13,  0.13,  0.13,  0.013 ,  0.013,  0.013], dtype=np.float32) # the output scale factor
        self.flow_norm = 20 # scale factor for flow

    def load_model(self, model, modelname):
        preTrainDict = torch.load(modelname)
        model_dict = model.state_dict()
        preTrainDictTemp = {k:v for k,v in preTrainDict.items() if k in model_dict}

        if( 0 == len(preTrainDictTemp) ):
            print("Does not find any module to load. Try DataParallel version.")
            for k, v in preTrainDict.items():
                kk = k[7:]
                if ( kk in model_dict ):
                    preTrainDictTemp[kk] = v

        if ( 0 == len(preTrainDictTemp) ):
            raise Exception("Could not load model from %s." % (modelname), "load_model")

        model_dict.update(preTrainDictTemp)
        model.load_state_dict(model_dict)
        print('Model loaded...')
        return model

    def load_model_uie(self, model_name):
        checkpoint = torch.load(f'models/{model_name}')
        self.uienet.load_state_dict(checkpoint['state_dict'])
        # for param in self.uienet.ANet.parameters():
        #     param.requires_grad = False
        # for param in self.uienet.TNet.parameters():
        #     param.requires_grad = False

    def test_batch(self, sample):
        self.test_count += 1
        
        # import ipdb;ipdb.set_trace()
        img0   = sample['img1'].cuda()
        img1   = sample['img2'].cuda()
        intrinsic = sample['intrinsic'].cuda()
        inputs = [img0, img1, intrinsic]

        self.vonet.eval()

        with torch.no_grad():
            starttime = time.time()
            
            # img0_gray = torch.einsum('nhwc,nc->nhw', img0[:,:3,:,:].permute(0,2,3,1), torch.tensor([0.2989, 0.5870, 0.1140], device='cuda')[None, :]) #torch.dot(img0[0,:3,:,:].squeeze(), torch.tensor([0.2989, 0.5870, 0.1140], device='cuda'))
            # out_A = self.uienet.ANet(img0_gray)
            # out_T = self.uienet.tNet(torch.cat([img0_gray[:,None,:,:]*0+out_A, img0_gray[:,None,:,:]], 1))
            #breakpoint()

            out_A = self.uienet.ANet(img0[:,:,:,:])
            out_T = self.uienet.tNet(torch.cat([img0[:,:,:,:]*0+out_A, img0[:,:,:,:]], 1))

            
            #breakpoint()
            # out_T -= out_T.min()
            # out_T /= out_T.max()
            #out_T += 0.5

            flow, pose = self.vonet(inputs, out_T)
            inferencetime = time.time()-starttime
            # import ipdb;ipdb.set_trace()
            posenp = pose.data.cpu().numpy()
            posenp = posenp * self.pose_std # The output is normalized during training, now scale it back
            flownp = flow.data.cpu().numpy()
            flownp = flownp * self.flow_norm

        # calculate scale from GT posefile
        if 'motion' in sample:
            motions_gt = sample['motion']
            scale = np.linalg.norm(motions_gt[:,:3], axis=1)
            trans_est = posenp[:,:3]
            trans_est = trans_est/np.linalg.norm(trans_est,axis=1).reshape(-1,1)*scale.reshape(-1,1)
            posenp[:,:3] = trans_est 
        else:
            print('    scale is not given, using 1 as the default scale value..')

        print("{} Pose inference using {}s: \n{}".format(self.test_count, inferencetime, posenp))
        return posenp, flownp

    def upToScaleLossFunc(self, pose_est, pose_truth):
        e = torch.tensor(1e-6)
        trans_est = pose_est[:,:3]
        rot_est = pose_est[:,3:]
        trans_truth = pose_truth[:,:3]
        rot_truth = pose_truth[:,3:]
        trans_est_norm = torch.linalg.norm(trans_est)
        trans_truth_norm = torch.linalg.norm(trans_truth)

        trans_loss = torch.linalg.norm(trans_est/torch.max(trans_est_norm, e) - trans_truth/torch.max(trans_truth_norm, e))
        rot_loss = torch.linalg.norm(rot_est - rot_truth)
        return trans_loss + rot_loss, trans_loss, rot_loss

    def train(self, data_loader, optimizer, num_epochs, dataset_size, saved_path):
        criterion = nn.L1Loss()
        pose_std = torch.from_numpy(self.pose_std).unsqueeze(0).cuda()
        print(pose_std)
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            self.vonet.train()

            running_loss = 0.0
            running_loss_trans = 0.0
            running_loss_rot = 0.0
            # Iterate over data.
            for bi, sample in enumerate(data_loader):

                img0   = sample['img1'].cuda()
                img1   = sample['img2'].cuda()
                intrinsic = sample['intrinsic'].cuda()
                label = sample['motion'].cuda()
                inputs = [img0, img1, intrinsic]

                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    with torch.no_grad():
                        out_A = self.uienet.ANet(img0[:,:,:,:])
                        out_T = self.uienet.tNet(torch.cat([img0[:,:,:,:]*0+out_A, img0[:,:,:,:]], 1))

                    flow, pose = self.vonet(inputs, out_T)
                    pose = torch.mul(pose, pose_std)
                    
                    # breakpoint()
                    
                    # print("pose", pose[:,:3])
                    # print("GT", label)
                    # loss = criterion(pose, label)
                    loss, trans_loss, rot_loss = self.upToScaleLossFunc(pose_est=pose, pose_truth=label)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_loss_trans += trans_loss.item()
                running_loss_rot += rot_loss.item()

            epoch_loss = running_loss / dataset_size
            epoch_loss_trans = running_loss_trans / dataset_size
            epoch_loss_rot = running_loss_rot / dataset_size
            print('Loss: {:.4f} - Trans loss: {:.4f} - Rot loss: {:.4f}'.format(epoch_loss,epoch_loss_trans,epoch_loss_rot))

            tb_logger.add_scalars("train", {"total_loss": epoch_loss, "trans_loss":epoch_loss_trans, "rot_loss":epoch_loss_rot}, epoch)

            #if epoch == 0 or (epoch+1) % 10 == 0 or (epoch+1)==num_epochs :
            torch.save(self.vonet.state_dict(), f"{saved_path}/tartanvo_wflow_{epoch}.pkl") 
        # return model
