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
import torch.nn.functional as F
from .PWC import PWCDCNet as FlowNet
from .VOFlowNet import VOFlowRes as FlowPoseNet
from torchvision.transforms.functional import rgb_to_grayscale

from torchvision.utils import flow_to_image
from PIL import Image

import numpy as np

class UIEVONetRGB(nn.Module):
    def __init__(self):
        super(UIEVONetRGB, self).__init__()

        self.flowNet     = FlowNet()
        self.flowPoseNet = FlowPoseNet()
        self.counter = 0

    def forward(self, x, out_T):
        # import ipdb;ipdb.set_trace()
        with torch.no_grad():
            flow = self.flowNet(x[0:2])
        flow_ = flow
        out_T_resized = F.interpolate(out_T, size=(flow.shape[2], flow.shape[3]), mode='area')
        
        out_T_resized -= out_T_resized.min()
        out_T_resized /= out_T_resized.max()

        out_T_resized = 1 - out_T_resized
        # tmp_max = out_T_resized.max()
        alpha= 1 #0.25#0.5 #2
        beta=4 #4
        out_T_resized = alpha*out_T_resized
        out_T_resized = out_T_resized + 1 - out_T_resized.max()/beta #2
        out_T_resized = rgb_to_grayscale(out_T_resized)

        # out_T_resized -= out_T_resized.min()
        # out_T_resized /= out_T_resized.max()
        # out_T_resized = (1-out_T_resized) + 0.7 #1.2

        #out_T_resized[out_T_resized<0.3] = -1
        #out_T_resized[out_T_resized>=0.3] = 0
        #out_T_resized=out_T_resized+1
        # breakpoint()
        flow = flow * out_T_resized
        # flow = flow * (1/out_T_resized )
        # if self.counter == 356:
        print(out_T_resized.min(), out_T_resized.max())
        breakpoint()

        # tmp_ = flow_to_image(flow_).squeeze().permute(1,2,0).detach().cpu()
        # tmp = flow_to_image(flow).squeeze().permute(1,2,0).detach().cpu()
        # tmp_T = out_T.squeeze().cpu()
        # tmp_T_ = tmp_T - tmp_T.min()
        # tmp_T_ = (tmp_T_ / tmp_T_.max() ) * 255
        # # breakpoint()
        # Image.fromarray(tmp_.numpy()).save(f"tmp/flow/{self.counter}.png")
        # Image.fromarray(tmp.numpy()).save(f"tmp/wflow/{self.counter}.png")
        # Image.fromarray(tmp_T_.numpy()).convert("L").save(f"tmp/T/{self.counter}.png")
        # self.counter += 1

        # tmp_ = flow_to_image(flow_).squeeze().permute(1,2,0).detach().cpu()
        # tmp = flow_to_image(flow).squeeze().permute(1,2,0).detach().cpu()
        # tmp_T_inv = out_T.squeeze().cpu()
        # tmp_T = 1-tmp_T_inv

        # tmp_T_ = tmp_T - tmp_T.min()
        # tmp_T_ = (tmp_T_ / tmp_T_.max() ) * 255
        # tmp_T_inv_ = tmp_T_inv - tmp_T_inv.min()
        # tmp_T_inv_ = (tmp_T_inv_ / tmp_T_inv_.max() ) * 255

        # breakpoint()
        # Image.fromarray(tmp_.numpy()).save(f"tmp/flow/{self.counter}.png")
        # Image.fromarray(tmp.numpy()).save(f"tmp/wflow/{self.counter}.png")
        # Image.fromarray((tmp_T_inv_.numpy().transpose(1,2,0)*255).astype(np.uint8)).convert("L").save(f"tmp/T_inv/{self.counter}.png")
        # Image.fromarray((tmp_T_.numpy().transpose(1,2,0)*255).astype(np.uint8)).convert("L").save(f"tmp/T/{self.counter}.png")
        # self.counter += 1

        flow_input = torch.cat( ( flow, x[2] ), dim=1 )        
        pose = self.flowPoseNet( flow_input )

        return flow, pose

