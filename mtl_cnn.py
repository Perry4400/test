# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 14:26:16 2022

@author: admin
"""

import torch
import torch.nn as nn

lenf = 101

def make_model(args):
    return MultiTaskLossWrapper(2,args)

# 卷积层
class conv(nn.Module):
    def __init__(self, in_channels, out_channels, args, flow):
        super(conv, self).__init__()
        if args.ABCD[1] == '0':
            self.dp = 0
        elif args.ABCD[1] == '1':
            self.dp = 0.3
        elif args.ABCD[1] == '2':
            self.dp = 0.5
        self.flow = flow
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,
                              kernel_size=3,stride=1,padding=1,bias=False,groups=in_channels)
        self.pconv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                              kernel_size=1,stride=1,padding=0,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if self.flow == 'M':
            self.prelu = nn.PReLU(in_channels)
        else:
            self.prelu = nn.PReLU(out_channels)
        self.dropout = nn.Dropout2d(p=self.dp)
        
    def forward(self,x):
        if self.flow == 'M':
            out = self.bn(self.pconv(self.conv(self.dropout(self.prelu(x)))))
        else:
            out = self.dropout(self.prelu(self.bn(self.pconv(self.conv(x)))))
        return out
    
class nconv(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(nconv, self).__init__()
        if args.ABCD[1] == '0':
            self.dp = 0
        elif args.ABCD[1] == '1':
            self.dp = 0.3
        elif args.ABCD[1] == '2':
            self.dp = 0.5
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                              kernel_size=3,stride=1,padding=1,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)
        self.dropout = nn.Dropout2d(p=self.dp)
        
    def forward(self,x):
        out = self.dropout(self.prelu(self.bn(self.conv(x))))
        return out

# 中间层
class middle_block(nn.Module):
    def __init__(self,args):
        super(middle_block,self).__init__()
        self.conv = conv
        #中间层
        self.conv1 = self.conv(512,512,args=args,flow='M')
        self.conv2 = self.conv(512,512,args=args,flow='M')
        self.conv3 = self.conv(512,512,args=args,flow='M')
        
    def forward(self,x):
        residual = x
        out = residual + self.conv3(self.conv2(self.conv1(x)))
        return out
    
# model
class MTL_CNN(nn.Module):
    def __init__(self,input_channel,args):
        super(MTL_CNN,self).__init__()
        #初始化
        self.conv = conv
        self.nconv = nconv
        self.block = middle_block(args=args)
        
        #输入层
        self.nconv_E1 = self.nconv(input_channel,256,args=args)#52代表输入数据的通道数
        self.nconv_E11 = self.nconv(256,256,args=args)
        self.nconv_E2 = self.nconv(256,384,args=args)
        self.nconv_E22 = self.nconv(384,384,args=args)
        self.conv_E3 = self.conv(384,512,args=args,flow='E')
        self.conv_E33 = self.conv(512,512,args=args,flow='E')
        self.nconv_E3 = self.nconv(384,512,args=args)
        
        #中间层
        self.stage1 = self.make_layer(self.block)
        self.conv_M2 = self.conv(512,640,args=args,flow='M')
        self.conv_M22 = self.conv(640,640,args=args,flow='M')
        
        #输出层
        self.conv_EX1r = self.conv(640,768,args=args,flow='EX')
        self.conv_EX2r = self.conv(768,1024,args=args,flow='EX')
        self.avgr = nn.AdaptiveAvgPool2d(1)
        self.fcr = nn.Linear(1024,1)
        
        self.conv_EX1d = self.conv(640,768,args=args,flow='EX')
        self.conv_EX2d = self.conv(768,1024,args=args,flow='EX')
        self.avgd = nn.AdaptiveAvgPool2d(1)
        self.fcd = nn.Linear(1024,1)
        
    def forward(self, x):
        #输入层
        out = self.nconv_E22(self.nconv_E2(self.nconv_E11(self.nconv_E1(x))))
        residual = self.nconv_E3(out)
        out = self.conv_E33(self.conv_E3(out))
        out = out + residual
        
        #中间层
        out = self.conv_M22(self.conv_M2(self.stage1(out)))
        
        #输出层
        out_r = self.avgr(self.conv_EX2r(self.conv_EX1r(out)))
        r = self.fcr(out_r.view(out_r.size(0),out_r.size(1))).view(-1)
        out_d = self.avgd(self.conv_EX2d(self.conv_EX1d(out)))
        d = self.fcd(out_d.view(out_d.size(0),out_d.size(1))).view(-1)
        return [r,d]
         
    def make_layer(self,block):
        block_list = []
        for i in range(0,7):
            block_list.append(block)
            
        return nn.Sequential(*block_list)
    
# 多任务损失层
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num, args):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = MTL_CNN(input_channel=args.lenf*2,args=args)
        self.task_num = task_num
        self.log_vars2 = nn.Parameter(torch.zeros((task_num)))

    def forward(self, input, targets):

        output = self.model(input)

        precision1 = torch.exp(-self.log_vars2[0])
        loss = torch.sum(0.5 * precision1 * (targets[0] - output[0]) ** 2., -1)+ 0.5*self.log_vars2[0]

        precision2 = torch.exp(-self.log_vars2[1])
        loss += torch.sum(0.5 * precision2 * (targets[1] - output[1]) ** 2., -1) + 0.5*self.log_vars2[1]

        return loss, self.log_vars2.data.tolist(), output
    
        #调试代码
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Radar Target Detection Based on Deep Neural Networks')
    parser.add_argument('--dp', type=float, default=0.1, help='丢弃率')
    parser.add_argument('--lenf', type=int, default=151, help='length of freqvec')
    parser.add_argument('--ABCD', type=str, default='11', help='丢弃率')
    args = parser.parse_args()
    
    mtl = MultiTaskLossWrapper(2, args=args)
    inputs = torch.randn(5,args.lenf*2,18,18)
    r = torch.rand(5)
    z = torch.rand(5)
    targets = [r,z]
    loss, log_vars2, output = mtl(inputs,[r,z])
    print(loss, log_vars2, output )