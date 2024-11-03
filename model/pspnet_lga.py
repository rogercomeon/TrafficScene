import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18,resnet34,resnet50,resnet101
import os

from torch.autograd import Variable
def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()

        # 将初始化好的权重移到 GPU 上
        model.cuda()
    
up_kwargs = {'mode': 'bilinear', 'align_corners': True}



class SelfAttention(nn.Module):
    def __init__(self, Cv,Ck,h,w):
        super(SelfAttention, self).__init__()

        self.Ck = Ck
        self.Cv = Cv
        self.qconv = nn.Conv2d(self.Cv,self.Ck,1, bias=False)
        self.kconv = nn.Conv2d(self.Cv,self.Ck,1, bias=False)
        self.vconv = nn.Conv2d(self.Cv,self.Cv,1, bias=False)

        self.dropout = nn.Dropout(0.1)
        # 注意力分数归一化的比例系数
        # self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)
        initialize_weights(self)

    def forward(self, x):
        """
        x: 输入的特征矩阵，维度为(batch_size, seq_len, hidden_dim)
        """
        batch_size, Cv, h ,w   = x.size()
        Q = self.qconv(x).permute(0,2,3,1).contiguous().view(batch_size,-1,self.Ck)
        K = self.kconv(x).permute(0,2,3,1).contiguous().view(batch_size,-1,self.Ck)
        V = self.vconv(x).permute(0,2,3,1).contiguous().view(batch_size,-1,self.Cv)
        # 计算注意力分数（内积）
        scores = torch.matmul(Q, K.transpose(1, 2))  ##(b,N,N)
        #Q b,N,Ck   K b,N,Ck     score =  b,N,N
        attn_weights = torch.softmax(scores, dim=-1)  
        # 对注意力权重进行dropout
        # attn_weights = self.dropout(attn_weights)
        # 将注意力权重与Value相乘，得到self-attention后的表示
        attn_output = torch.matmul(attn_weights, V)  # （b,N,cv）
        return attn_output, Q

    def forward(self, x):
        """
        x: 输入的特征矩阵，维度为(batch_size, seq_len, hidden_dim)
        """
        # 获取batch_size和seq_len
        batch_size, Cv, h,w   = x.size()
        # 得到Query, Key, Value
        K = self.kconv(x).permute(0,2,3,1).contiguous().view(batch_size,-1,self.Ck)
        V = self.vconv(x).permute(0,2,3,1).contiguous().view(batch_size,-1,self.Cv)
        return K,V

class encoder(nn.Module):
    def __init__(self,C=3,Cv=3,bias=False):
        super(encoder,self).__init__()
        # self.enconder1 = nn.Linear(in_features=C,out_features=Cv,bias=False)
        self.enconder1 = nn.Conv2d(C,Cv,1,bias=False)
        # initialize_weights(self)
    def forward(self,x):
        return self.enconder1(x)



class BatchNorm2d(nn.BatchNorm2d):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, num_features, activation='none'):
        super(BatchNorm2d, self).__init__(num_features=num_features)
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'none':
            self.activation = lambda x:x
        else:
            raise Exception("Accepted activation: ['leaky_relu']")

    def forward(self, x):
        return self.activation(super(BatchNorm2d, self).forward(x))


up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class pspne_lga(nn.Module):

    def __init__(self,
            nclass=15,
            criterion=nn.CrossEntropyLoss(ignore_index=255),
            norm_layer=nn.BatchNorm2d,
            backbone='resnet50',
            dilated=True,
            aux=True,
            multi_grid=True,
            model_path=None,
            Cv=256,
            Ck = 256,
            h=68,
            w = 90
        ):
        super(pspne_lga, self).__init__()
        self.psp_path = model_path
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        self.nclass = nclass
        self.criterion = criterion
        # copying modules from pretrained models
        self.backbone = backbone
        self.Cv = Cv
        if backbone == 'resnet18':
            self.pretrained = resnet18(dilated=dilated, multi_grid=multi_grid,
                                               deep_base=False)
            self.pretrained1 = resnet18(dilated=dilated, multi_grid=multi_grid,
                                               deep_base=False)
            self.expansion = 1
        elif backbone == 'resnet34':
            self.pretrained = resnet34(dilated=dilated, multi_grid=multi_grid,
                                               deep_base=False)
            self.pretrained1 = resnet34(dilated=dilated,multi_grid=multi_grid,
                                        norm_layer=norm_layer)
            self.expansion = 1
        elif backbone == 'resnet50':
            self.pretrained = resnet50(dilated=dilated,multi_grid=multi_grid,
                                              norm_layer=norm_layer)
            self.pretrained1 = resnet18(dilated=dilated,multi_grid=multi_grid,
                                               norm_layer=norm_layer)
            self.expansion = 4
        elif backbone == 'resnet101':
            self.pretrained = resnet101(dilated=dilated,multi_grid=multi_grid,
                                               norm_layer=norm_layer)
            self.pretrained1 = resnet101(dilated=dilated,multi_grid=multi_grid,
                                               norm_layer=norm_layer)
            self.expansion = 4
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options

        self.head1 = PSPHead(512*self.expansion+self.Cv, nclass, norm_layer, self._up_kwargs)  
        self.head2 = PSPHead(512*self.expansion+self.Cv, nclass, norm_layer, self._up_kwargs)        
        self.head3 = PSPHead(512*self.expansion+self.Cv, nclass, norm_layer, self._up_kwargs)        
        self.head4 = PSPHead(512*self.expansion+self.Cv, nclass, norm_layer, self._up_kwargs)        
        self.head5 = PSPHead(512*self.expansion+self.Cv, nclass, norm_layer, self._up_kwargs)        
        self.head6 = PSPHead(512*self.expansion+self.Cv, nclass, norm_layer, self._up_kwargs)   
        self.head7 = PSPHead(512*self.expansion+self.Cv, nclass, norm_layer, self._up_kwargs)        
        self.head8 = PSPHead(512*self.expansion+self.Cv, nclass, norm_layer, self._up_kwargs)        
        self.head9 = PSPHead(512*self.expansion+self.Cv, nclass, norm_layer, self._up_kwargs)        
                
        self.auxlayer = FCNHead(256*self.expansion,nclass, norm_layer, self._up_kwargs) #这里输入的网络层可根据真实情况修改
        self.auxlayer1 = FCNHead(256,nclass, norm_layer, self._up_kwargs) #这里输入的网络层可根据真实情况修改
        
        self.pretrained_mp_load()
        self.layer0 = nn.Sequential(self.pretrained.conv1,self.pretrained.bn1,self.pretrained.relu,self.pretrained.maxpool)   
                
        self.SelfAttention = SelfAttention(Cv,Ck,h,w)
        self.encoder1 = encoder(2048*9,256)
        self.encoder2 = encoder(4096,2048)
        self.encoder3 = encoder(4096,Cv)
        initialize_weights(self.encoder1)
        initialize_weights(self.encoder2)
    def forward(self, img_all,y=None):
        b, _, h, w = img_all[0].size()
        img_all_other = img_all.copy()
        img_all1 = [self.pretrained.conv1(i) for i in img_all_other]
        img_all1 = [self.pretrained.bn1(i) for i in img_all1]
        img_all1 = [self.pretrained.relu(i) for i in img_all1]
        img_all1 = [self.pretrained.maxpool(i) for i in img_all1]
        other1 = [self.pretrained.layer1(i) for i in img_all1]
        other2 = [self.pretrained.layer2(i) for i in other1]
        other3 = [self.pretrained.layer3(i) for i in other2]
        other4 = [self.pretrained.layer4(i) for i in other3]
        _,_,h1,w1 = other4[0].shape
        allimg = torch.cat(other4,dim=1)
        fuse_img  = self.encoder1(allimg)
        att_img,Q = self.SelfAttention(fuse_img)
        Fself = att_img.permute(0,2,1).contiguous().view(b,self.Cv,h1,w1)
        other4 = [torch.cat((i,Fself),dim=1)   for i in  other4]
        x =[]
        x.append(self.head1(other4[0]))
        x.append(self.head2(other4[1]))
        x.append(self.head3(other4[2]))
        x.append(self.head4(other4[3]))
        x.append(self.head5(other4[4]))
        x.append(self.head6(other4[5]))
        x.append(self.head7(other4[6]))
        x.append(self.head8(other4[7]))
        x.append(self.head9(other4[8]))

        x = [F.interpolate(i, (h,w), **self._up_kwargs) for i in x]

        if self.training:       
            main_loss = [self.criterion(x[i], y[i]) for i in range(9) ]
            return x, main_loss

        return x        
 

    def pretrained_mp_load(self):
        if self.psp_path is not None:
            if os.path.isfile(self.psp_path):
                print("Loading pretrained model from '{}'".format(self.psp_path))
                model_state = torch.load(self.psp_path)
                self.load_state_dict(model_state, strict=True)
            else:
                print("No pretrained found at '{}'".format(self.psp_path))


class PSPHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(PSPHead, self).__init__()
        inter_channels = in_channels // 4

        self.norm_layer = norm_layer
        self.conv5 = nn.Sequential(PyramidPooling(in_channels, norm_layer, up_kwargs),
                                   nn.Conv2d(in_channels * 2, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))
    def forward(self, x):
        return self.conv5(x)


class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.norm_layer = norm_layer
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)



class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs={}, with_global=False):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self._up_kwargs = up_kwargs
        self.norm_layer = norm_layer
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)
