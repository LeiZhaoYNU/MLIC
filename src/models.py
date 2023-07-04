import torchvision.models as models
from mobilenetv2 import *
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
import time 

def feats_pooling(x, method='avg', sh=8, sw=8):
    if method == 'avg':
       x = F.avg_pool2d(x, (sh, sw))
    if method == 'max':
       x = F.max_pool2d(x, (sh, sw))
    if method == 'gwp':
       x1 = F.max_pool2d(x, (sh, sw))
       x2 = F.avg_pool2d(x, (sh, sw))
       x = (x1 + x2)/2
    return x

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MCARMobilenetv2(nn.Module):
    def __init__(self, model, num_classes, ps, topN=4, threshold=0.5, vis=False):
        super(MCARMobilenetv2, self).__init__()
        self.features = nn.Sequential(
            model.features,
            model.conv         
        )
        num_features = model.conv[0].out_channels
        self.convclass = nn.Conv2d(num_features, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.ps = ps
        self.num_classes = num_classes
        self.num_features = num_features
        self.topN = topN
        self.threshold = threshold
        self.vis = vis
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, inputs):
        # global stream
        b, c, h, w = inputs.size()
        ga = self.features(inputs)          # activation map 
        gf = feats_pooling(ga, method=self.ps, sh=int(h/32), sw=int(w/32))
        gs = self.convclass(gf)             # bxcx1x1
        gs = torch.sigmoid(gs)              # bxcx1x1          
        gs = gs.view(gs.size(0), -1)        # bxc

        # from global to local
        torch.cuda.synchronize()
        start_time = time.time() 
        camscore = self.convclass(ga.detach()) 
        camscore = torch.sigmoid(camscore)                 
        camscore = F.interpolate(camscore, size=(h, w), mode='bilinear', align_corners=True)
        wscore = F.max_pool2d(camscore, (h, 1)).squeeze(dim=2)  
        hscore = F.max_pool2d(camscore, (1, w)).squeeze(dim=3)
       

        linputs = torch.zeros([b, self.topN, 3, h, w]).cuda() 
        if self.vis == True:
           region_bboxs = torch.FloatTensor(b, self.topN, 6)
        for i in range(b): 
            gs_inv, gs_ind = gs[i].sort(descending=True)
            for j in range(self.topN):
                xs = wscore[i,gs_ind[j],:].squeeze()
                ys = hscore[i,gs_ind[j],:].squeeze()
                if xs.max() == xs.min():
                   xs = xs/xs.max()
                else: 
                   xs = (xs-xs.min())/(xs.max()-xs.min())
                if ys.max() == ys.min():
                   ys = ys/ys.max()
                else:
                   ys = (ys-ys.min())/(ys.max()-ys.min())
                x1, x2 = obj_loc(xs, self.threshold)
                y1, y2 = obj_loc(ys, self.threshold)
                linputs[i:i+1, j ] = F.interpolate(inputs[i:i+1, :, y1:y2, x1:x2], size=(h, w), mode='bilinear', align_corners=True)
                if self.vis == True:
                   region_bboxs[i,j] = torch.Tensor([x1, x2, y1, y2, gs_ind[j].item(), gs[i, gs_ind[j]].item()]) 

        
        # local stream 
        linputs = linputs.view(b * self.topN, 3, h, w)         
        la = self.features(linputs.detach())
        lf = feats_pooling(la, method=self.ps, sh=int(h/32), sw=int(w/32))
        lf = self.convclass(lf)
        ls = torch.sigmoid(lf)
        ls = F.max_pool2d(ls.reshape(b, self.topN, self.num_classes, 1).permute(0,3,1,2), (self.topN, 1))
        ls = ls.view(ls.size(0), -1)  #bxc
        
        if self.vis == True:
           return gs, ls, region_bboxs
        else:
           return gs, ls  

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.convclass.parameters(), 'lr': lr},
                ]


class MCARResnet(nn.Module):
    # 后三个指标是针对GCN
    def __init__(self, model, num_classes, ps, topN, threshold,  vis=False, in_channel=300, t=0, adj_file=None):
        super(MCARResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        num_features = model.layer4[1].conv1.in_channels
        self.convclass = nn.Conv2d(num_features, num_classes, kernel_size=1, stride=1, padding=0, bias=True)  #绿色1*1卷积进行分类预测
        self.ps = ps
        self.num_classes = num_classes
        self.num_features = num_features
        self.topN = topN
        self.threshold = threshold
        self.vis = vis
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        # gcn 
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(8, 8) ##代表的是全局最大池化
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)
        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

    def forward(self, inputs, inp):
        # global stream
        b, c, h, w = inputs.size() ##(batch, 3 ,imgsize, imgsize)
        ga = self.features(inputs) ##最终输出的特征分辨率为 h/32, w/32
        gf = feats_pooling(ga, method=self.ps, sh=int(h/32), sw=int(w/32)) # global pooling（全局池化）, （B,C,1,1）
        gf = self.convclass(gf)             #(b,num_classes,1,1)
        gs = torch.sigmoid(gf)              # 映射为[0，1]          
        gs = gs.view(gs.size(0), -1)        # (B,num_classes) 
        ####

        ####gcn, combine global and label
        global_fea = self.pooling(ga) # (b,C, 1, 1)
        global_fea = global_fea.view(global_fea.size(0), -1) #(B,C) (16,2048)
        inp = inp[0] #global和local共用
        adj = gen_adj(self.A).detach()
        global_x = self.gc1(inp, adj)
        global_x = self.relu(global_x)
        global_x = self.gc2(global_x, adj)
        global_x = global_x.transpose(0, 1)
        global_x = torch.matmul(global_fea, global_x) ##
        ######
        
        # from global to local
        camscore = self.convclass(ga.detach()) #(B,num_classes,h/32,w/32) 
        camscore = torch.sigmoid(camscore)
        # pytorch 0.4.0 把interpolate替换成upsample
        camscore = F.upsample(camscore, size=(h, w), mode='bilinear', align_corners=True)
        wscore = F.max_pool2d(camscore, (h, 1)).squeeze(dim=2)  #(B,num_classes,w)
        hscore = F.max_pool2d(camscore, (1, w)).squeeze(dim=3) # (B,num_classes,h)
        
        #选择topN个类别对应的channel
        linputs = torch.zeros([b, self.topN, 3, h, w]).cuda() 
        if self.vis == True:
           region_bboxs = torch.FloatTensor(b, self.topN, 6)
        for i in range(b): 
            # topN for MCAR method
            # sort前者值，后者索引
            gs_inv, gs_ind = gs[i].sort(descending=True)  ## gs按照从大到小排序  
        
            # bootomN for ablation study
            # gs_inv, gs_ind = gs[i].sort(descending=False)
            
            # randomN for ablation study
            # perm = torch.randperm(gs[i].size(0))
            # gs_inv = gs[i][perm]
            # gs_ind = perm

            for j in range(self.topN):
                xs = wscore[i,gs_ind[j],:].squeeze()  
                ys = hscore[i,gs_ind[j],:].squeeze()

                #min-max归一化操作
                if xs.max() == xs.min():
                   xs = xs/xs.max()
                else: 
                   xs = (xs-xs.min())/(xs.max()-xs.min())
                if ys.max() == ys.min():
                   ys = ys/ys.max()
                else:
                   ys = (ys-ys.min())/(ys.max()-ys.min())
               
                # 目标是在给定的图像中识别多类物体，每个类别只需要选择一个区分区域。
                # 因此，必须添加一些约束，以便在多个可行区间中选择唯一的区间
                x1, x2 = obj_loc(xs, self.threshold)
                y1, y2 = obj_loc(ys, self.threshold)
                # 选择topN（每一个）类别对应的区域
                linputs[i:i+1, j ] = F.upsample(inputs[i:i+1, :, y1:y2, x1:x2], size=(h, w), mode='bilinear', align_corners=True)
                if self.vis == True:
                   region_bboxs[i,j] = torch.Tensor([x1, x2, y1, y2, gs_ind[j].item(), gs[i, gs_ind[j]].item()])

        # local stream
        linputs = linputs.view(b * self.topN, 3, h, w)    ## (B*self.topN, 3, h, w)     
        la = self.features(linputs.detach())  ##(B*self.topN, 2048, h, w) 
        lf = feats_pooling(la, method=self.ps, sh=int(h/32), sw=int(w/32))
        lf = self.convclass(lf) ##(B*self.topN, num_classes, 1 , 1) 
        ls = torch.sigmoid(lf)  
        ls = F.max_pool2d(ls.reshape(b, self.topN, self.num_classes, 1).permute(0,2,1,3), (self.topN, 1))
        ls = ls.view(ls.size(0), -1)
        ###

        ####gcn, combine local and label
        local_fea = self.pooling(la) # (B*topN,C, 1, 1)
        # 这个地方reshape的时候是channel维度（2048）
        local_fea = F.max_pool2d(local_fea.reshape(b, self.topN, local_fea.shape[1], 1).permute(0,2,1,3), (self.topN, 1)) #(B,C,1,1)
        local_fea = local_fea.view(local_fea.size(0), -1) #(B,C)
        local_x = self.gc1(inp, adj)
        local_x = self.relu(local_x)
        local_x = self.gc2(local_x, adj)
        local_x = local_x.transpose(0, 1)
        local_x = torch.matmul(local_fea, local_x) #（B，num_classes)
        ######
        
        if self.vis == True:
            return global_x, local_x, region_bboxs
        #    return gs, ls, region_bboxs
        else:
            return global_x, local_x
        #    return gs, ls

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.convclass.parameters(), 'lr': lr},
                ]

def mcar_resnet50(num_classes, ps, topN, threshold, pretrained=True,  vis=False):
    model = models.resnet50(pretrained=pretrained)
    return MCARResnet(model, num_classes, ps, topN, threshold,  vis)

def mcar_resnet101(num_classes, ps, topN, threshold, pretrained=True,  vis=False,in_channel=300, t=0, adj_file=None):
    model = models.resnet101(pretrained=pretrained)
    return MCARResnet(model, num_classes, ps, topN, threshold,  vis, in_channel=in_channel, t=t, adj_file=adj_file)

def mcar_mobilenetv2(num_classes, ps, topN, threshold, pretrained=True, vis=False):
    model = mobilenetv2()
    if pretrained == True:
       m2net = torch.load('../pretrained/mobilenetv2_1.0-0c6065bc.pth')
       model.load_state_dict(m2net)
    return  MCARMobilenetv2(model, num_classes, ps, topN, threshold, vis)



