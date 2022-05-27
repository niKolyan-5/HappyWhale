import timm

import torch
import torch.nn as nn
import torch.nn.functional as F

import math



class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6, requires_grad=False):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=requires_grad)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class MultiAtrous(nn.Module):

    def __init__(self, in_channel, out_channel, size, dilation_rates=[6, 12, 18]):
        super().__init__()
        self.dilated_convs = [
            nn.Conv2d(in_channel, int(out_channel / 4),
                      kernel_size=3, dilation=rate, padding=rate)
            for rate in dilation_rates
        ]
        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(int(out_channel / 4), int(out_channel / 2), kernel_size=1),
            nn.ReLU(),
            nn.Upsample(size=(size, size), mode='bilinear')
        )
        # self.dilated_convs.append(self.gap_branch)
        self.dilated_convs = nn.ModuleList(self.dilated_convs)

    def forward(self, x):
        local_feat = []
        for dilated_conv in self.dilated_convs:
            out = dilated_conv(x)
            out = self.gap_branch(out)
            local_feat.append(out)
        local_feat = torch.cat(local_feat, dim=1)
        return local_feat


class DolgLocalBranch(nn.Module):

    def __init__(self, in_channel, out_channel, img_size, hidden_channel=1024):
        super().__init__()
        self.multi_atrous = MultiAtrous(in_channel, hidden_channel, size=int(img_size / 32))
        self.conv1x1_1 = nn.Conv2d(int(3 * hidden_channel / 2), out_channel, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=1, bias=False)
        self.conv1x1_3 = nn.Conv2d(out_channel, out_channel, kernel_size=1)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channel)
        self.softplus = nn.Softplus()

    def forward(self, x):
        local_feat = self.multi_atrous(x)

        local_feat = self.conv1x1_1(local_feat)
        local_feat = self.relu(local_feat)
        local_feat = self.conv1x1_2(local_feat)
        local_feat = self.bn(local_feat)

        attention_map = self.relu(local_feat)
        attention_map = self.conv1x1_3(attention_map)
        attention_map = self.softplus(attention_map)

        local_feat = F.normalize(local_feat, p=2, dim=1)
        local_feat = local_feat * attention_map

        return local_feat


class OrthogonalFusion(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, local_feat, global_feat):
        global_feat_norm = torch.norm(global_feat, p=2, dim=1)
        projection = torch.bmm(global_feat.unsqueeze(1), torch.flatten(
            local_feat, start_dim=2))
        projection = torch.bmm(global_feat.unsqueeze(
            2), projection).view(local_feat.size())
        projection = projection / \
                     (global_feat_norm * global_feat_norm).view(-1, 1, 1, 1)
        orthogonal_comp = local_feat - projection
        global_feat = global_feat.unsqueeze(-1).unsqueeze(-1)
        return torch.cat([global_feat.expand(orthogonal_comp.size()), orthogonal_comp], dim=1)


class ArcFace(nn.Module):

    def __init__(self, in_features, out_features, scale_factor=64.0, margin=0.50, criterion=None):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.margin = margin
        self.scale_factor = scale_factor

        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):

        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = phi.type(cosine.type())
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        logit = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logit *= self.scale_factor

        return logit


class DolgNet(nn.Module):
    """ DolgNet model class, based on method
        Deep Orthogonal Fusion of Local and Global Features:
        https://paperswithcode.com/paper/dolg-single-stage-image-retrieval-with-deep
    Attributes:
        input_dim: input dimension
        hidden_dim: hidden dimension
        output_dim: output dimension for embeddings
        num_of_classes: number of classes
        scale: scale parameter for ArcFace Layer
        margin: margin parameter for ArcFace Layer
        backbone_name: backbone model name
        img_size: input img size
        local_branch_input_dim: input dimension for DolgLocalBranch Layer
        fc_1_input_dim: input dimension fo первого FC Layer
        drop_rate: probability for DropOut Layer
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_of_classes: int,
                 scale: int, margin: float, backbone_name: str, img_size: int,
                 local_branch_input_dim: int, fc_1_input_dim: int, drop_rate: float):
        super().__init__()

        self.cnn = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            in_chans=input_dim,
            out_indices=(3, 4)
        )


        self.orthogonal_fusion = OrthogonalFusion()
        self.local_branch = DolgLocalBranch(local_branch_input_dim, hidden_dim, img_size)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gem_pool = GeM()
        self.fc_1 = nn.Linear(fc_1_input_dim, hidden_dim)
        self.fc_2 = nn.Linear(int(2 * hidden_dim), output_dim)

        self.arc = ArcFace(
            in_features=output_dim,
            out_features=num_of_classes,
            scale_factor=scale,
            margin=margin
        )

        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, labels=None):
        local_feat, global_feat = self.cnn(x)

        local_feat = self.local_branch(local_feat)  # ,hidden_channel,16,16
        global_feat = self.fc_1(self.gem_pool(global_feat).squeeze())  # ,1024

        feat = self.orthogonal_fusion(local_feat, global_feat)

        feat = self.gap(feat).squeeze()

        feat_droped = self.dropout(feat)

        emb = self.fc_2(feat_droped)

        if labels is not None:
            output = self.arc(emb, labels)
            return emb, output

        else:
            return emb




