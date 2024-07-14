from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from ._base import Distiller
# from mmcv.runner import _load_checkpoint, load_state_dict

class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None):
        n,_,h,w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            shape = x.shape[-2:]
            y = F.interpolate(y, shape, mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
        # output 
        y = self.conv2(x)
        return y, x

class ReviewKD(Distiller):
    def __init__(
        self, cfg, student, teacher
    ):
        super(ReviewKD, self).__init__(student, teacher)

        self.weight = cfg.REVIEWKD.WEIGHT
        # initialization config: init_student
        # if cfg.DISTILLER.INIT_STUDENT:
        #     t_checkpoint = _load_checkpoint(cfg.DISTILLER.TEACHER.CKPT)
        #     all_name = []
        #     for name, v in t_checkpoint["model"].items():
        #         if name.startswith("backbone.body."):
        #             continue
        #         elif name.startswith("backbone.fpn."):
        #             continue
        #         else:
        #             all_name.append((name, v))

        #     state_dict = OrderedDict(all_name)
        #     load_state_dict(self.student, state_dict)
        
        in_channels = [256,256,256,256,256]
        out_channels = [256,256,256,256,256]
        mid_channel = 256

        abfs = nn.ModuleList()

        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))

        self.abfs = abfs[::-1]

    def forward(self, images, targets, **kwargs):
        # training function for the distillation method
        student_loss = self.student(images, targets)
        # make the input of both student and teacher no difference
        s_images, s_targets = self.student.get_input_network()
        s_features = self.student.get_features()

        img_shapes = s_images.image_sizes

        with torch.no_grad():
            t_features = self.teacher.extract_features(s_images)
        
        # ReviewKD process
        stu_features = [s_features[f] for f in s_features]
        tea_features = [t_features[f] for f in t_features]
        x = stu_features[::-1]
        new_stu_features = []
        out_features, res_features = self.abfs[0](x[0])
        new_stu_features.append(out_features)
        for features, abf in zip(x[1:], self.abfs[1:]):
            out_features, res_features = abf(features, res_features)
            new_stu_features.insert(0, out_features)

        student_loss["reviewkd"] = self.weight * hcl(new_stu_features, tea_features)
        return student_loss

def hcl(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n,c,h,w = fs.shape
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in [4,2,1]:
            if l >=h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
            tmpft = F.adaptive_avg_pool2d(ft, (l,l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all