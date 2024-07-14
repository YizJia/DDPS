from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch
from ._base import Distiller, constant_init, kaiming_init
# from mmcv.runner import _load_checkpoint, load_state_dict

def set_distiller_cfg(cfg):
    distill_cfg = [
        dict(student_module = 'backbone.fpn.layer_blocks.3',
                         teacher_module = 'backbone.fpn.layer_blocks.3',
                         output_hook = True,
                         methods = [dict(student_channels = cfg.FGD.STUDENT_CHANNELS[0],
                                         teacher_channels = cfg.FGD.TEACHER_CHANNELS[0],
                                         name='loss_fgd_fpn_layer_3',
                                         temp = cfg.FGD.TEMP,
                                         alpha_fgd=cfg.FGD.ALPHA_FGD,
                                         beta_fgd=cfg.FGD.BATE_FGD,
                                         gamma_fgd=cfg.FGD.GAMMA_FGD,
                                         lambda_fgd=cfg.FGD.LAMBDA_FGD,
                                         )
                                    ]
                        ),
                   dict(student_module = 'backbone.fpn.layer_blocks.2',
                        teacher_module = 'backbone.fpn.layer_blocks.2',
                        output_hook = True,
                        methods = [dict(student_channels=cfg.FGD.STUDENT_CHANNELS[1],
                                        teacher_channels=cfg.FGD.TEACHER_CHANNELS[1],
                                        name='loss_fgd_fpn_layer_2',
                                        temp=cfg.FGD.TEMP,
                                        alpha_fgd=cfg.FGD.ALPHA_FGD,
                                        beta_fgd=cfg.FGD.BATE_FGD,
                                        gamma_fgd=cfg.FGD.GAMMA_FGD,
                                        lambda_fgd=cfg.FGD.LAMBDA_FGD,
                                        )
                                   ]
                         ),
                   dict(student_module = 'backbone.fpn.layer_blocks.1',
                        teacher_module = 'backbone.fpn.layer_blocks.1',
                        output_hook = True,
                        methods=[dict(student_channels = cfg.FGD.STUDENT_CHANNELS[2],
                                      teacher_channels = cfg.FGD.TEACHER_CHANNELS[2],
                                      name='loss_fgd_fpn_layer_1',
                                      temp = cfg.FGD.TEMP,
                                      alpha_fgd=cfg.FGD.ALPHA_FGD,
                                      beta_fgd=cfg.FGD.BATE_FGD,
                                      gamma_fgd=cfg.FGD.GAMMA_FGD,
                                      lambda_fgd=cfg.FGD.LAMBDA_FGD,
                                      )
                                 ]
                        ),
                   dict(student_module='backbone.fpn.layer_blocks.0',
                        teacher_module='backbone.fpn.layer_blocks.0',
                        output_hook=True,
                        methods=[dict(student_channels=cfg.FGD.STUDENT_CHANNELS[3],
                                      teacher_channels=cfg.FGD.TEACHER_CHANNELS[3],
                                      name='loss_fgd_fpn_layer_0',
                                      temp=cfg.FGD.TEMP,
                                      alpha_fgd=cfg.FGD.ALPHA_FGD,
                                      beta_fgd=cfg.FGD.BATE_FGD,
                                      gamma_fgd=cfg.FGD.GAMMA_FGD,
                                      lambda_fgd=cfg.FGD.LAMBDA_FGD,
                                      )
                                 ]
                        ),
                   ]
    return distill_cfg


class FGD(Distiller):
    def __init__(self, cfg, student, teacher):
        super(FGD, self).__init__(student, teacher)
        self.device = torch.device(cfg.DEVICE)
        # FGD distiller
        self.distill_losses = nn.ModuleDict()
        self.distill_cfg = set_distiller_cfg(cfg)
        self.distill_lw = cfg.FGD.LW_FGD

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

        student_modules = dict(self.student.named_modules())
        teacher_modules = dict(self.teacher.named_modules())
        def regitster_hooks(student_module,teacher_module):
            def hook_teacher_forward(module, input, output):
                self.register_buffer(teacher_module,output)

            def hook_student_forward(module, input, output):
                self.register_buffer(student_module,output )
            return hook_teacher_forward, hook_student_forward
    
        for item_loc in self.distill_cfg:
            # print(item_loc)
            student_module = 'student_' + item_loc['student_module'].replace('.','_')
            teacher_module = 'teacher_' + item_loc['teacher_module'].replace('.','_')

            self.register_buffer(student_module, None)
            self.register_buffer(teacher_module, None)

            hook_teacher_forward, hook_student_forward = regitster_hooks(student_module, teacher_module)
            teacher_modules[item_loc['teacher_module']].register_forward_hook(hook_teacher_forward)
            student_modules[item_loc['student_module']].register_forward_hook(hook_student_forward)

            for item_loss in item_loc['methods']:                
                loss_name = item_loss['name']
                self.distill_losses[loss_name] = FeatureLoss(item_loss['student_channels'],item_loss['teacher_channels'], 
                                                            item_loss['name'],item_loss['temp'],item_loss['alpha_fgd'],
                                                            item_loss['beta_fgd'],item_loss['gamma_fgd'],item_loss['lambda_fgd']).to(self.device)

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        parameters = super().get_learnable_parameters()
        for item_loc in self.distill_cfg:
            for item_loss in item_loc['methods']:
                loss_name = item_loss['name']
                parameters += list(self.distill_losses[loss_name].parameters())
        return parameters

    def get_state_dict(self):
        # TODO
        # only for student module in save parameter stage
        distiller_state_dict = {}
        distiller_state_dict["student"] = self.student.state_dict()
        fgd_state_dict = {}
        for item_loc in self.distill_cfg:
            for item_loss in item_loc['methods']:
                loss_name = item_loss['name']                
                fgd_state_dict[loss_name] = self.distill_losses[loss_name].state_dict()
        
        distiller_state_dict["fgd"] = fgd_state_dict
        return distiller_state_dict
    
    def set_state_dict(self, ckpt_model):
        # TODO
        # only for studnet module in load parameter stage for eval
        self.student.load_state_dict(ckpt_model["student"], strict=False)
        fgd_state_dict = ckpt_model["fgd"]
        for item_loc in self.distill_cfg:
            for item_loss in item_loc['methods']:
                loss_name = item_loss['name']                
                self.distill_losses[loss_name].load_state_dict(fgd_state_dict[loss_name], strict = False)
        return 0

    def forward_train(self, images, targets, **kwargs):
        # training function for the distillation method
        student_loss = self.student(images, targets)
        # make the input of both student and teacher no difference
        s_images, s_targets = self.student.get_input_network()
        s_features = self.student.get_features()

        img_shapes = s_images.image_sizes

        with torch.no_grad():
            # teacher model forward propagation
            # forward propagation for FGD
            t_proposals, t_detections, t_reid_box_logits, t_box_pids, t_box_embeddings = self.teacher.extract_infor_forKD(s_images, s_targets)

        # after prepare the features/embeddings for KD methods
        # FGD process
        buffer_dict = dict(self.named_buffers())
        # fgd_lw_indx = 0
        for item_loc in self.distill_cfg:
            
            student_module = 'student_' + item_loc['student_module'].replace('.','_')
            teacher_module = 'teacher_' + item_loc['teacher_module'].replace('.','_')
            
            student_feat = buffer_dict[student_module]
            teacher_feat = buffer_dict[teacher_module]

            for item_loss in item_loc['methods']:
                loss_name = item_loss['name']
                student_loss[loss_name] = self.distill_losses[loss_name](student_feat, teacher_feat, s_targets, img_shapes)

        return student_loss

class FeatureLoss(nn.Module):

    """PyTorch version of `Focal and Global Knowledge Distillation for Detectors`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        alpha_fgd (float, optional): Weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): Weight of mask_loss. Defaults to 0.001
        lambda_fgd (float, optional): Weight of relation_loss. Defaults to 0.000005
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name,
                 temp=0.5,
                 alpha_fgd=0.001,
                 beta_fgd=0.0005,
                 gamma_fgd=0.001,
                 lambda_fgd=0.000005,
                 ):
        super(FeatureLoss, self).__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None
        
        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))

        self.reset_parameters()


    def forward(self,
                preds_S,
                preds_T,
                gt_bboxes,
                img_metas):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:], 'the output dim of teacher and student differ'

        if self.align is not None:
            preds_S = self.align(preds_S)

        N,C,H,W = preds_S.shape

        S_attention_t, C_attention_t = self.get_attention(preds_T, self.temp)
        S_attention_s, C_attention_s = self.get_attention(preds_S, self.temp)

        # TODO show attention map
        # visualization_attention(S_attention_t, C_attention_t)


        Mask_fg = torch.zeros_like(S_attention_t)
        Mask_bg = torch.ones_like(S_attention_t)
        wmin,wmax,hmin,hmax = [],[],[],[]

        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i]['boxes'])
            new_boxxes[:, 0] = gt_bboxes[i]['boxes'][:, 0]/img_metas[i][1]*W
            new_boxxes[:, 2] = gt_bboxes[i]['boxes'][:, 2]/img_metas[i][1]*W
            new_boxxes[:, 1] = gt_bboxes[i]['boxes'][:, 1]/img_metas[i][0]*H
            new_boxxes[:, 3] = gt_bboxes[i]['boxes'][:, 3]/img_metas[i][0]*H

            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            area = 1.0/(hmax[i].view(1,-1)+1-hmin[i].view(1,-1))/(wmax[i].view(1,-1)+1-wmin[i].view(1,-1))

            for j in range(len(gt_bboxes[i]['boxes'])):
                Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1] = \
                        torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1], area[0][j])

            Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
            if torch.sum(Mask_bg[i]):
                Mask_bg[i] /= torch.sum(Mask_bg[i])

        fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T, Mask_fg, Mask_bg, 
                           C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        mask_loss = self.get_mask_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        rela_loss = self.get_rela_loss(preds_S, preds_T)


        loss = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss \
               + self.gamma_fgd * mask_loss + self.lambda_fgd * rela_loss
            
        return loss


    def get_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        N, C, H, W= preds.shape

        value = torch.abs(preds)
        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map/temp).view(N,-1), dim=1)).view(N, H, W)

        # Bs*C
        channel_map = value.mean(axis=2,keepdim=False).mean(axis=2,keepdim=False)
        C_attention = C * F.softmax(channel_map/temp, dim=1)

        return S_attention, C_attention


    def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, C_s, C_t, S_s, S_t):
        loss_mse = nn.MSELoss(reduction='sum')
        
        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        S_t = S_t.unsqueeze(dim=1)

        fea_t= torch.mul(preds_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t)/len(Mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t)/len(Mask_bg)

        return fg_loss, bg_loss


    def get_mask_loss(self, C_s, C_t, S_s, S_t):

        mask_loss = torch.sum(torch.abs((C_s-C_t)))/len(C_s) + torch.sum(torch.abs((S_s-S_t)))/len(S_s)

        return mask_loss
     
    
    def spatial_pool(self, x, in_type):
        batch, channel, width, height = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        if in_type == 0:
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = F.softmax(context_mask, dim=2)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context


    def get_rela_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')

        context_s = self.spatial_pool(preds_S, 0)
        context_t = self.spatial_pool(preds_T, 1)

        out_s = preds_S
        out_t = preds_T

        channel_add_s = self.channel_add_conv_s(context_s)
        out_s = out_s + channel_add_s

        channel_add_t = self.channel_add_conv_t(context_t)
        out_t = out_t + channel_add_t

        rela_loss = loss_mse(out_s, out_t)/len(out_s)
        
        return rela_loss


    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            constant_init(m[-1], val=0)
        else:
            constant_init(m, val=0)

    
    def reset_parameters(self):
        kaiming_init(self.conv_mask_s, mode='fan_in')
        kaiming_init(self.conv_mask_t, mode='fan_in')
        self.conv_mask_s.inited = True
        self.conv_mask_t.inited = True

        self.last_zero_init(self.channel_add_conv_s)
        self.last_zero_init(self.channel_add_conv_t)