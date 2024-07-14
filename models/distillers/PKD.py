from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch
from ._base import Distiller
# from mmcv.runner import _load_checkpoint, load_state_dict

def set_distiller_cfg(cfg):
    distill_cfg = [
        dict(student_module = 'backbone.fpn.layer_blocks.3',
                         teacher_module = 'backbone.fpn.layer_blocks.3',
                         output_hook = True,
                         methods = [dict(student_channels = cfg.FGD.STUDENT_CHANNELS[0],
                                         teacher_channels = cfg.FGD.TEACHER_CHANNELS[0],
                                         name='loss_pkd_fpn_layer_3',
                                         pkd_weight = cfg.PKD.WEIGHT,
                                         )
                                    ]
                        ),
                   dict(student_module = 'backbone.fpn.layer_blocks.2',
                        teacher_module = 'backbone.fpn.layer_blocks.2',
                        output_hook = True,
                        methods = [dict(student_channels=cfg.FGD.STUDENT_CHANNELS[1],
                                        teacher_channels=cfg.FGD.TEACHER_CHANNELS[1],
                                        name='loss_pkd_fpn_layer_2',
                                        pkd_weight = cfg.PKD.WEIGHT,
                                        )
                                   ]
                         ),
                   dict(student_module = 'backbone.fpn.layer_blocks.1',
                        teacher_module = 'backbone.fpn.layer_blocks.1',
                        output_hook = True,
                        methods=[dict(student_channels = cfg.FGD.STUDENT_CHANNELS[2],
                                      teacher_channels = cfg.FGD.TEACHER_CHANNELS[2],
                                      name='loss_pkd_fpn_layer_1',
                                      pkd_weight = cfg.PKD.WEIGHT,
                                      )
                                 ]
                        ),
                   dict(student_module='backbone.fpn.layer_blocks.0',
                        teacher_module='backbone.fpn.layer_blocks.0',
                        output_hook=True,
                        methods=[dict(student_channels=cfg.FGD.STUDENT_CHANNELS[3],
                                      teacher_channels=cfg.FGD.TEACHER_CHANNELS[3],
                                      name='loss_pkd_fpn_layer_0',
                                      pkd_weight = cfg.PKD.WEIGHT,
                                      )
                                 ]
                        ),
                   ]
    return distill_cfg


class PKD(Distiller):
    def __init__(self, cfg, student, teacher):
        super(PKD, self).__init__(student, teacher)
        self.device = torch.device(cfg.DEVICE)
        # distiller
        self.distill_losses = nn.ModuleDict()
        self.distill_cfg = set_distiller_cfg(cfg)

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
                                                            item_loss['pkd_weight']).to(self.device)

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
        # pkd process
        buffer_dict = dict(self.named_buffers())
        for item_loc in self.distill_cfg:
            
            student_module = 'student_' + item_loc['student_module'].replace('.','_')
            teacher_module = 'teacher_' + item_loc['teacher_module'].replace('.','_')
            
            student_feat = buffer_dict[student_module]
            teacher_feat = buffer_dict[teacher_module]

            for item_loss in item_loc['methods']:
                loss_name = item_loss['name']
                student_loss[loss_name] = self.distill_losses[loss_name](student_feat, teacher_feat)

        return student_loss

class FeatureLoss(nn.Module):

    """PyTorch version of `pearson knowledge distillation`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        pkd_weight(float): The weight of pkd loss.
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 pkd_weight=1.0,
                 ):
        super(FeatureLoss, self).__init__()
        self.pkd_weight = pkd_weight

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None
        
    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:], 'the output dim of teacher and student differ'

        if self.align is not None:
            preds_S = self.align(preds_S)

        # pearson knowledge distillation loss
        pkd_loss = self.get_pkd_loss(preds_S, preds_T)

        loss = self.pkd_weight * pkd_loss
            
        return loss

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize the feature maps to have zero mean and unit variances.
        Args:
            feat (torch.Tensor): The original feature map with shape
                (N, C, H, W).
        """
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def get_pkd_loss(self, preds_S, preds_T):
        out_s = preds_S
        out_t = preds_T

        norm_out_s, norm_out_t = self.norm(out_s), self.norm(out_t)

        pkd_loss = F.mse_loss(norm_out_s, norm_out_t) / 2
        
        return pkd_loss