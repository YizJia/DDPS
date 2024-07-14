import torch
import torch.nn as nn
import torch.nn.functional as F

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()       
        self.student = student
        self.teacher = teacher

    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters() if v.requires_grad]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0
        
    def get_student(self):
        return self.student

    def get_state_dict(self):
        # only for student module in save parameter stage        
        return self.student.state_dict()
    
    def set_state_dict(self, ckpt_model):
        # only for studnet module in load parameter stage for eval
        self.student.load_state_dict(ckpt_model, strict=False)
        return 0

    def forward_train(self, images, targets, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()

    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, img, targets=None, **kwargs):
        if self.training:
            return self.forward_train(img, targets, **kwargs)
        return self.forward_test(img)


class Vanilla(nn.Module):
    def __init__(self, student):
        super(Vanilla, self).__init__()
        self.student = student

    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        loss = F.cross_entropy(logits_student, target)
        return logits_student, {"ce": loss}

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])

    def forward_test(self, image):
        return self.student(image)[0]
