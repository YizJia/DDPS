import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

class DKD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, cfg, student, teacher):
        super(DKD, self).__init__(student, teacher)
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.from_T = cfg.DKD.FROM_T

    def forward_train(self, images, targets, **kwargs):
        student_loss = self.student(images, targets)
        # input of the student and the teacher are kept the same.
        s_images, s_targets = self.student.get_input_network()
        s_features = self.student.get_features()

        # prepare for FGD
        img_shapes = s_images.image_sizes

        if not self.from_T:
            s_proposals, s_proposal_labels = self.student.roi_heads.get_select_proposals()
            s_det_pro_cls_scores, _ = self.student.roi_heads.get_logits()
            assert len(s_proposal_labels) == len(s_proposal_labels)

        # training function for the distillation method
        with torch.no_grad():
            if self.from_T:
                # forward propagation for FGD
                t_proposals, t_detections, t_reid_boxes_logits, t_box_pids, t_box_embeddings = self.teacher.extract_infor_forKD(s_images, s_targets)
                t_det_pro_cls_scores, _ = self.teacher.roi_heads.get_logits()
            else:
                t_features = self.teacher.extract_features(s_images)
                t_proposal_features = self.teacher.roi_heads.box_roi_pool(t_features, s_proposals, img_shapes)
                t_proposal_features = self.teacher.roi_heads.box_head(t_proposal_features)
                t_det_pro_cls_scores, _ = self.teacher.roi_heads.faster_rcnn_predictor(t_proposal_features["feat_afFC"]
        )

        if self.from_T:
            # student detection part
            s_det_pro_fea = self.student.roi_heads.box_roi_pool(s_features, t_proposals, img_shapes)
            s_det_pro_features = self.student.roi_heads.box_head(s_det_pro_fea)
            s_det_pro_cls_scores, _ = self.student.roi_heads.faster_rcnn_predictor(
                s_det_pro_features["feat_afFC"]
            )

        loss_det_dkd = dkd_loss(s_det_pro_cls_scores, t_det_pro_cls_scores, torch.cat(s_proposal_labels, dim=0),
                                self.alpha, self.beta, self.temperature)
        student_loss["loss_det_dkd"] = loss_det_dkd
        return student_loss