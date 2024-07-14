from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.ops import boxes as box_ops
from ._base import Distiller
# from mmcv.runner import _load_checkpoint, load_state_dict
from .FGD import  set_distiller_cfg, FeatureLoss

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class FGD_Det_Relation(Distiller):
    def __init__(self, cfg, student, teacher):
        super(FGD_Det_Relation, self).__init__(student, teacher)
        self.device = torch.device(cfg.DEVICE)
        # FGD distiller
        self.distill_losses = nn.ModuleDict()
        self.distill_cfg = set_distiller_cfg(cfg)
        self.distill_lw = cfg.FGD.LW_FGD
        # detection cls distiller
        self.temperature = cfg.DETKD.T
        self.warmup = cfg.DETKD.WARMUP
        self.from_T = cfg.DETKD.FROM_T
        # Relation distiller
        self.from_T = cfg.GRAPHRELA.FROM_T
        self.warmup = cfg.GRAPHRELA.WARMUP
        self.weight = cfg.GRAPHRELA.WEIGHT
        self.nms_threshold = cfg.GRAPHRELA.NMSTHRES

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
        def regitster_hooks(student_module, teacher_module):
            def hook_teacher_forward(module, input, output):
                self.register_buffer(teacher_module, output)

            def hook_student_forward(module, input, output):
                self.register_buffer(student_module, output)

            return hook_teacher_forward, hook_student_forward

        for item_loc in self.distill_cfg:
            # print(item_loc)
            student_module = 'student_' + item_loc['student_module'].replace('.', '_')
            teacher_module = 'teacher_' + item_loc['teacher_module'].replace('.', '_')

            self.register_buffer(student_module, None)
            self.register_buffer(teacher_module, None)

            hook_teacher_forward, hook_student_forward = regitster_hooks(student_module, teacher_module)
            teacher_modules[item_loc['teacher_module']].register_forward_hook(hook_teacher_forward)
            student_modules[item_loc['student_module']].register_forward_hook(hook_student_forward)

            for item_loss in item_loc['methods']:
                loss_name = item_loss['name']
                self.distill_losses[loss_name] = FeatureLoss(item_loss['student_channels'],
                                                             item_loss['teacher_channels'],
                                                             item_loss['name'], item_loss['temp'],
                                                             item_loss['alpha_fgd'],
                                                             item_loss['beta_fgd'], item_loss['gamma_fgd'],
                                                             item_loss['lambda_fgd']).to(self.device)

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        parameters = super().get_learnable_parameters()
        for item_loc in self.distill_cfg:
            for item_loss in item_loc['methods']:
                loss_name = item_loss['name']
                parameters += list(self.distill_losses[loss_name].parameters())
        return parameters

    def get_state_dict(self):
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
        # only for studnet module in load parameter stage for eval
        self.student.load_state_dict(ckpt_model["student"], strict=False)
        fgd_state_dict = ckpt_model["fgd"]
        for item_loc in self.distill_cfg:
            for item_loss in item_loc['methods']:
                loss_name = item_loss['name']
                self.distill_losses[loss_name].load_state_dict(fgd_state_dict[loss_name], strict=False)
        return 0

    # specific Relation KD method 
    def select_complete_persons(self, t_box_embeddings, s_box_embeddings, generate_boxes, generate_box_pids, reid_box_logits):
        boxes_per_images = [len(boxes_in_images) for boxes_in_images in generate_boxes]
        pred_boxes = torch.cat(generate_boxes, dim=0)
        pred_box_pids = torch.cat(generate_box_pids, dim=0)
        softmax_scores = F.softmax(reid_box_logits, dim=-1)

        assert sum(boxes_per_images) == pred_boxes.shape[0]
        assert pred_boxes.shape[0] == pred_box_pids.shape[0]

        person_box_pids = pred_box_pids - 1
        # only reserve the boxes belong to person instances
        inds = torch.nonzero(person_box_pids >= 0).squeeze(1)
        assert len(inds) == softmax_scores.shape[0]
        pred_boxes, pred_box_pids, t_embeddings, s_embeddings = (
                pred_boxes[inds],
                pred_box_pids[inds],
                t_box_embeddings[inds],
                s_box_embeddings[inds],
            )
        
        # get pred_scores from softmax output based on the pids
        pred_scores = torch.zeros([softmax_scores.shape[0]]).to(self.device)
        for i in range(softmax_scores.shape[0]):
            person_indx = pred_box_pids[i]
            assert person_indx > 0
            if person_indx < 5555:
                pred_scores[i] = softmax_scores[i][person_indx-1]
            else:
                pred_scores[i] = torch.max(softmax_scores[i][5532:])

        # get the number of person instances in each image
        persons_per_images = []
        start = 0
        for i, boxes_image in enumerate(boxes_per_images):
            count = 0
            for j in range(len(inds)):
                if inds[j] < start:
                    continue
                elif inds[j] >= start+boxes_image:
                    break
                else:
                    count = count + 1
            persons_per_images.append(count)
            start = start + boxes_image
        
        assert pred_scores.shape[0] == sum(persons_per_images)

        
        pred_boxes = pred_boxes.split(persons_per_images,0)
        pred_box_pids = pred_box_pids.split(persons_per_images,0)
        pred_scores = pred_scores.split(persons_per_images,0)
        pred_t_embeddings = t_embeddings.split(persons_per_images,0)
        pred_s_embeddings = s_embeddings.split(persons_per_images,0)

        all_boxes = []
        all_box_pids = []
        all_scores = []
        all_t_embeddings = []
        all_s_embeddings = []

        for boxes, box_pids, scores, t_embeddings, s_embeddings in zip(
            pred_boxes, pred_box_pids, pred_scores, pred_t_embeddings, pred_s_embeddings
        ):
            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            box_pids = box_pids.flatten()
            scores = scores.flatten()
            t_embeddings = t_embeddings.reshape(-1, 256)
            s_embeddings = s_embeddings.reshape(-1, 256)

            # select embeddings
            # which correspond to boxes with higher scores (using LT & CQ), more complete person
            # remove low scoreing reid embeddings
            # modify v2.3
            # modify v2.1
            # inds = torch.nonzero(scores > self.threshold).squeeze(1)
            # boxes, box_pids, scores, t_embeddings, s_embeddings = (
            #     boxes[inds],
            #     box_pids[inds],
            #     scores[inds],
            #     t_embeddings[inds],
            #     s_embeddings[inds],
            # )
                
            # remove empty boxes
            inds = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, box_pids, scores, t_embeddings, s_embeddings = (
                boxes[inds],
                box_pids[inds],
                scores[inds],
                t_embeddings[inds],
                s_embeddings[inds],
            ) 
            
            # non-maximum suppression, independently done per class
            inds = box_ops.batched_nms(boxes, scores, box_pids, self.nms_threshold)
            boxes, box_pids, scores, t_embeddings, s_embeddings = (
                boxes[inds],
                box_pids[inds],
                scores[inds],
                t_embeddings[inds],
                s_embeddings[inds],
            )
            
            all_boxes.append(boxes)
            all_box_pids.append(box_pids)
            all_scores.append(scores)
            all_t_embeddings.append(t_embeddings)
            all_s_embeddings.append(s_embeddings)

        t_person_embeddings = torch.cat(all_t_embeddings,0)
        s_person_embeddings = torch.cat(all_s_embeddings,0)

        return t_person_embeddings, s_person_embeddings

    def forward_train(self, images, targets, **kwargs):
        # training function for the distillation method
        student_loss = self.student(images, targets)
        # make the input of both student and teacher no difference
        s_images, s_targets = self.student.get_input_network()
        s_features = self.student.get_features()

        img_shapes = s_images.image_sizes

        if not self.from_T:          
            # get proposals from student
            s_proposals, _ = self.student.roi_heads.get_select_proposals() 
            s_det_pro_cls_scores, _ = self.student.roi_heads.get_logits()
            # get the detection results -- boxes from student
            s_boxes, s_box_pids = self.student.roi_heads.get_select_boxes()
            s_box_embeddings = self.student.roi_heads.get_embeddings()

            generate_boxes = s_boxes
            generate_box_pids = s_box_pids

        with torch.no_grad():
            # teacher model forward propagation
            if self.from_T:
                # forward propagation for FGD
                t_proposals, t_detections, t_reid_box_logits, t_box_pids, t_box_embeddings = self.teacher.extract_infor_forKD(s_images, s_targets)
                t_det_pro_cls_scores, _ = self.teacher.roi_heads.get_logits()

                generate_boxes = [box_per_images["boxes"] for box_per_images in t_detections]
                generate_box_pids = t_box_pids
            else:
                t_features = self.teacher.extract_features(s_images)
                # detection head part
                t_proposal_features = self.teacher.roi_heads.box_roi_pool(t_features, s_proposals, img_shapes)
                t_proposal_features = self.teacher.roi_heads.box_head(t_proposal_features)
                t_det_pro_cls_scores, _ = self.teacher.roi_heads.faster_rcnn_predictor(t_proposal_features["feat_afFC"])
                # reid head part
                t_box_features = self.teacher.roi_heads.box_roi_pool(t_features, s_boxes, img_shapes)
                t_box_features = self.teacher.roi_heads.reid_head(t_box_features)
                t_box_embeddings, _ = self.teacher.roi_heads.embedding_head(t_box_features)
                t_reid_box_logits = self.teacher.roi_heads.reid_loss.get_soften_logits(t_box_embeddings, s_box_pids)

        if self.from_T:
            # student detection part
            s_det_pro_fea = self.student.roi_heads.box_roi_pool(s_features, t_proposals, img_shapes)
            s_det_pro_features = self.student.roi_heads.box_head(s_det_pro_fea)
            s_det_pro_cls_scores, _ = self.student.roi_heads.faster_rcnn_predictor(
                s_det_pro_features["feat_afFC"]
            )
            # student reid part
            s_reid_box_logits, _, s_box_embeddings = self.student.roi_heads.extract_infor_forKD(s_features, t_detections, img_shapes, box_labels=t_box_pids)

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

        # Detection cls KD process
        loss_det_kd = min(kwargs["epoch"] / self.warmup, 1.0) * kd_loss(
            s_det_pro_cls_scores, t_det_pro_cls_scores, self.temperature
        )
        student_loss["loss_det_kd"] = loss_det_kd

        # Relation KD process
        t_person_embeddings, s_person_embeddings = self.select_complete_persons(t_box_embeddings, s_box_embeddings,
                                                                               generate_boxes, generate_box_pids,
                                                                               t_reid_box_logits)

        # t_person_embeddings, s_person_embeddings = t_box_embeddings, s_box_embeddings
        # the embeddings already are normed
        t_similarity = torch.mm(t_person_embeddings, t_person_embeddings.T)
        s_similarity = torch.mm(s_person_embeddings, s_person_embeddings.T)

        assert t_similarity.shape == s_similarity.shape

        loss_mse = nn.MSELoss(reduction='sum')
        node_loss = loss_mse(s_similarity, t_similarity)
        # student_loss["reid_graph_loss"] = min(kwargs["epoch"] / self.warmup, 1.0) * node_loss
        student_loss["reid_graph_loss"] =  self.weight * node_loss
        return student_loss