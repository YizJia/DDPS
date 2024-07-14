import argparse
import datetime
import os.path as osp
import time

import torch
import torch.utils.data

from datasets import build_test_loader, build_train_loader
from defaults import get_default_cfg
from engine import evaluate_performance, train_one_epoch
from models.seqnet_d import SeqNet_D as Net
from models.distillers import distiller_dict
from utils.utils import mkdir, resume_from_ckpt, save_on_master, set_random_seed


def main(args):
    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(cfg.DEVICE)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    if cfg.BASELINE.PROJECT:
        print("Creating baseline model")
        trainer = Net(cfg, cfg.BASELINE)
    elif cfg.DISTILLER.PROJECT:
        print("Creating model: teacher and student")
        model_teacher = Net(cfg, cfg.DISTILLER.TEACHER)
        # Teacher model is pre_trained
        resume_from_ckpt(cfg.DISTILLER.TEACHER.CKPT, model_teacher, pretrained=cfg.DISTILLER.TEACHER.PRETRAINED)
        model_student = Net(cfg, cfg.DISTILLER.STUDENT)
        trainer = distiller_dict[cfg.DISTILLER.TYPE](cfg, model_student, model_teacher)
    else:
        print("Error")
        exit(0)

    trainer.to(device)

    print("Loading data")
    train_loader = build_train_loader(cfg)
    gallery_loader, query_loader = build_test_loader(cfg)

    if args.eval:
        assert args.ckpt, "--ckpt must be specified when --eval enabled"
        if cfg.BASELINE.PROJECT:
            resume_from_ckpt(args.ckpt, trainer, pretrained=cfg.BASELINE.PROJECT)
            eva_model = trainer
        elif cfg.DISTILLER.PROJECT:
            resume_from_ckpt(args.ckpt, trainer)
            eva_model = trainer.get_student()

        evaluate_performance(
            eva_model,
            gallery_loader,
            query_loader,
            device,
            use_gt=cfg.EVAL_USE_GT,
            use_cache=cfg.EVAL_USE_CACHE,
            use_cbgm=cfg.EVAL_USE_CBGM,
        )
        exit(0)

    if cfg.BASELINE.PROJECT:
        params = [p for p in trainer.parameters() if p.requires_grad]
    elif cfg.DISTILLER.PROJECT:
        params = trainer.get_learnable_parameters()
    else:
        print("Error")
        exit(0)
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=0.1
    )

    start_epoch = 0
    if args.resume:
        assert args.ckpt, "--ckpt must be specified when --resume enabled"
        if cfg.BASELINE.PROJECT:
            start_epoch = resume_from_ckpt(args.ckpt, trainer, optimizer, lr_scheduler, cfg.BASELINE.PROJECT) + 1
        elif cfg.DISTILLER.PROJECT:
            start_epoch = resume_from_ckpt(args.ckpt, trainer, optimizer, lr_scheduler) + 1

    print("Creating output folder")
    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)
    path = osp.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(cfg.dump())
    print(f"Full config is saved to {path}")
    tfboard = None
    if cfg.TF_BOARD:
        from torch.utils.tensorboard import SummaryWriter

        tf_log_path = osp.join(output_dir, "tf_log")
        mkdir(tf_log_path)
        tfboard = SummaryWriter(log_dir=tf_log_path)
        print(f"TensorBoard files are saved to {tf_log_path}")

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        train_one_epoch(cfg, trainer, optimizer, train_loader, device, epoch, tfboard)
        lr_scheduler.step()

        if (epoch > 9 and (epoch + 1) % cfg.EVAL_PERIOD == 0) or (epoch == cfg.SOLVER.MAX_EPOCHS - 1):
            if cfg.BASELINE.PROJECT:
                eva_model = trainer
            elif cfg.DISTILLER.PROJECT:
                eva_model = trainer.get_student()

            evaluate_performance(
                eva_model,
                gallery_loader,
                query_loader,
                device,
                use_gt=cfg.EVAL_USE_GT,
                use_cache=cfg.EVAL_USE_CACHE,
                use_cbgm=cfg.EVAL_USE_CBGM,
            )

        if (epoch + 1) % cfg.CKPT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            if cfg.BASELINE.PROJECT:
                save_on_master(
                    {
                        "model": trainer.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    osp.join(output_dir, f"epoch_{epoch}.pth"),
                )
            elif cfg.DISTILLER.PROJECT:
                save_on_master(
                    {
                        "model": trainer.get_state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    osp.join(output_dir, f"epoch_{epoch}.pth"),
                )

    if tfboard:
        tfboard.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg", dest="cfg_file", help="Path to configuration file.")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the performance of a given checkpoint."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the specified checkpoint."
    )
    parser.add_argument("--ckpt", help="Path to checkpoint to resume or evaluate.")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    args = parser.parse_args()
    main(args)
