import argparse
import os
import time
import yaml
import torch
import tqdm
import numpy as np

from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam

from dataset.dataset import SeqDataset
from dataset.pt_seq import PTSeqDataset
from model.model import E2VIDRecurrent
from model.loss import LPIPSLoss, TCLoss
from utils.loading_utils import load_model, get_device
from utils.data_utils import SeqCrop128

def _as_float(x, name=""):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            raise ValueError(f"Config field {name} should be numeric, got: {x!r}")
    raise ValueError(f"Config field {name} should be numeric, got type {type(x)}")

def save_checkpoint(save_dir, epoch, model, optimizer, tag=None, name=None):
    os.makedirs(save_dir, exist_ok=True)
    if name is None:
        name = f"adapter_epoch_{epoch:04d}.pth.tar" if tag is None else f"adapter_{tag}.pth.tar"
    path = os.path.join(save_dir, name)
    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }, path)
    print(f"[ckpt] saved → {path}")


def train_one_epoch(
    e2vid,
    train_loader,
    loss_obj,
    loss_weights,
    optimizer,
    device,
    num_bins: int = 5,
    L0: int = 2,
    writer: SummaryWriter = None,
    epoch_idx: int = 0,
    grad_clip: float = 1.0,
):
    e2vid.train()

    l1loss, lpipsloss, tcloss = loss_obj
    l1_weight, lpips_weight, tc_weight = loss_weights
    epoch_loss = {"total": 0.0, "l1": 0.0, "lpips": 0.0, "tc": 0.0}
    num_batches = len(train_loader)
    
    pbar = tqdm.tqdm(train_loader, desc=f"[Train] Epoch {epoch_idx+1}", ncols=120)
    for batch_idx, batch in enumerate(pbar):
        frames = batch["frames"].to(device)   # [B,T+1,1,H,W] 
        events = batch["events"].to(device)   # [B,T,Bin,H,W]
        flows  = batch["flows"].to(device)    # [B,T,2,H,W]

        _, T1, _, _, _ = frames.shape
        T = T1 - 1

        prev_states = None
        p0 = None

        batch_loss = {"total": 0.0, "l1": 0.0, "lpips": 0.0, "tc": 0.0}
        optimizer.zero_grad(set_to_none=True)

        for t in range(T):
            f0 = frames[:, t]
            f1 = frames[:, t+1]
            ev = events[:, t]
            fl = flows[:, t]
            
            if p0 is None:
                p0 = f0

            recon, states = e2vid(ev, prev_states)
            
            l1_loss = l1loss(recon, f1)
            if t < L0:
                tc_loss = 0
            else:
                tc_loss = tcloss(f0, f1, p0, recon, fl)
            lpips_loss = lpipsloss(recon, f1)
            
            batch_loss["total"] += (l1_weight * l1_loss + lpips_weight * lpips_loss + tc_weight * tc_loss)
            batch_loss["l1"]    += l1_weight * l1_loss
            batch_loss["lpips"] += lpips_weight * lpips_loss
            batch_loss["tc"]    += tc_weight * tc_loss
            
            prev_states = states
            p0 = recon 

        batch_loss["total"].backward()
        optimizer.step()
        
        batch_avg_total = batch_loss["total"].item() / T
        batch_avg_l1    = batch_loss["l1"].item()    / T
        batch_avg_lpips = batch_loss["lpips"].item() / T
        batch_avg_tc    = batch_loss["tc"].item()    / T

        # tqdm 显示
        pbar.set_postfix({
            "loss":  f"{batch_avg_total:.4f}",
            "l1":    f"{batch_avg_l1:.4f}",
            "lpips": f"{batch_avg_lpips:.4f}",
            "tc":    f"{batch_avg_tc:.4f}",
        })

        # 累加到 epoch（按 batch 平均）
        epoch_loss["total"] += batch_avg_total
        epoch_loss["l1"]    += batch_avg_l1
        epoch_loss["lpips"] += batch_avg_lpips
        epoch_loss["tc"]    += batch_avg_tc

    # 计算 epoch 平均
    epoch_avg = {k: v / max(1, num_batches) for k, v in epoch_loss.items()}

    # 写 TensorBoard（按 epoch）
    if writer is not None:
        writer.add_scalar("train/epoch_total_loss", epoch_avg["total"], epoch_idx + 1)
        writer.add_scalar("train/epoch_l1_loss",    epoch_avg["l1"],    epoch_idx + 1)
        writer.add_scalar("train/epoch_lpips_loss", epoch_avg["lpips"], epoch_idx + 1)
        writer.add_scalar("train/epoch_tc_loss",    epoch_avg["tc"],    epoch_idx + 1)

        # 记录不同 param group 的 lr
        for gi, g in enumerate(optimizer.param_groups):
            writer.add_scalar(f"train/lr_group{gi}", g["lr"], epoch_idx + 1)

    return epoch_avg  # dict: {'total','lpips','tc'}


@torch.no_grad()
def validate_one_epoch(
    e2vid,
    val_loader,
    loss_obj,
    loss_weights,
    device,
    L0: int = 2,
    num_bins: int = 5,
    writer: SummaryWriter = None,
    epoch_idx: int = 0,
    vis_first_batch: bool = True,
):
    e2vid.eval()
    l1loss, lpipsloss, tcloss = loss_obj
    l1_weight, lpips_weight, tc_weight = loss_weights

    epoch_loss = {"total": 0.0, "l1": 0.0, "lpips": 0.0, "tc": 0.0}
    num_steps_sum = 0

    pbar = tqdm.tqdm(val_loader, desc=f"[Val]   Epoch {epoch_idx+1}", ncols=120)
    first_vis_done = False

    for batch in pbar:
        frames = batch["frames"].to(device)   # [B,T+1,1,H,W]
        events = batch["events"].to(device)   # [B,T,Bin,H,W]
        flows  = batch["flows"].to(device)    # [B,T,2,H,W]

        _, T1, _, _, _ = frames.shape
        T = T1 - 1

        current_L = 0
        prev_states = None
        p0 = None

        for t in range(T):
            f0 = frames[:, t]
            f1 = frames[:, t+1]
            ev = events[:, t]
            fl = flows[:, t]
            
            if p0 is None:
                p0 = f0

            recon, states = e2vid(ev, prev_states)
            
            l1_loss = l1loss(recon, f1)
            if current_L < L0:
                tc_loss = 0
                current_L += 1
            else:
                tc_loss = tcloss(f0, f1, p0, recon, fl)
            lpips_loss = lpipsloss(recon, f1)

            epoch_loss["total"] += (l1_weight * l1_loss + lpips_weight * lpips_loss + tc_weight * tc_loss)
            epoch_loss["l1"]    += l1_weight * l1_loss
            epoch_loss["lpips"] += lpips_weight * lpips_loss
            epoch_loss["tc"]    += tc_weight * tc_loss

            num_steps_sum += 1

            prev_states = states
            p0 = recon

    # 计算 epoch 平均（按 step）
    if num_steps_sum == 0:
        avg = {k: 0.0 for k in epoch_loss}
    else:
        avg = {k: v / num_steps_sum for k, v in epoch_loss.items()}

    # 写 TensorBoard
    if writer is not None:
        writer.add_scalar("val/epoch_total_loss", avg["total"], epoch_idx + 1)
        writer.add_scalar("val/epoch_l1_loss",    avg["l1"],    epoch_idx + 1)
        writer.add_scalar("val/epoch_lpips_loss", avg["lpips"], epoch_idx + 1)
        writer.add_scalar("val/epoch_tc_loss",    avg["tc"],    epoch_idx + 1)

    return avg

def main():
    parser = argparse.ArgumentParser(description="Train Adapter for E2VID Recurrent Model")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file for the model.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_config   = config["model"]
    train_config   = config["train"]

    save_dir = train_config.get("save_dir", "./logs/adapter_training/")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)
    device = get_device(use_gpu=True)

    e2vid = E2VIDRecurrent(model_config).to(device)

    lr_e2vid     = _as_float(train_config.get("lr_e2vid",   1e-4), "train.lr_e2vid")
    weight_decay = _as_float(train_config.get("weight_decay", 0.0), "train.weight_decay")

    params_group = [
        {"params": e2vid.parameters(),   "lr": lr_e2vid},
    ]
    optimizer = Adam(params_group, weight_decay=weight_decay)

    # Loss
    loss_obj = (
        L1Loss(),
        LPIPSLoss(use_gpu=True),
        TCLoss(alpha=50.0),
    )
    loss_weights = (
        _as_float(train_config.get("l1_weight", 1.0), "train.l1_weight"),
        _as_float(train_config.get("lpips_weight", 1.0), "train.lpips_weight"),
        _as_float(train_config.get("tc_weight", 5.0), "train.tc_weight"),
    )

    # 数据
    train_dataset = SeqDataset(
        root=train_config["dataroot"], 
        split="train", 
        transform=SeqCrop128(mode="random", pad_if_small=True),
        num_iter=train_config.get("num_iter", 30)
    )
    val_dataset = SeqDataset(
        root=train_config["dataroot"], 
        split="validation", 
        transform=SeqCrop128(mode="center", pad_if_small=True),
        num_iter=train_config.get("num_iter", 30)
    )
    # train_dataset = PTSeqDataset(
    #     root=train_config["dataroot"], 
    #     split="train", 
    #     transform=SeqCrop128(mode="random", pad_if_small=True),
    #     num_iter=train_config.get("num_iter", 30)
    # )
    # val_dataset = PTSeqDataset(
    #     root=train_config["dataroot"], 
    #     split="validation", 
    #     transform=SeqCrop128(mode="center", pad_if_small=True),
    #     num_iter=train_config.get("num_iter", 30)
    # )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=train_config.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=train_config.get("num_workers", 4),
        pin_memory=True,
    )

    # 训练循环
    epochs     = train_config["num_epochs"]
    save_freq  = train_config.get("save_freq", 10)
    num_bins   = model_config["num_bins"]
    grad_clip  = train_config.get("grad_clip", 1.0)
    
    # scheduler = None
    # scheduler_config = train_config.get("scheduler", None)
    # if scheduler_config is not None:
    #     if scheduler_config["type"] == "CosineAnnealingLR":
    #         T_max = int(scheduler_config.get("T_max", epochs))
    #         eta_min = _as_float(scheduler_config.get("eta_min", 1e-7), "scheduler.eta_min")
    #         scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    #         print(f"[sched] CosineAnnealingLR with T_max={T_max}, eta_min={eta_min}")
    #     else:
    #         print(f"[sched] unknown scheduler type: {scheduler_config['type']}, skip.")
            
    best_val = float("inf")
    
    val_avg = {"total": float("inf"), "lpips": float("inf"), "tc": float("inf")}
    for epoch in range(epochs):
        train_avg = train_one_epoch(
            e2vid=e2vid,
            train_loader=train_loader,
            loss_obj=loss_obj,
            loss_weights=loss_weights,
            optimizer=optimizer,
            device=device,
            num_bins=num_bins,
            writer=writer,
            epoch_idx=epoch,
            grad_clip=grad_clip
        )

        if (epoch + 1) % train_config.get("val_freq", 10) == 0:
            val_avg = validate_one_epoch(
                e2vid=e2vid,
                val_loader=val_loader,
                loss_obj=loss_obj,
                loss_weights=loss_weights,
                device=device,
                num_bins=num_bins,
                writer=writer,
                epoch_idx=epoch
            )

        print(f"[Epoch {epoch+1}/{epochs}]",
              f"Train(total/l1/lpips/tc): {train_avg['total']:.4f}/{train_avg['l1']:.4f}/{train_avg['lpips']:.4f}/{train_avg['tc']:.4f}")
        
        # if scheduler:
        #     scheduler.step()

        # 保存策略：定频保存 + 追踪最优 val
        if (epoch + 1) % save_freq == 0:
            save_checkpoint(save_dir, epoch + 1, e2vid, optimizer, name=f"e2vid_epoch_{epoch+1:04d}.pth.tar")
        if val_avg["total"] < best_val:
            best_val = val_avg["total"]
            save_checkpoint(save_dir, epoch + 1, e2vid, optimizer, name="e2vid_best.pth.tar")

    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()
