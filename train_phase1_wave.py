"""
train_phase1_wave.py
====================
WaveDCE 单阶段训练脚本：频率解耦 + U-Net bottleneck 阻尼波传播 + 空频一致性损失

输出形式对齐原 train_phase2.py：
1. 每 5 个 epoch 保存测试集最终输出，不保存 input/gt/中间分支图。
2. 保存目录为 results_dir/epoch_XXX/，文件名沿用验证集原文件名。
3. checkpoint 命名为 epoch_XXX_PSNR.pth 与 best_PSNR.pth。
4. 终端打印、TensorBoard 标量命名尽量保持原 train_phase2.py 的形式。

注意：本脚本已经去掉 NAS/MDP/controller/reward/order，只有一个 WaveDCE 主模型。
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU 0
import time
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from DSC_ICME_NAS import DSC_Net
from dataset_phase2 import get_dataloaders
from lossNAS import get_loss_wave_phase1
from utils import calculate_ssim, save_image, torch_to_np


class Config:
    # ===== 路径配置 =====
    save_dir = 'results10_batchsize=2_lr=8e-5/UIEB/checkpoints'
    log_dir = 'results10_batchsize=2_lr=8e-5/UIEB/logs'
    results_dir = 'results10_batchsize=2_lr=8e-5/UIEB/results'

    # ===== 数据路径配置 =====
    train_input_dir = 'dataset/UIEB/train/train-sny600'
    train_label_dir = 'dataset/UIEB/train/train-clear600'
    val_input_dir = 'dataset/UIEB/test/130-test-syn'
    val_label_dir = 'dataset/UIEB/test/130-test-clear'

    # ===== 数据配置 =====
    batch_size = 4
    num_workers = 4
    img_size = 256

    # ===== 训练配置 =====
    total_epochs = 500

    # ===== Optimizer 配置 =====
    # 为了避免 NaN，波动方程 + 频域损失加入后先用保守学习率。
    lr = 5e-5
    weight_decay = 1e-4
    grad_clip = 0.5

    # ===== 日志与保存：对齐原 train_phase2.py =====
    log_interval = 10
    eval_interval = 1
    save_interval = 5
    vis_interval = 5

    # ===== 设备 =====
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_tensor(x: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=max_val, neginf=min_val).clamp(min_val, max_val)


def is_finite_tensor(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item()


@torch.no_grad()
def safe_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred = safe_tensor(pred)
    target = safe_tensor(target)
    mse = torch.mean((pred - target) ** 2).clamp_min(eps)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return float(psnr.item())


class WaveDCETrainer:
    def __init__(self, config):
        self.cfg = config

        # 创建保存目录
        ensure_dir(self.cfg.save_dir)
        ensure_dir(self.cfg.log_dir)
        ensure_dir(self.cfg.results_dir)

        # TensorBoard
        self.writer = SummaryWriter(self.cfg.log_dir)

        # 训练统计
        self.global_step = 0
        self.best_val_psnr = 0.0

        # 初始化模块
        self._init_modules()
        self._init_optimizers()

        # 加载数据：仍然复用 phase2 数据接口，因为它返回真实 input/gt 配对和 file_name。
        self.train_loader, self.val_loader = get_dataloaders(
            train_input_dir=self.cfg.train_input_dir,
            train_label_dir=self.cfg.train_label_dir,
            val_input_dir=self.cfg.val_input_dir,
            val_label_dir=self.cfg.val_label_dir,
            batch_size=self.cfg.batch_size,
            image_size=self.cfg.img_size,
            num_workers=self.cfg.num_workers
        )

        print(f"\n{'='*60}")
        print("WaveDCE Trainer Initialized")
        print(f"{'='*60}")
        print(f"  Device: {self.cfg.device}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"  Results dir: {self.cfg.results_dir}")
        print(f"  Model: Frequency-Decoupled Wave Propagation, no NAS/MDP routing")
        print(f"{'='*60}\n")

    def _init_modules(self):
        """初始化模型和损失函数"""
        print("Initializing modules...")

        # 单模型：不再初始化 CascadedExperts / StateEncoder / Controller / Executor。
        self.model = DSC_Net().to(self.cfg.device)

        # 空频一致性损失：Charbonnier + SSIM + Gradient + Color + FFT amplitude + band loss。
        self.loss_fn = get_loss_wave_phase1().to(self.cfg.device)

        print("✅ All modules initialized")
        print(f"   - Model params: {sum(p.numel() for p in self.model.parameters())} params\n")

    def _init_optimizers(self):
        """初始化优化器和调度器"""
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.lr,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=self.cfg.weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.total_epochs,
            eta_min=1e-6
        )

        print("✅ Optimizer initialized")
        print(f"   - LR: {self.cfg.lr}")
        print(f"   - Grad clip: {self.cfg.grad_clip}\n")

    def train_one_epoch(self, epoch):
        """训练一个 epoch。输出统计字段尽量对齐原 train_phase2.py。"""
        self.model.train()

        epoch_stats = {
            'loss_dyn': 0.0,
            'psnr_dyn': 0.0,
            'skipped': 0,
            'grad_norm': 0.0,
            'loss_freq': 0.0,
            'loss_color_balance': 0.0,
            'loss_illum': 0.0,
        }

        valid_samples = 0
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch+1}/{self.cfg.total_epochs} [WaveDCE]'
        )

        for batch_idx, (degraded, clean, _) in enumerate(pbar):
            degraded = safe_tensor(degraded.to(self.cfg.device))
            clean = safe_tensor(clean.to(self.cfg.device))
            batch_size = degraded.size(0)

            # 1. 前向
            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(degraded)
            pred = safe_tensor(pred)

            # 防 NaN：输出异常则跳过该 batch，避免污染权重。
            if not is_finite_tensor(pred):
                epoch_stats['skipped'] += batch_size
                continue

            # 2. 计算损失
            loss, loss_dict = self.loss_fn(pred, clean)

            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                epoch_stats['skipped'] += batch_size
                continue

            # 3. 反向与梯度裁剪
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            if not torch.isfinite(grad_norm):
                self.optimizer.zero_grad(set_to_none=True)
                epoch_stats['skipped'] += batch_size
                continue

            self.optimizer.step()

            # 4. 统计
            psnr_dyn = safe_psnr(pred, clean)
            epoch_stats['loss_dyn'] += float(loss.item()) * batch_size
            epoch_stats['psnr_dyn'] += psnr_dyn * batch_size
            epoch_stats['grad_norm'] += float(grad_norm) * batch_size
            epoch_stats['loss_freq'] += float(loss_dict.get('freq', 0.0)) * batch_size
            epoch_stats['loss_color_balance'] += float(loss_dict.get('color_balance', 0.0)) * batch_size
            epoch_stats['loss_illum'] += float(loss_dict.get('illum', 0.0)) * batch_size
            valid_samples += batch_size

            pbar.set_postfix({
                'L_dyn': f"{loss.item():.4f}",
                'PSNR': f"{psnr_dyn:.2f}",
                'GN': f"{float(grad_norm):.3f}",
                'Skip': epoch_stats['skipped']
            })

            # 5. TensorBoard：保留原 train_phase2.py 的核心命名
            if batch_idx % self.cfg.log_interval == 0:
                self._log_training(
                    loss_dyn=float(loss.item()),
                    psnr_dyn=psnr_dyn,
                    grad_norm=float(grad_norm),
                    loss_dict=loss_dict
                )

            self.global_step += 1

        if valid_samples > 0:
            for key in ['loss_dyn', 'psnr_dyn', 'grad_norm', 'loss_freq', 'loss_color_balance', 'loss_illum']:
                epoch_stats[key] /= valid_samples
        else:
            print("⚠️ 当前 epoch 所有 batch 都被跳过，请检查学习率、数据范围或 loss 是否异常。")

        return epoch_stats

    def _log_training(self, loss_dyn, psnr_dyn, grad_norm, loss_dict):
        """精简版训练日志：形式对齐原 train_phase2.py。"""
        self.writer.add_scalar('Train/Total_Loss', loss_dyn, self.global_step)
        self.writer.add_scalar('Train/PSNR', psnr_dyn, self.global_step)
        self.writer.add_scalar('Train/GradNorm', grad_norm, self.global_step)

        # 新增空频损失分量，但不会改变原有主输出形式。
        self.writer.add_scalar('Loss/Frequency', float(loss_dict.get('freq', 0.0)), self.global_step)
        self.writer.add_scalar('Loss/ColorBalance', float(loss_dict.get('color_balance', 0.0)), self.global_step)
        self.writer.add_scalar('Loss/Illumination', float(loss_dict.get('illum', 0.0)), self.global_step)

    @torch.no_grad()
    def validate(self, epoch: int = 0, save_images: bool = False):
        """验证函数。保存形式对齐原 train_phase2.py：只保存最终输出，文件名沿用原图。"""
        self.model.eval()

        val_stats = {
            'loss_dyn': 0.0,
            'psnr_dyn': 0.0,
            'ssim_dyn': 0.0,
        }

        if save_images:
            output_dir = os.path.join(self.cfg.results_dir, f'epoch_{epoch+1:03d}')
            os.makedirs(output_dir, exist_ok=True)
            print(f"💾 Saving test results to {output_dir}")

        pbar = tqdm(self.val_loader, desc='Validation')

        for batch_idx, batch_data in enumerate(pbar):
            degraded, clean, file_names = batch_data
            degraded = safe_tensor(degraded.to(self.cfg.device))
            clean = safe_tensor(clean.to(self.cfg.device))

            pred_dyn = self.model(degraded)
            pred_dyn = safe_tensor(pred_dyn)

            loss_dyn, _ = self.loss_fn(pred_dyn, clean)
            psnr_dyn = safe_psnr(pred_dyn, clean)
            ssim_dyn = calculate_ssim(pred_dyn, clean)
            ssim_dyn = float(ssim_dyn.item() if hasattr(ssim_dyn, 'item') else ssim_dyn)

            # 保存预测结果：不保存 input/gt，不保存分支图，只保存最终输出。
            if save_images:
                batch_size = pred_dyn.shape[0]
                for i in range(batch_size):
                    pred_np = torch_to_np(pred_dyn[i])

                    if isinstance(file_names, (list, tuple)):
                        file_name = file_names[i]
                    else:
                        file_name = f'img_{batch_idx * batch_size + i:04d}.png'

                    if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_name = f"{file_name}.png"

                    save_image(file_name, pred_np, output_dir)

            batch_size = degraded.size(0)
            val_stats['loss_dyn'] += float(loss_dyn.item()) * batch_size
            val_stats['psnr_dyn'] += psnr_dyn * batch_size
            val_stats['ssim_dyn'] += ssim_dyn * batch_size

        num_samples = len(self.val_loader.dataset)
        val_stats['loss_dyn'] /= num_samples
        val_stats['psnr_dyn'] /= num_samples
        val_stats['ssim_dyn'] /= num_samples

        self._print_val_results(epoch, val_stats)

        # TensorBoard 名字保留 Dynamic，是为了兼容你原先看曲线的方式。
        # 这里的 Dynamic 实际表示 WaveDCE 单模型输出，不表示动态路由。
        self.writer.add_scalar('Val/PSNR_Dynamic', val_stats['psnr_dyn'], epoch)
        self.writer.add_scalar('Val/SSIM_Dynamic', val_stats['ssim_dyn'], epoch)
        self.writer.add_scalar('Val/Loss_Dynamic', val_stats['loss_dyn'], epoch)

        if save_images:
            total_images = len(self.val_loader.dataset)
            print(f"✅ Saved {total_images} output images (epoch {epoch}):")
            print(f"   - {output_dir}\n")

        return val_stats

    def _print_val_results(self, epoch, stats):
        """打印验证结果：结构对齐原 train_phase2.py，但去掉固定顺序对比。"""
        print(f"\n{'='*60}")
        print(f"Validation Results (Epoch {epoch+1})")
        print(f"{'='*60}")

        print("WaveDCE Output:")
        print(f"  PSNR: {stats['psnr_dyn']:.2f} dB")
        print(f"  SSIM: {stats['ssim_dyn']:.4f}")
        print(f"  Loss: {stats['loss_dyn']:.4f}")

        print("\nNote:")
        print("  NAS/MDP routing has been removed. No fixed-order comparison is printed.")
        print(f"{'='*60}\n")

    def save_checkpoint(self, epoch, stats, is_best=False, save_regular=False):
        """
        保存训练检查点。

        保存规则：
        1. latest.pth：每次验证后覆盖保存；
        2. best.pth：验证 PSNR 创新高时覆盖保存；
        3. best_info.txt：记录 best.pth 对应的 epoch、PSNR、SSIM；
        4. epoch_XXX_YY.YY.pth：按 save_interval 保存普通 checkpoint。
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'val_psnr': stats['psnr_dyn'],
            'val_ssim': stats['ssim_dyn'],
            'best_val_psnr': self.best_val_psnr,
            'model_type': 'WaveDCE_Frequency_Decoupled_No_Routing',
            'note': 'No NAS/MDP/controller/order. FTD-WaveUIE single-model training.'
        }

        psnr_value = stats['psnr_dyn']

        # 1. 每次验证后覆盖 latest.pth
        latest_path = os.path.join(self.cfg.save_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)

        # 2. 按 save_interval 保存普通 checkpoint
        if save_regular:
            save_path = os.path.join(
                self.cfg.save_dir,
                f'epoch_{epoch + 1:03d}_{psnr_value:.2f}.pth'
            )
            torch.save(checkpoint, save_path)
            print(f"✅ Checkpoint saved: {save_path}")

        # 3. 只保留一个 best.pth，并写 best_info.txt
        if is_best:
            best_path = os.path.join(self.cfg.save_dir, 'best.pth')
            torch.save(checkpoint, best_path)

            best_info_path = os.path.join(self.cfg.save_dir, 'best_info.txt')
            with open(best_info_path, 'w', encoding='utf-8') as f:
                f.write("Best Model Information\n")
                f.write("=" * 40 + "\n")
                f.write(f"Best Epoch: {epoch + 1}\n")
                f.write(f"Best PSNR: {psnr_value:.4f} dB\n")
                f.write(f"Best SSIM: {stats['ssim_dyn']:.6f}\n")
                f.write(f"Global Step: {self.global_step}\n")
                f.write(f"Checkpoint Path: {best_path}\n")
                f.write(f"Latest Path: {latest_path}\n")
                f.write(f"Model Type: WaveDCE_Frequency_Decoupled_No_Routing\n")
                f.write("=" * 40 + "\n")

            print(f"🏆 Best checkpoint updated: {best_path} | PSNR={psnr_value:.2f} dB")
            print(f"📝 Best info updated: {best_info_path}")

    def train(self):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print("Starting WaveDCE Training")
        print(f"{'='*60}\n")

        for epoch in range(self.cfg.total_epochs):
            epoch_start = time.time()

            # 1. 训练一个 epoch
            train_stats = self.train_one_epoch(epoch)

            # 2. 打印训练统计：形式接近原 train_phase2.py
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Loss (WaveDCE): {train_stats['loss_dyn']:.4f}")
            print(f"  PSNR (WaveDCE): {train_stats['psnr_dyn']:.2f} dB")
            print(f"  Grad Norm: {train_stats['grad_norm']:.4f}")
            print(f"  Frequency Loss: {train_stats['loss_freq']:.4f}")
            print(f"  Color Balance Loss: {train_stats['loss_color_balance']:.4f}")
            print(f"  Illumination Loss: {train_stats['loss_illum']:.4f}")
            print(f"  Skipped Samples: {train_stats['skipped']}")
            print(f"  Time: {time.time() - epoch_start:.1f}s")

            # 3. 验证与可视化
            if (epoch + 1) % self.cfg.eval_interval == 0:
                should_save_images = (epoch == 0) or ((epoch + 1) % self.cfg.vis_interval == 0)
                val_stats = self.validate(epoch, save_images=should_save_images)
                
                
                # 更新最佳模型记录
                #最佳模型不是根据训练过程直接更新的，而是每个 epoch 训练完成后，先在验证集上“考试”，
                #再根据验证 PSNR 判断当前模型是否最好；可视化只是顺便保存输出图像，真正决定 best 的是验证指标，不是图像保存。
                is_best = val_stats['psnr_dyn'] > self.best_val_psnr
                if is_best:
                    self.best_val_psnr = val_stats['psnr_dyn']

                # 是否保存普通 epoch checkpoint
                save_regular = ((epoch + 1) % self.cfg.save_interval == 0)

                # 保存 checkpoint：
                # latest.pth 每次覆盖；
                # best.pth 只保留一个；
                # epoch_XXX_YY.YY.pth 按 save_interval 保存。
                self.save_checkpoint(
                    epoch,
                    val_stats,
                    is_best=is_best,
                    save_regular=save_regular
                )

            # 4. 学习率调度和日志
            self.scheduler.step()
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            print(f"-"*30)

        print(f"\n{'='*60}")
        print("Training Completed!")
        print(f"Best Val PSNR: {self.best_val_psnr:.2f} dB")
        print(f"{'='*60}\n")

        self.writer.close()


def main():
    config = Config()

    # 1. 打印配置信息：形式对齐原 train_phase2.py
    print("\n" + "="*60)
    print("WaveDCE: Frequency-Decoupled Training Configuration")
    print("="*60)
    print(f"Train input:  {config.train_input_dir}")
    print(f"Train label:  {config.train_label_dir}")
    print(f"Val input:    {config.val_input_dir}")
    print(f"Val label:    {config.val_label_dir}")
    print(f"Batch size:   {config.batch_size}")
    print(f"Total epochs: {config.total_epochs}")
    print(f"Vis interval: every {config.vis_interval} epochs")
    print(f"Save interval: every {config.save_interval} epochs")
    print(f"Results dir:  {config.results_dir}")
    print("="*60)

    # 2. 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    # 3. 开始训练
    trainer = WaveDCETrainer(config)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Closing SummaryWriter...")
        trainer.writer.close()


if __name__ == '__main__':
    main()
