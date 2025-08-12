import hydra
from omegaconf import DictConfig, OmegaConf

import logging
from accelerate.logging import get_logger
from accelerate import Accelerator
from accelerate.utils import set_seed
from schedulefree import RAdamScheduleFree

logger = get_logger(__file__)

set_seed(0xAAAA)
accelerator = Accelerator()

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
import copy

from vqvae2 import VQVAE, VQVAE2
from data import get_dataset
from utils import init_wandb, MetricGroup, setup_directory

import wandb as wandb_module


def save_model(net, path):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        net = accelerator.unwrap_model(net)
        accelerator.save(net.state_dict(), path)


# TODO: in general, all the logging needs a rework
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    logger.info("Loaded Hydra config:")
    logger.info(OmegaConf.to_yaml(cfg))

    assert (
        cfg.vqvae.training.batch_size % accelerator.num_processes == 0
    ), "Batch size must be divisible by number of Acccelerate processes"
    cfg.vqvae.training.batch_size //= accelerator.num_processes

    exp_dir = setup_directory(cfg=cfg)
    checkpoint_dir = exp_dir / "checkpoints"
    recon_dir = exp_dir / "recon"

    checkpoint_dir.mkdir(exist_ok=True)
    recon_dir.mkdir(exist_ok=True)

    if accelerator.is_main_process:
        wandb = init_wandb(cfg, exp_dir)
    accelerator.wait_for_everyone()

    @torch.cuda.amp.autocast(enabled=cfg.vqvae.training.amp)
    def loss_fn(net, x, eval=False):
        recon, _, idx, diff = net(x)
        if eval:
            x, recon = accelerator.gather_for_metrics((x, recon))
        mse_loss = F.mse_loss(recon, x)
        # mse_loss = F.l1_loss(recon, x)
        return mse_loss + diff * cfg.vqvae.training.beta, mse_loss, diff, recon, idx

    net = VQVAE2.build_from_config(
        cfg.vqvae.model, codebook_gumbel_temperature=0.1, codebook_init_type="kaiming_uniform", codebook_cosine=True
    )
    # optim = torch.optim.AdamW(net.parameters(), lr=cfg.vqvae.training.lr)
    optim = opt = RAdamScheduleFree(net.parameters(), lr=cfg.vqvae.training.lr, weight_decay=0.01)
    train_loader, test_loader = get_dataset(cfg)

    net, optim, train_loader, test_loader = accelerator.prepare(net, optim, train_loader, test_loader)

    columns = [f"codeword_{i:04}" for i in range(cfg.vqvae.model.codebook_size)]
    idx_table = [
        # wandb_module.Table(columns=[f"codeword_{i:04}" for i in range(cfg.vqvae.model.codebook_size)])
        []
        for _ in cfg.vqvae.model.resample_factors
    ]

    steps = 0
    preloss = 1000
    best_loss = 1000
    increasing = 0
    max_steps = cfg.vqvae.training.max_steps
    while steps <= max_steps:
    # while increasing <= 10:
        wandb_log = {}
        it = train_loader
        if accelerator.is_local_main_process:
            it = tqdm(train_loader)

        metrics = MetricGroup("loss", "mse_loss", "kl_loss")
        total_idx = [torch.zeros(cfg.vqvae.model.codebook_size).cpu().long() for _ in cfg.vqvae.model.resample_factors]
        net.train()
        optim.train()
        for batch in it:
            if isinstance(batch, (list, tuple)):
                batch, *_ = batch
            optim.zero_grad()
            loss, *m, _, idx = loss_fn(net, batch)
            accelerator.backward(loss)
            # max_norm = 1.0  # 勾配ノルムの最大値
            # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)
            #     # 勾配クリッピングの実装
            optim.step()

            for i in range(len(idx)):
                total_idx[i] += torch.bincount(idx[i].cpu().detach().flatten(), minlength=cfg.vqvae.model.codebook_size)

            metrics.log(loss, *m)

            if steps > max_steps:
            # if increasing >= 10 or steps > max_steps:
                save_model(best_net, checkpoint_dir / f"state_dict_final.pt")
                break

            steps += 1

            if steps % cfg.vqvae.training.save_frequency == 0:
                save_model(net, checkpoint_dir / f"state_dict_{steps:06}.pt")

        if steps <= max_steps:
        # if increasing <= 10:
            metrics.print_summary(f"training {steps}/{max_steps}")
            if accelerator.is_main_process:
                wandb_log["train"] = {}

                # for i in range(len(total_idx)):
                #     total_vectors_in_epoch = total_idx[i].sum()
                #     if total_vectors_in_epoch > 0:
                #         proportion = total_idx[i].float() / total_vectors_in_epoch # .float() を追加して型を保証
                #     else:
                #         proportion = torch.zeros_like(total_idx[i], dtype=torch.float32)

                #     # --- ▼▼▼ ここからが修正・追加部分 ▼▼▼ ---

                #     # 1. エントロピーを計算 (値が0の場合のlog(0)を避けるため、微小な値 1e-9 を足す)
                #     entropy = -torch.sum(proportion * torch.log(proportion + 1e-9))

                #     # 2. wandb_log にエントロピーを追加
                #     wandb_log["train"].update({f"codebook_entropy.{i}": entropy.item()})

                #     # 3. 既存の未使用率のログはそのまま残す
                #     wandb_log["train"].update(
                #         {f"unused_codewords_proportion.{i}": (total_idx[i] == 0).sum() / cfg.vqvae.model.codebook_size}
                #     )

                # 4. 巨大な line_series を送る処理は削除する
                # wandb_log.update({ f"codebook_usage.{i}": wandb_module.plot.line_series(...) }) # この行を削除またはコメントアウト

                for i in range(len(total_idx)):
                    # idx_table[i].add_data(*((total_idx[i] / (idx[i].numel() * len(train_loader))).tolist()))
                    idx_table[i].append((total_idx[i] / (idx[i].numel() * len(train_loader))))

                    wandb_log.update(
                        {
                            f"codebook_usage.{i}": wandb_module.plot.line_series(
                                xs=list(range(len(idx_table[i]))),
                                ys=torch.stack(idx_table[i], dim=-1).numpy(),
                                keys=columns,
                                xname="Epochs",
                            )
                        }
                    )
                    wandb_log["train"].update(
                        {f"unused_codewords_proportion.{i}": (total_idx[i] == 0).sum() / cfg.vqvae.model.codebook_size}
                    )

                wandb_log["train"].update(metrics.summarise())

        metrics = MetricGroup("loss", "mse_loss", "kl_loss")
        net.eval()
        optim.eval()
        with torch.no_grad():
            total_idx = [
                torch.zeros(cfg.vqvae.model.codebook_size).cpu().long() for _ in cfg.vqvae.model.resample_factors
            ]
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    batch, *_ = batch
                loss, *m, recon, idx = loss_fn(net, batch)
                metrics.log(loss, *m)
                for i in range(len(idx)):
                    total_idx[i] += torch.bincount(idx[i].cpu().flatten(), minlength=cfg.vqvae.model.codebook_size)

        loss = metrics.summarise()["loss"]
        mse_loss = metrics.summarise()["mse_loss"]        
        print(f"preloss: {preloss:.8f}, loss: {mse_loss:.8f}")
        if preloss > mse_loss:
            increasing = 0
            if best_loss > mse_loss:
                best_net = copy.deepcopy(net)
                best_loss=mse_loss
        else:
            increasing += 1
        preloss = mse_loss

        metrics.print_summary(f"evaluation {steps}/{max_steps} {increasing}")

        if accelerator.is_main_process:
            wandb_log["eval"] = {}
            save_image(
                torch.concat([batch, recon], axis=0),
                recon_dir / f"recon_{steps:06}.png",
                nrow=len(recon),
                normalize=True,
            )

            # print(f"Shape of batch: {batch.shape}")
            input_images = []
            for img in batch:
                img_np = img.squeeze().cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                img_np_transposed = np.transpose(img_np, (1, 2, 0))  # 次元を転置
                input_images.append(wandb_module.Image(img_np_transposed, caption="Input Image"))

            # print(f"Shape of recon: {recon.shape}")
            recon_images = []
            for img in recon:
                img_np = img.squeeze().cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                img_np_transposed = np.transpose(img_np, (1, 2, 0))  # 次元を転置
                recon_images.append(wandb_module.Image(img_np_transposed, caption="Reconstruction"))

            wandb_log.update(
                {
                    "input": input_images,
                    "recon": recon_images,
                }
            )

            # wandb_log.update(
            #     {
            #         "input": wandb_module.Image(batch.permute(0, 2, 3, 1).squeeze(-1), caption="Input Image", mode="L"),
            #         "recon": wandb_module.Image(recon.permute(0, 2, 3, 1).squeeze(-1), caption="Reconstruction", mode="L"),
            #     }
            # )
            for i in range(len(total_idx)):
                wandb_log["eval"].update(
                    {f"unused_codewords_proportion.{i}": (total_idx[i] == 0).sum() / cfg.vqvae.model.codebook_size}
                )
            wandb_log["eval"].update(metrics.summarise())
            wandb.log(wandb_log)
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main(None)
