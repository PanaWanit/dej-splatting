import os
from accelerate import Accelerator
from torch.optim import Adam
from lib.data_utils import read_all
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from torch.profiler import profile, ProfilerActivity
from lib.camera_utils import to_viewpoint_camera

import contextlib

USE_GPU_PYTORCH = True
USE_PROFILE = False


def exists(x):
    return x is not None


class GaussTrainer:
    def __init__(self, model, config, **kwargs):
        super().__init__()

        self.data = read_all(
            os.path.join(config.database_path, "image.txt"),
            os.path.join(config.database_path, "camera.txt"),
        )

        ##### TODO: Implement gauss renderer
        self.gaussRender = GaussRenderer(**kwargs.get("render_kwargs", {}))
        ###

        self.lambda_dssim = config.lambda_dssim
        self.lambda_depth = config.lambda_depth

        self.accelerator = Accelerator(
            split_batches=config.split_batches,
            mixed_precision="fp16" if config.fp16 else "no",
            project_dir=config.results_folder if config.with_tracking else None,
            log_with="all",
        )

        self.accelerator.native_amp = config.amp

        self.model = model

        self.train_lr = config.lr
        self.train_batch_size = config.batch_size
        self.train_num_steps = config.epochs

        self.i_save = config.i_save
        self.i_print = config.i_print
        self.i_image = config.i_image

        self.results_folder = config.results_folder
        self.gradient_accumulate_every = config.gradient_accumulate_every
        self.with_tracking = config.with_tracking
        self.step = 0

        self.opt = Adam(self.model.parameters(), lr=config.train_lr, betas=config.adam_betas)

        if self.accelerator.is_main_process:
            self.results_folder = Path(config.results_folder)
            self.results_folder.mkdir(exist_ok=True)

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # tracking
        if self.with_tracking:
            run = os.path.split(__file__)[-1].split(".")[0]
            self.accelerator.init_trackers(
                run,
                config={
                    "train_lr": config.train_lr,
                    "train_batch_size": config.train_batch_size,
                    "gradient_accumulate_every": config.gradient_accumulate_every,
                    "train_num_steps": config.train_num_steps,
                },
            )

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "scaler": self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    def load(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f"model-{milestone}.pt"), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def on_train_step(self):
        # TODO: adapt to index
        ind = np.random.choice(len(self.data))
        data = self.data[ind]

        # TODO: Implement camera util
        # NOTE: to_view_point pass data instead
        if USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(data["w2c"], data["camera_df"])

        if USE_PROFILE:
            prof = profile(activities=[ProfilerActivity.CUDA], with_stack=True)
        else:
            prof = contextlib.nullcontext()

        with prof:
            out = self.gaussRender(pc=self.model, camera=camera)

        if USE_PROFILE:
            print(prof.key_averages(group_by_stack_n=True).table(sort_by="self_cuda_time_total", row_limit=20))
        # TODO: Implement loss utils
        l1_loss = loss_utils.l1_loss(out["render"], rgb)
        depth_loss = loss_utils.l1_loss(out["depth"][..., 0][mask], depth[mask])
        ssim_loss = 1.0 - loss_utils.ssim(out["render"], rgb)

        total_loss = (1 - self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss + depth_loss * self.lambda_depth
        psnr = utils.img2psnr(out["render"], rgb)
        log_dict = {"total": total_loss, "l1": l1_loss, "ssim": ssim_loss, "depth": depth_loss, "psnr": psnr}

        return total_loss, log_dict

    def on_evaluate_step(self, **kwargs):
        import matplotlib.pyplot as plt

        ind = np.random.choice(len(self.data["camera"]))
        camera = self.data["camera"][ind]
        if USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(camera)

        rgb = self.data["rgb"][ind].detach().cpu().numpy()
        out = self.gaussRender(pc=self.model, camera=camera)
        rgb_pd = out["render"].detach().cpu().numpy()
        depth_pd = out["depth"].detach().cpu().numpy()[..., 0]
        depth = self.data["depth"][ind].detach().cpu().numpy()
        depth = np.concatenate([depth, depth_pd], axis=1)
        depth = 1 - depth / depth.max()
        depth = plt.get_cmap("jet")(depth)[..., :3]
        image = np.concatenate([rgb, rgb_pd], axis=1)
        image = np.concatenate([image, depth], axis=0)
        utils.imwrite(str(self.results_folder / f"image-{self.step}.png"), image)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):

                    with self.accelerator.autocast():
                        loss, log_dict = self.on_train_step()
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss

                    self.accelerator.backward(loss)

                # all reduce to get the total loss
                total_loss = accelerator.reduce(total_loss)
                total_loss = total_loss.item()
                log_str = f"loss: {total_loss:.3f}"

                for k in log_dict.keys():
                    log_str += " {}: {:.3f}".format(k, log_dict[k])

                pbar.set_description(log_str)

                self.opt.step()
                self.opt.zero_grad()

                self.step += 1
                if accelerator.is_main_process:

                    if self.step % self.i_image == 0:
                        self.on_evaluate_step()

                    if self.step != 0 and (self.step % self.i_save == 0):
                        milestone = self.step // self.i_save
                        self.save(milestone)

                pbar.update(1)

        if self.with_tracking:
            accelerator.end_training()