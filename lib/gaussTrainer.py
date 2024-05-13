import os
import numpy as np
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.profiler import profile, ProfilerActivity

from accelerate import Accelerator
from pathlib import Path

import lib as utils
import lib.loss_utils as loss_utils
from lib.data_utils import read_all, read_points3D_colmap
from lib.camera_utils import to_viewpoint_camera
from lib.point_utils import PointCloud
# from lib._GS_renderer import GaussRenderer
from lib.GS_renderer import GaussRenderer

import contextlib


def exists(x):
    return x is not None


class GaussTrainer:
    def __init__(self, model, config, **kwargs):
        super().__init__()

        self.USE_GPU_PYTORCH = True
        self.USE_PROFILE = False

        self.data = read_all(
            os.path.join(config.database_path, "images.txt"),
            os.path.join(config.database_path, "cameras.txt"),
            scale_factor=config.scale_factor,
        )
        self.img_scale = config.scale_factor

        points_df = read_points3D_colmap(os.path.join(config.database_path, "points3D.txt"))
        points_coor = points_df[["X", "Y", "Z"]].to_numpy()
        # print(points_coor[:10])
        channels = {
            "R": points_df["R"].to_numpy(),
            "G": points_df["G"].to_numpy(),
            "B": points_df["B"].to_numpy(),
        }
        print("min max mean")
        print(channels["R"].min(), channels["R"].max(), channels["R"].mean())
        print(channels["G"].min(), channels["G"].max(), channels["G"].mean())
        print(channels["B"].min(), channels["B"].max(), channels["B"].mean())
        # print(channels["R"][:10])
        # print(channels["G"][:10])
        # print(channels["B"][:10])
        points_cloud = PointCloud(points_coor, channels)
        raw_points = points_cloud.random_sample(2**9 * 7)
        model.create_from_pcd(raw_points)

        ##### TODO: Implement gauss renderer
        self.gaussRender = GaussRenderer(
            **kwargs.get("render_kwargs", {}), width=self.data[0]["scaledW"], height=self.data[0]["scaledH"], image_df=self.data[0]["image_df"]
        )

        # create_from_pcd from raw_points (sample from points_cloud)
        ###

        self.lambda_dssim = config.lambda_dssim
        self.lambda_depth = config.lambda_depth

        #####
        self.accelerator = Accelerator(
            split_batches=config.split_batches,
            mixed_precision="fp16" if config.fp16 else "no",
            project_dir=config.results_folder if config.with_tracking else None,
            log_with="all",
        )

        self.accelerator.native_amp = config.amp
        #####

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

        self.opt = Adam(self.model.parameters(), lr=config.lr, betas=config.adam_betas)

        #####
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
        #####

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            # "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, os.path.join(self.results_folder, f"model-{milestone}.pt"))

    def load(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        accelerator = self.accelerator
        device = accelerator.device
        # device = "cuda"

        data = torch.load(str(self.results_folder / f"model-{milestone}.pt"), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        # model = self.model
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
        if self.USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(data)

        if self.USE_PROFILE:
            prof = profile(activities=[ProfilerActivity.CUDA], with_stack=True)
        else:
            prof = contextlib.nullcontext()

        with prof:
            out = self.gaussRender(pc=self.model, camera=camera, data=data)

        if self.USE_PROFILE:
            print(prof.key_averages(group_by_stack_n=True).table(sort_by="self_cuda_time_total", row_limit=20))

        l1_loss = loss_utils.l1_loss(out["render"], data["rgb"])
        out_np = out["render"].detach().cpu().numpy()
        print("out min max mean")
        print(out_np[..., 0].min(), out_np[..., 0].max(), out_np[..., 0].mean())
        print(out_np[..., 1].min(), out_np[..., 1].max(), out_np[..., 1].mean())
        print(out_np[..., 2].min(), out_np[..., 2].max(), out_np[..., 2].mean())
        data_np = data["rgb"].detach().cpu().numpy()
        print("data min max mean")
        print(data_np[..., 0].min(), data_np[..., 0].max(), data_np[..., 0].mean())
        print(data_np[..., 1].min(), data_np[..., 1].max(), data_np[..., 1].mean())
        print(data_np[..., 2].min(), data_np[..., 2].max(), data_np[..., 2].mean())
        # print("complete L1")
        # depth_loss = loss_utils.l1_loss(out["depth"][..., 0][mask], depth[mask]) # we don't have depth
        ssim_loss = 1.0 - loss_utils.ssim(out["render"], data["rgb"])

        total_loss = (1 - self.lambda_dssim) * l1_loss * 0.3 * + self.lambda_dssim * ssim_loss
        total_loss += max(0, out["render"].mean() - 1) ** 2 * 0.1
        total_loss += max(0, out["color"].mean() - 1) ** 2 *0.3
        total_loss += (out["color"][out["color"] > 1] - 1).mean() * 0.1
        psnr = utils.img2psnr(out["render"], data["rgb"])
        # print("complete PSNR")
        # log_dict = {"total": total_loss, "l1": l1_loss, "ssim": ssim_loss, "depth": depth_loss, "psnr": psnr}
        log_dict = {"total": total_loss, "l1": l1_loss, "ssim": ssim_loss, "psnr": psnr}

        return total_loss, log_dict

    def on_evaluate_step(self, **kwargs):
        import matplotlib.pyplot as plt

        ind = np.random.choice(len(self.data))
        data = self.data[ind]
        if self.USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(data)

        rgb = data["rgb"].detach().cpu().numpy()
        # print("real", rgb[..., 0].min(), rgb[..., 0].max(), rgb[..., 1].min(), rgb[..., 1].max(), rgb[..., 2].min(), rgb[..., 2].max())
        # exit(0)
        out = self.gaussRender(pc=self.model, camera=camera)
        # from torchinfo import summary
        # summary gaussRender(pc=self.model, camera=camera)
        # print(summary(self.gaussRender, [self.model, camera]))
        rgb_pd = out["render"].detach().cpu().numpy()
        # print("pred", rgb_pd[..., 0].min(), rgb_pd[..., 0].max(), rgb_pd[..., 1].min(), rgb_pd[..., 1].max(), rgb_pd[..., 2].min(), rgb_pd[..., 2].max())
        # print(rgb.shape, rgb_pd.shape)
        # exit(0)
        image = np.concatenate([rgb, rgb_pd], axis=1)
        # image = np.concatenate([image, depth], axis=0)
        utils.imwrite(os.path.join(self.results_folder, f"image-{self.step}.png"), image)
        # print('saved')
        # exit(0)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        # device = "cuda"

        # with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            
            torch.autograd.set_detect_anomaly(True)
            while self.step < self.train_num_steps:

                # print("Step: ", self.step)

                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):

                    with self.accelerator.autocast():
                        loss, log_dict = self.on_train_step()
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss
                    # print("start backward")
                    self.accelerator.backward(loss)
                    # print("end backward")

                    # loss, log_dict = self.on_train_step()
                    # loss = loss / self.gradient_accumulate_every
                    # total_loss += loss

                    # loss.backward()

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
                # if accelerator.is_main_process:
                if True:

                    if self.step % self.i_image == 0:
                        self.on_evaluate_step()

                    if self.step != 0 and (self.step % self.i_save == 0):
                        milestone = self.step // self.i_save
                        self.save(milestone)

                pbar.update(1)

        if self.with_tracking:
            # accelerator.end_training()
            pass
