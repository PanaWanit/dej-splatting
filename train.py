import torch
import numpy as np
import argparse
import os
from lib.data_utils import read_all
from lib.gaussTrainer import GaussTrainer

if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument("--database_path", type=str, help="Images and colmap settings folder", required=True)
    """
    % database_path/
    %      |- IMG00001.png
    %      |- IMG00002.png
    %      |- .....
    %      |- image.txt
    %      |- camera.txt
    %      |- points3D.txt
    """
    parser.add_argument("--results_folder", type=str, default="./result", help="Output folder")
    parser.add_argument("--lr", type=float, default=1e-2, help="Train learning rate")
    parser.add_argument("--batch_size", type=int, default=4096, help="Train batch size")
    parser.add_argument("--epochs", type=int, default=25000, help="Training epochs")

    parser.add_argument("--gradient_accumulate_every", type=int, default=1)
    parser.add_argument("--adam_betas", type=tuple(float, float), default=(0.9, 0.99), help="Training epochs")

    parser.add_argument("--i_print", type=int, default=100)
    parser.add_argument("--i_image", type=int, default=1000)
    parser.add_argument("--i_save", type=int, default=50000)

    parser.add_argument("--split_batches", type=bool, default=False)
    parser.add_argument("--amp", type=bool, default=False)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--with_tracking", type=bool, default=False)

    # Gauss part
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--lambda_depth", type=float, default=0.0)

    args = parser.parse_args()
    # TODO: Add Gauss Model

    # gauss_model = GaussModel()
    render_kwargs = {
        "white_bkgd": True,
    }
    trainer = GaussTrainer(gauss_model, args=args, render_kwargs=render_kwargs)

    trainer.on_evaluate_step()
    trainer.train()
