import argparse
import copy
import csv
import os
import time

import h5py
import numpy as np
import torch
from ano3ddpm import dist_util, plot_util
from ano3ddpm.combined_mri_dataset import load_data
from ano3ddpm.eval_util import MSE, DiceCoefficient
from ano3ddpm.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from torchvision import transforms


def main():
    args = create_argsparser().parse_args()
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)
    h5 = h5py.File(f"{args.output_dir}/preds.hdf5", "w")

    dist_util.setup_dist()

    # load model
    model_args = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**model_args)

    # load data
    transform = transforms.Compose([transforms.Normalize(0.5, 0.5)])

    logfile = f"{args.output_dir}/eval.csv"
    with open(logfile, "w") as f:
        f.write("model,sample_distance,healthy_mse,healthy_dice")

    model_file = sorted([c for c in os.listdir(args.model_dir) if "ema" in c])[-1]
    sample_distance = 250

    model.load_state_dict(torch.load(f"{args.model_dir}/{model_file}"))
    model.to(dist_util.dev())

    # DO INFERENCE ON HEALTHY SAMPLES
    # count = 0
    # start = time.time()
    # data = load_data(
    #     file_path=args.file_path,
    #     batch_size=args.batch_size,
    #     split=args.split,
    #     organs=args.organs,
    #     labels="healthy",
    #     class_cond=args.class_cond,
    #     transform=transform,
    # )

    # group_healthy = h5.create_group("healthy")
    # while count < 1024:
    #     count += args.batch_size
    #     img, additional_data = next(data)
    #     with torch.no_grad():
    #         output = diffusion.sample_from_img(
    #             model=model,
    #             x_start=img,
    #             sample_distance=sample_distance,
    #             model_kwargs={"y": torch.zeros((args.batch_size,), dtype=torch.int)}
    #             if args.class_cond
    #             else {},
    #         )
    #     img = img.to("cpu")
    #     output = output.to("cpu")

    #     for i in range(img.shape[0]):
    #         img_in = img[i, ...]
    #         scan_id = additional_data["scan_id"][i]
    #         seg = additional_data["seg"][i]
    #         img_out = output[i, ...]
    #         h5group = group_healthy.create_group(scan_id)
    #         h5group.create_dataset("img", data=img_in, compression="gzip")
    #         h5group.create_dataset("pred", data=img_out, compression="gzip")
    #         h5group.create_dataset("seg", data=seg, compression="gzip")

    #     print(
    #         f"Processed {count}/1024 healthy samples in {int((time.time() - start) / 60)}",
    #     )

    # DO INFERENCE ON UNHEALTHY SAMPLES
    count = 0

    data = load_data(
        file_path=args.file_path,
        batch_size=args.batch_size,
        split=args.split,
        organs=args.organs,
        labels="healthy",
        class_cond=args.class_cond,
        transform=transform,
        deterministic=True,
    )

    group_healthy = h5.create_group("healthy")
    while count < 1024:
        count += args.batch_size
        img, additional_data = next(data)
        with torch.no_grad():
            output = diffusion.sample_from_img(
                model=model,
                x_start=img,
                sample_distance=sample_distance,
                model_kwargs={"y": torch.zeros((args.batch_size,), dtype=torch.int)}
                if args.class_cond
                else {},
            )
        img = img.to("cpu")
        output = output.to("cpu")

        for i in range(img.shape[0]):
            img_in = img[i, ...]
            scan_id = additional_data["scan_id"][i]
            seg = additional_data["seg"][i]
            img_out = output[i, ...]
            h5group = group_healthy.create_group(scan_id)
            h5group.create_dataset("img", data=img_in, compression="gzip")
            h5group.create_dataset("pred", data=img_out, compression="gzip")
            h5group.create_dataset("seg", data=seg, compression="gzip")

        print(f"progress: {count}/1024")


def create_argsparser() -> argparse.ArgumentParser:
    defaults = dict(
        file_path="",
        model_dir="",
        output_dir="",
        batch_size=1,
        microbatch=-1,
        organs="all",
        labels="healthy",
        split="validate",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
