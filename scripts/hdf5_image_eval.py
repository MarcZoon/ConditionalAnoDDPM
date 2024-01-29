import argparse
import copy
import os

import torch
from ano3ddpm import dist_util, plot_util
from ano3ddpm.hdf5_image_dataset import load_data
from ano3ddpm.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def main():
    args = create_argsparser().parse_args()

    for dir in ["plots"]:
        os.makedirs(f"{args.output_dir}/{dir}", exist_ok=True)

    dist_util.setup_dist()

    # load model
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # load data
    data = load_data(
        file_path=args.file_path,
        batch_size=args.batch_size,
        split="validate",
        label=args.label,
    )

    img, _ = next(data)

    checkpoint_file = sorted([c for c in os.listdir(args.model_dir) if "model" in c])[
        -1
    ]
    print(checkpoint_file)
    model.load_state_dict(torch.load(f"{args.model_dir}/{checkpoint_file}"))

    model.to(dist_util.dev())

    # plot diffusion process
    samples = []
    img = img.to(dist_util.dev())
    with torch.no_grad():
        for sample in diffusion.sample_from_img_progressive(
            model=model,
            x_start=img,
            sample_distance=args.sample_distance,
        ):
            samples.append(sample["sample"].to("cpu"))

    img = img.to("cpu")
    plotlist = [
        img,
        samples[0],
        samples[-1],
        (img - samples[-1]) ** 2,
        ((img - samples[-1]) ** 2) >= 0.5,
    ]
    plot_util.save_as_plot(plotlist, f"{args.output_dir}/plots/sample.png", stepsize=1)


def create_argsparser() -> argparse.ArgumentParser:
    defaults = dict(
        file_path="",
        model_dir="",
        output_dir="",
        batch_size=1,
        microbatch=-1,
        label="benign",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
