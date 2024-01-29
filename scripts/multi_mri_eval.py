import argparse
import copy
import csv
import os

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

    for dir in ["plots", "video"]:
        os.makedirs(f"{args.output_dir}/{dir}", exist_ok=True)

    dist_util.setup_dist()

    # load model
    model_args = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**model_args)

    # load data
    transform = transforms.Compose([transforms.Normalize(0.5, 0.5)])

    data = load_data(
        file_path=args.file_path,
        batch_size=args.batch_size,
        split=args.split,
        organs=args.organs,
        labels=args.labels,
        class_cond=args.class_cond,
        transform=transform,
    )

    img, additional_data = next(data)

    final_file = sorted([c for c in os.listdir(args.model_dir) if "model" in c])[-1]
    ema_file = sorted([c for c in os.listdir(args.model_dir) if "ema" in c])[-1]

    healthy_data = load_data(
        file_path=args.file_path,
        batch_size=args.batch_size,
        split=args.split,
        organs=args.organs,
        labels="healthy",
        class_cond=args.class_cond,
        transform=transform,
    )

    logfile = f"{args.output_dir}/eval.csv"
    with open(logfile, "w") as f:
        f.write("model,sample_distance,healthy_mse,healthy_dice")
    for model_type, model_file in (("model", final_file), ("ema", ema_file)):
        print(f"{model_type}: sampling entire markov chain")
        model.load_state_dict(torch.load(f"{args.model_dir}/{model_file}"))
        model.to(dist_util.dev())

        # GENERATE HEALTHY SAMPLES FROM PURE NOISE
        # samples = diffusion.p_sample_loop(
        #     model,
        #     (args.batch_size, args.in_channels, args.image_size, args.image_size),
        #     model_kwargs={"y": torch.zeros((8,), dtype=torch.int)},
        # ).to("cpu")
        # plot_util.plot_samples(
        #     samples, f"{args.output_dir}/plots/{model_type}_samples.png"
        # )

        # EVALUATE DETECTION OF ANOMALIES
        for sample_distance in [250]:
            print(
                f"{model_type}: sampling from lambda={sample_distance} on unhealthy samples"
            )
            # plot diffusion process
            samples = []
            img = img.to(dist_util.dev())
            with torch.no_grad():
                for sample in diffusion.ddim_sample_from_img_progressive(
                    model=model,
                    x_start=img,
                    sample_distance=sample_distance,
                    model_kwargs={"y": torch.zeros((8,), dtype=torch.int)},
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
            if "seg" in additional_data.keys():
                plotlist.append(additional_data["seg"])

            se = (img - samples[-1]) ** 2
            sum_se = torch.sum(se, dim=1, keepdim=True)
            sum_pred = sum_se >= 0.5
            mean_se = torch.mean(se, dim=1, keepdim=True)
            mean_pred = mean_se >= 0.5
            plot_util.save_evaluation_plot(
                diffusion_data=[
                    img,
                    samples[0],
                    samples[-1],
                ],
                sum_se=sum_se,
                sum_pred=sum_pred,
                mean_se=mean_se,
                mean_pred=mean_pred,
                segmentation=additional_data["seg"],
                path=f"{args.output_dir}/plots/{model_type}_{sample_distance}.png",
                labels=[
                    "x_0",
                    f"x_{sample_distance}",
                    "x_0_pred",
                    "SUM:E_sq",
                    "SUM:E_sq >= 0.5",
                    "MEAN:E_sq",
                    "MEAN:E_sq >= 0.5",
                    "GT",
                ],
                ids=additional_data["scan_id"],
            )

            # eval healthy
            count = 0
            batches = []
            results = []
            while count < 64:
                count += args.batch_size
                batch, _ = next(healthy_data)
                batches.append(batch)

                with torch.no_grad():
                    result = diffusion.sample_from_img(
                        model=model,
                        x_start=batch,
                        sample_distance=sample_distance,
                        model_kwargs={
                            "y": torch.zeros((args.batch_size), dtype=np.int64)
                        },
                    )
                    results.append(result)

            truth = torch.cat(batches)
            results = torch.cat(results)
            truth = truth.to("cpu")
            results = results.to("cpu")
            mse = MSE(truth, results)

            dice = DiceCoefficient(
                truth=truth, prediction=results, truth_mask=torch.zeros_like(truth)
            )

            with open(logfile, "a") as f:
                f.write(
                    f"\n{model_type},{str(sample_distance)},{str(mse.item())},{str(dice.item())}"
                )

            # eval prediction to GT
            count = 0
            batches = []
            truth = []
            results = []
            while count < 64:
                count += args.batch_size
                batch, cond = next(healthy_data)
                batches.append(batch)
                truth.append(cond["y"])

                with torch.no_grad():
                    result = diffusion.sample_from_img(
                        model=model,
                        x_start=batch,
                        sample_distance=sample_distance,
                        model_kwargs={"y": torch.zeros(())},
                    )
                    results.append(result)

            x0 = torch.cat(batches)
            truth = torch.cat(truth) > 0
            results = torch.cat(results)
            pred = ((x0 - results) ** 2) >= 0.5

            mse = MSE(truth, pred)
            dice = None


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
