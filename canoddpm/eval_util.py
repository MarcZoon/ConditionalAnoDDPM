import torch


def MSE(truth: torch.Tensor, prediction: torch.Tensor):
    mse = (truth - prediction) ** 2
    mse = torch.mean(mse)
    return mse


def DiceCoefficient(
    truth: torch.Tensor,
    prediction: torch.Tensor,
    truth_mask: torch.Tensor,
    smooth: float = 0.00000000001,
):
    pred_limit = ((truth - prediction).square() >= 0.5).float()
    truth_mask = (truth_mask > 0).float()

    intersection = torch.sum(pred_limit * truth_mask, dim=[1, 2, 3])
    union = torch.sum(pred_limit, dim=[1, 2, 3]) + torch.sum(truth_mask, dim=[1, 2, 3])
    dice = torch.mean((2.0 * intersection + smooth) / (union + smooth), dim=0)
    return dice


def main():
    truth = torch.zeros((8, 1, 256, 256))
    prediction = torch.rand_like(truth)
    dice = DiceCoefficient(truth, prediction, truth)
    print(dice)


if __name__ == "__main__":
    main()
