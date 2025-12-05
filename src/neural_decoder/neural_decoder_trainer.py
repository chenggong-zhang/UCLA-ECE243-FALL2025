import os
import pickle
import time
import math

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .model import GRUDecoder
from .model_transformer import TransformerDecoder

from .augmentations import GaussianSmoothing
from .dataset import SpeechDataset


class LabelSmoothingCTCLoss(torch.nn.Module):
    """
    CTC loss with optional label smoothing to reduce overconfidence.
    """

    def __init__(self, blank=0, smoothing=0.1, reduction="mean", zero_infinity=True):
        super().__init__()
        self.blank = blank
        self.smoothing = smoothing
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        self.ctc_loss = torch.nn.CTCLoss(
            blank=blank, reduction="none", zero_infinity=zero_infinity
        )

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # Standard CTC loss (per-example)
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

        if self.smoothing > 0:
            # Encourage higher entropy (less peaky distributions)
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            loss = (1 - self.smoothing) * loss - self.smoothing * entropy

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class CRCTCLoss(torch.nn.Module):
    """
    Consistency-regularized CTC loss (CR-CTC).
    Combines two-view CTC with symmetric KL between the views.
    """

    def __init__(self, base_ctc_loss, lambda_cr=0.1):
        super().__init__()
        self.base_ctc_loss = base_ctc_loss
        self.lambda_cr = lambda_cr
        self.kl = torch.nn.KLDivLoss(reduction="none", log_target=True)

    def forward(self, logits_1, logits_2, targets, input_lengths, target_lengths):
        # Log-probabilities
        log_probs_1 = logits_1.log_softmax(dim=2)
        log_probs_2 = logits_2.log_softmax(dim=2)

        # CTC loss on each view (base_ctc_loss expects [T, B, C])
        loss_ctc_1 = self.base_ctc_loss(
            log_probs_1.permute(1, 0, 2), targets, input_lengths, target_lengths
        )
        loss_ctc_2 = self.base_ctc_loss(
            log_probs_2.permute(1, 0, 2), targets, input_lengths, target_lengths
        )
        loss_ctc_mean = 0.5 * (loss_ctc_1 + loss_ctc_2)

        # Symmetric KL (detach teacher)
        kl_1 = self.kl(log_probs_2, log_probs_1.detach()).sum(dim=2)  # KL(p1 || p2)
        kl_2 = self.kl(log_probs_1, log_probs_2.detach()).sum(dim=2)  # KL(p2 || p1)
        kl_sym = 0.5 * (kl_1 + kl_2)  # [B, T]

        # Mask padded steps
        B, T_out, _ = log_probs_1.shape
        time_ids = torch.arange(T_out, device=log_probs_1.device).unsqueeze(0)  # [1, T]
        mask = (time_ids < input_lengths.unsqueeze(1)).float()  # [B, T]
        loss_cr = (kl_sym * mask).sum() / mask.sum().clamp_min(1.0)

        return loss_ctc_mean + self.lambda_cr * loss_cr


def WarmupLinearLR(optimizer, warmup_steps, total_steps, lr_start, lr_end):
    """
    Warmup + linear decay scheduler implemented via LambdaLR.
    (Defined but not used in this merged version; kept for future use.)
    """

    total_steps = max(1, total_steps)
    warmup_steps = min(max(0, warmup_steps), total_steps)
    target_factor = lr_end / lr_start if lr_start > 0 else 1.0

    def lr_lambda(step):
        # Warmup from 0 -> 1 over warmup_steps
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        # Linear decay from 1 -> target_factor for the remaining steps
        remaining_steps = max(1, total_steps - warmup_steps)
        decay_step = min(step - warmup_steps, remaining_steps)
        decay_progress = decay_step / remaining_steps
        factor = 1.0 - decay_progress * (1.0 - target_factor)
        return max(factor, target_factor)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData


def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )

    if args.get("use_transformer", False):
        model = TransformerDecoder(
            neural_dim=args["nInputFeatures"],
            n_classes=args["nClasses"],
            hidden_dim=args["nUnits"],
            layer_dim=args["nLayers"],
            nDays=len(loadedData["train"]),
            dropout=args["dropout"],
            device=device,
            strideLen=args["strideLen"],
            kernelLen=args["kernelLen"],
            gaussianSmoothWidth=args["gaussianSmoothWidth"],
            bidirectional=args["bidirectional"],
            nhead=args.get("nhead", 4),
            dim_feedforward=args.get("dim_feedforward", 1024),
            use_layer_norm=args.get("use_layer_norm", False),
            use_rope=args.get("use_rope", True),
        ).to(device)
    else:
        model = GRUDecoder(
            neural_dim=args["nInputFeatures"],
            n_classes=args["nClasses"],
            hidden_dim=args["nUnits"],
            layer_dim=args["nLayers"],
            nDays=len(loadedData["train"]),
            dropout=args["dropout"],
            device=device,
            strideLen=args["strideLen"],
            kernelLen=args["kernelLen"],
            gaussianSmoothWidth=args["gaussianSmoothWidth"],
            bidirectional=args["bidirectional"],
            use_layer_norm=args.get("use_layer_norm", False),
        ).to(device)

    # Loss: label-smoothed CTC + optional CR-CTC
    smoothing = args.get("labelSmoothing", 0.1)
    if smoothing > 0:
        base_ctc_loss = LabelSmoothingCTCLoss(
            blank=0, smoothing=smoothing, reduction="mean", zero_infinity=True
        )
    else:
        base_ctc_loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    use_cr_ctc = args.get("use_cr_ctc", False)
    lambda_cr = args.get("lambda_cr", 0.1)
    if use_cr_ctc:
        loss_fn = CRCTCLoss(base_ctc_loss, lambda_cr=lambda_cr)
    else:
        loss_fn = base_ctc_loss

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=args["l2_decay"],
    )

    # Scheduler (cosine / cosine_warmup / linear)
    scheduler_type = args.get(
        "scheduler_type", "cosine_warmup"
    )  # "linear", "cosine_warmup", "cosine"
    warmup_steps = args.get("warmup_steps", 0)

    if scheduler_type == "cosine_warmup" and warmup_steps > 0:
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                progress = float(current_step - warmup_steps) / float(
                    max(1, args["nBatch"] - warmup_steps)
                )
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

                min_lr_ratio = args["lrEnd"] / args["lrStart"]
                return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

        scheduler = LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args["nBatch"],
            eta_min=args["lrEnd"],
        )
    else:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=args["lrEnd"] / args["lrStart"],
            total_iters=args["nBatch"],
        )

    gauss_smoother = None
    gs_width = args.get("gaussianSmoothWidth", 0.0)
    if gs_width is not None and gs_width > 0.0:
        kernel_size = int(4 * gs_width + 1)
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1

        gauss_smoother = GaussianSmoothing(
            channels=args["nInputFeatures"],
            kernel_size=kernel_size,
            sigma=gs_width,
            dim=1,
        ).to(device)

    # augmentation schedule
    total_steps = args["nBatch"]
    ramp_frac = args.get("augRampupFrac", 0.3)
    ramp_steps = max(1, int(total_steps * ramp_frac))

    def get_aug_scale(step_idx: int) -> float:
        if step_idx >= ramp_steps:
            return 1.0
        return float(step_idx) / float(ramp_steps)

    # ---- Train loop ----
    testLoss = []
    testCER = []
    startTime = time.time()

    early_stopping_patience = args.get("early_stopping_patience", None)
    best_cer = None
    best_batch = 0
    patience_counter = 0

    if early_stopping_patience is not None:
        print(
            f"\nEarly stopping enabled with patience={early_stopping_patience} "
            f"evaluations ({early_stopping_patience * 100} batches)"
        )
        print(
            f"Training will stop if CER doesn't improve for "
            f"{early_stopping_patience} consecutive evaluations.\n"
        )

    max_grad_norm = args.get("max_grad_norm", 1.0)

    train_iter = iter(trainLoader)
    global_step = 0

    for step in range(args["nBatch"]):
        model.train()

        try:
            X, y, X_len, y_len, dayIdx = next(train_iter)
        except StopIteration:
            train_iter = iter(trainLoader)
            X, y, X_len, y_len, dayIdx = next(train_iter)

        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Gaussian smoothing
        if gauss_smoother is not None:
            X = X.permute(0, 2, 1)
            X = gauss_smoother(X)
            X = X.permute(0, 2, 1)

        # Augmentation schedule
        aug_scale = get_aug_scale(step)

        # Global time-stretch
        if args.get("timeStretching", False) and aug_scale > 0:
            B, T, C = X.shape
            base_min = args.get("timeStretchMinFactor", 0.9)
            base_max = args.get("timeStretchMaxFactor", 1.1)

            for b in range(B):
                Tb = int(X_len[b].item())
                if Tb <= 1:
                    continue

                min_s = 1.0 + (base_min - 1.0) * aug_scale
                max_s = 1.0 + (base_max - 1.0) * aug_scale

                s = torch.empty(1, device=X.device).uniform_(min_s, max_s).item()

                t = torch.arange(Tb, device=X.device, dtype=torch.float32)
                src_idx = t / s
                src_idx = torch.clamp(src_idx, 0, Tb - 1)

                idx0 = torch.floor(src_idx).long()
                idx1 = torch.clamp(idx0 + 1, max=Tb - 1)
                w = (src_idx - idx0.float()).unsqueeze(-1)

                v0 = X[b, idx0, :]
                v1 = X[b, idx1, :]
                warped = (1.0 - w) * v0 + w * v1

                X[b, :Tb, :] = warped

        # Temporal jitter
        if args.get("timeJittering", False) and aug_scale > 0:
            B, T, C = X.shape
            base_max_shift = int(args.get("timeJitterMaxShift", 5))

            max_shift = int(base_max_shift * aug_scale)
            if max_shift > 0:
                for b in range(B):
                    Tb = int(X_len[b].item())
                    if Tb <= 1:
                        continue

                    shift = torch.randint(
                        -max_shift, max_shift + 1, (1,), device=X.device
                    ).item()
                    if shift == 0:
                        continue

                    xb = X[b, :Tb, :]
                    if shift > 0:
                        new_xb = torch.zeros_like(xb)
                        new_xb[shift:Tb, :] = xb[0 : Tb - shift, :]
                    else:
                        k = -shift
                        if k >= Tb:
                            new_xb = torch.zeros_like(xb)
                        else:
                            new_xb = torch.zeros_like(xb)
                            new_xb[0 : Tb - k, :] = xb[k:Tb, :]

                    X[b, :Tb, :] = new_xb

        # Noise
        if args["whiteNoiseSD"] > 0 and aug_scale > 0:
            X += torch.randn(X.shape, device=device) * (
                args["whiteNoiseSD"] * aug_scale
            )

        if args["constantOffsetSD"] > 0 and aug_scale > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * (args["constantOffsetSD"] * aug_scale)
            )

        # Time masking
        if args.get("timeMasking", False) and aug_scale > 0:
            B, T, C = X.shape
            mask_len = int(args.get("timeMaskLen", 20))
            base_num = int(args.get("timeMaskNum", 2))
            num_masks = max(1, math.ceil(base_num * aug_scale))
            for b in range(B):
                for _ in range(num_masks):
                    if T > mask_len:
                        start = torch.randint(0, T - mask_len, (1,)).item()
                        X[b, start : start + mask_len, :] = 0.0

        # Feature masking
        if args.get("featureMasking", False) and aug_scale > 0:
            B, T, C = X.shape
            mask_channels = int(args.get("featureMaskLen", 20))
            base_num = int(args.get("featureMaskNum", 2))
            num_masks = max(1, math.ceil(base_num * aug_scale))
            for b in range(B):
                for _ in range(num_masks):
                    if C > mask_channels:
                        start = torch.randint(0, C - mask_channels, (1,)).item()
                        X[b, :, start : start + mask_channels] = 0.0

        # Adversarial input perturbation
        use_fgsm = args.get("useFGSM", False)
        adv_eps = float(args.get("advEps", 0.0))
        adv_prob = float(args.get("advProb", 0.0))

        if use_fgsm and adv_eps > 0 and adv_prob > 0 and aug_scale > 0:
            if torch.rand(1, device=device).item() < adv_prob:
                X_adv = X.detach().clone().requires_grad_(True)

                pred_tmp = model.forward(X_adv, dayIdx)
                loss_tmp = base_ctc_loss(
                    torch.permute(pred_tmp.log_softmax(2), [1, 0, 2]),
                    y,
                    ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                    y_len,
                )

                grad_X = torch.autograd.grad(
                    loss_tmp,
                    X_adv,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]

                g = grad_X.sign()
                eps_eff = adv_eps * aug_scale
                X = X + eps_eff * g
                X = X.detach()

        adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)

        if use_cr_ctc:
            pred_1 = model.forward(X, dayIdx)
            pred_2 = model.forward(X, dayIdx)
            loss = loss_fn(pred_1, pred_2, y, adjustedLens, y_len)
        else:
            pred = model.forward(X, dayIdx)
            loss = loss_fn(
                torch.permute(pred.log_softmax(2), [1, 0, 2]),
                y,
                adjustedLens,
                y_len,
            )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        if max_grad_norm and max_grad_norm > 0:
            clip_grad_norm_(model.parameters(), max_grad_norm)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf loss detected at step {step}! Loss: {loss}")
            print(
                f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            continue

        optimizer.step()
        scheduler.step()
        global_step += 1

        # Eval
        if step % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    if gauss_smoother is not None:
                        X = X.permute(0, 2, 1)
                        X = gauss_smoother(X)
                        X = X.permute(0, 2, 1)

                    pred = model.forward(X, testDayIdx)
                    loss_eval = base_ctc_loss(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len - model.kernelLen) / model.strideLen).to(
                            torch.int32
                        ),
                        y_len,
                    )
                    allLoss.append(loss_eval.cpu().detach().numpy())

                    adjustedLens_eval = (
                        (X_len - model.kernelLen) / model.strideLen
                    ).to(torch.int32)
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(
                                pred[iterIdx, 0 : adjustedLens_eval[iterIdx], :]
                            ),
                            dim=-1,
                        )
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length
                current_lr = optimizer.param_groups[0]["lr"]

                endTime = time.time()
                print(
                    f"batch {step}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, "
                    f"lr: {current_lr:.6f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

            # Early stopping & best model tracking
            if best_cer is None:
                best_cer = cer
                best_batch = step
                patience_counter = 0
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
                print(f"  -> Initial CER: {cer:.6f}, saving model")
            elif cer < best_cer:
                best_cer = cer
                best_batch = step
                patience_counter = 0
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
                print(f"  -> New best CER: {cer:.6f}, saving model")
            else:
                if early_stopping_patience is not None:
                    patience_counter += 1
                    if patience_counter % 10 == 0:
                        print(
                            f"  -> No improvement for {patience_counter}/"
                            f"{early_stopping_patience} evaluations "
                            f"(patience: {patience_counter * 100} batches)"
                        )

            testLoss.append(avgDayLoss)
            testCER.append(cer)

            if (
                early_stopping_patience is not None
                and patience_counter >= early_stopping_patience
            ):
                print("\nEarly stopping triggered!")
                print(f"  Best CER: {best_cer:.6f} at batch {best_batch}")
                print(f"  Current CER: {cer:.6f}")
                print(
                    f"  No improvement for {patience_counter} evaluations "
                    f"({patience_counter * 100} batches)"
                )
                print(
                    f"  Training stopped at batch {step} / {args['nBatch']}"
                )
                break

            # Save stats
            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)

            # Save CSV
            csv_path = args["outputDir"] + "/stats.csv"
            mode = "a" if os.path.exists(csv_path) else "w"
            with open(csv_path, mode) as f:
                if mode == "w":
                    f.write("batch,ctc_loss,cer,lr,time_per_batch\n")
                f.write(
                    f"{step},{avgDayLoss},{cer},{current_lr},"
                    f"{(endTime - startTime)/100}\n"
                )


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    if args.get("use_transformer", False):
        model = TransformerDecoder(
            neural_dim=args["nInputFeatures"],
            n_classes=args["nClasses"],
            hidden_dim=args["nUnits"],
            layer_dim=args["nLayers"],
            nDays=nInputLayers,
            dropout=args["dropout"],
            device=device,
            strideLen=args["strideLen"],
            kernelLen=args["kernelLen"],
            gaussianSmoothWidth=args["gaussianSmoothWidth"],
            bidirectional=args["bidirectional"],
            nhead=args.get("nhead", 4),
            dim_feedforward=args.get("dim_feedforward", 1024),
            use_layer_norm=args.get("use_layer_norm", False),
            use_rope=args.get("use_rope", True),
        ).to(device)
    else:
        model = GRUDecoder(
            neural_dim=args["nInputFeatures"],
            n_classes=args["nClasses"],
            hidden_dim=args["nUnits"],
            layer_dim=args["nLayers"],
            nDays=nInputLayers,
            dropout=args["dropout"],
            device=device,
            strideLen=args["strideLen"],
            kernelLen=args["kernelLen"],
            gaussianSmoothWidth=args["gaussianSmoothWidth"],
            bidirectional=args["bidirectional"],
            use_layer_norm=args.get("use_layer_norm", False),
        ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)


if __name__ == "__main__":
    main()
