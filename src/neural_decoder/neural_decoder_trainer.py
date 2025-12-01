import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import GRUDecoder
from .model_transformer import TransformerDecoder

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


def WarmupLinearLR(optimizer, warmup_steps, total_steps, lr_start, lr_end):
    """
    Warmup + linear decay scheduler implemented via LambdaLR.
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

    smoothing = args.get("labelSmoothing", 0.1)
    if smoothing > 0:
        loss_ctc = LabelSmoothingCTCLoss(
            blank=0, smoothing=smoothing, reduction="mean", zero_infinity=True
        )
    else:
        loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=args["lrStart"],
    #     betas=(0.9, 0.999),
    #     eps=0.1,
    #     weight_decay=args["l2_decay"],
    # )
    # 11/19/2025: Using AdamW instead of Adam
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.98), # Better for Transformers
        eps=1e-9, 
        weight_decay=args["l2_decay"],
    )


    warmup_steps = args.get("warmupSteps", 500)

    scheduler = WarmupLinearLR(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=args["nBatch"],
        lr_start=args["lrStart"],
        lr_end=args["lrEnd"],
    )
    grad_clip = args.get("gradClip", 5.0)

    # --train--
    testLoss = []
    testCER = []
    best_cer = float("inf")
    patience = args.get("earlyStoppingPatience", 50)
    patience_counter = 0
    startTime = time.time()
    for batch in range(args["nBatch"]):
        model.train()

        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )
        
        # Time Masking (SpecAugment)
        if args.get("timeMasking", False):
            # Mask roughly 5% of time steps in 10-step blocks
            B, T, C = X.shape
            mask_len = 20 # Mask 20 time steps (~400ms)
            # Apply to each item in batch
            for b in range(B):
                # Apply 2 masks per sequence on average
                for _ in range(2): 
                    if T > mask_len:
                        start = torch.randint(0, T - mask_len, (1,)).item()
                        X[b, start:start+mask_len, :] = 0.0
        
        # Feature Masking (SpecAugment)
        if args.get("featureMasking", False):
            B, T, C = X.shape
            mask_channels = 20 # Mask 20 channels
            for b in range(B):
                # Apply 2 masks per sequence
                for _ in range(2):
                    if C > mask_channels:
                        start = torch.randint(0, C - mask_channels, (1,)).item()
                        X[b, :, start:start+mask_channels] = 0.0

        # Compute prediction error
        pred = model.forward(X, dayIdx)

        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        if grad_clip and grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        # print(endTime - startTime)

        # Eval
        if batch % 100 == 0:
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

                    pred = model.forward(X, testDayIdx)
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                        y_len,
                    )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
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

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

            if cer < best_cer:
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
                best_cer = cer
                patience_counter = 0
            else:
                patience_counter += 1
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)
            
            # Save to CSV for easier visualization
            csv_path = args["outputDir"] + "/stats.csv"
            # Append if file exists, else write header
            mode = 'a' if os.path.exists(csv_path) else 'w'
            with open(csv_path, mode) as f:
                if mode == 'w':
                    f.write("batch,ctc_loss,cer,time_per_batch\n")
                f.write(f"{batch},{avgDayLoss},{cer},{(endTime - startTime)/100}\n")

            if patience_counter >= patience:
                print(f"Early stopping triggered at batch {batch} (patience {patience})")
                break


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
