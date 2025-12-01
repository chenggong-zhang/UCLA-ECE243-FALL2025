import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import GRUDecoder
from .model_transformer import TransformerDecoder

from .dataset import SpeechDataset


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
        ).to(device)

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    criterion_kl = torch.nn.KLDivLoss(reduction="none", log_target=True)
    lambda_cr = args.get("lambda_cr", 0.1)
    
    optimizer_name = str(args.get("optimizer", "adamw")).lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args["lrStart"],
            betas=(0.9, 0.999),
            eps=0.1,
            weight_decay=args["l2_decay"],
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args["lrStart"],
            betas=(0.9, 0.98),  # Better defaults for Transformers
            eps=1e-9,
            weight_decay=args["l2_decay"],
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args["lrEnd"] / args["lrStart"],
        total_iters=args["nBatch"],
    )

    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    train_iter = iter(trainLoader)
    for batch in range(args["nBatch"]):
        model.train()

        # Persistent iterator so we cycle through the loader instead of reusing the first batch
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

        # Two forward passes (weak/strong views)
        pred_1 = model.forward(X, dayIdx)  # [B, T_out, C]
        pred_2 = model.forward(X, dayIdx)  # [B, T_out, C]

        # CTC on each view (permute to [T, B, C] as required by CTCLoss)
        adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)
        log_probs_1_ctc = pred_1.log_softmax(2).permute(1, 0, 2)
        log_probs_2_ctc = pred_2.log_softmax(2).permute(1, 0, 2)
        loss_ctc_1 = loss_ctc(log_probs_1_ctc, y, adjustedLens, y_len)
        loss_ctc_2 = loss_ctc(log_probs_2_ctc, y, adjustedLens, y_len)
        loss_ctc_mean = 0.5 * (loss_ctc_1 + loss_ctc_2)

        # Symmetric KL consistency regularization in log-prob space
        log_probs_1 = pred_1.log_softmax(dim=2)  # [B, T_out, C]
        log_probs_2 = pred_2.log_softmax(dim=2)  # [B, T_out, C]
        kl_1 = criterion_kl(log_probs_2, log_probs_1.detach()).sum(dim=2)  # KL(p1||p2)
        kl_2 = criterion_kl(log_probs_1, log_probs_2.detach()).sum(dim=2)  # KL(p2||p1)
        kl_sym = 0.5 * (kl_1 + kl_2)  # [B, T_out]

        # Mask padded time steps using adjusted lengths
        B, T_out, _ = log_probs_1.shape
        time_ids = torch.arange(T_out, device=log_probs_1.device).unsqueeze(0)  # [1, T_out]
        mask = (time_ids < adjustedLens.unsqueeze(1)).float()  # [B, T_out]
        loss_cr = (kl_sym * mask).sum() / mask.sum().clamp_min(1.0)

        # Final loss combines CTC and consistency regularization
        if lambda_cr > 0:
            loss = loss_ctc_mean + lambda_cr * loss_cr
        else:
            loss = loss_ctc_mean

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
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

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
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
        ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()
