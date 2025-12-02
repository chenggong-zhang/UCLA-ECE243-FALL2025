import os
import pickle
import time
import math

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import GRUDecoder
from .model_transformer import TransformerDecoder

from .augmentations import GaussianSmoothing
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

    scheduler_type = args.get("scheduler_type", "cosine_warmup")  # "linear", "cosine_warmup", "cosine"
    warmup_steps = args.get("warmup_steps", 0)
    
    if scheduler_type == "cosine_warmup" and warmup_steps > 0:
        from torch.optim.lr_scheduler import LambdaLR
        import math
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                progress = float(current_step - warmup_steps) / float(max(1, args["nBatch"] - warmup_steps))
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

                min_lr_ratio = args["lrEnd"] / args["lrStart"]
                return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
        
        scheduler = LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == "cosine":
        # Cosine annealing without warmu
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args["nBatch"],
            eta_min=args["lrEnd"]
        )
    else:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=args["lrEnd"] / args["lrStart"],
            total_iters=args["nBatch"],
        )

    # Optional Gaussian smoothing pre-processing
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

    # Total training steps = epochs × batches per epoch
    total_steps = args["nBatch"]
    ramp_frac = args.get("augRampupFrac", 0.3)
    ramp_steps = max(1, int(total_steps * ramp_frac))

    def get_aug_scale(step_idx):
        if step_idx >= ramp_steps:
            return 1.0
        return float(step_idx) / float(ramp_steps)

    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    
    early_stopping_patience = args.get("early_stopping_patience", None)
    best_cer = None
    patience_counter = 0
    best_batch = 0
    
    if early_stopping_patience is not None:
        print(f"\nEarly stopping enabled with patience={early_stopping_patience} evaluations ({early_stopping_patience * 100} batches)")
        print(f"Training will stop if CER doesn't improve for {early_stopping_patience} consecutive evaluations.\n")
    
    global_step = 0
    train_iter = iter(trainLoader)

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
            X = X.permute(0, 2, 1)    # [B, C, T]
            X = gauss_smoother(X)
            X = X.permute(0, 2, 1)    # [B, T, C]

        # Augmentation schedule
        aug_scale = get_aug_scale(step)

        # Noise
        if args["whiteNoiseSD"] > 0 and aug_scale > 0:
            X += torch.randn(X.shape, device=device) * (args["whiteNoiseSD"] * aug_scale)

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
                        X[b, start:start + mask_len, :] = 0.0

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
                        X[b, :, start:start + mask_channels] = 0.0

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
        
        max_grad_norm = args.get("max_grad_norm", 1.0)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf loss detected at batch {step}! Loss: {loss}")
            print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
            continue
        
        optimizer.step()
        scheduler.step()
        global_step += 1

        # print(endTime - startTime)

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

                    # Gaussian smoothing at evaluation time
                    if gauss_smoother is not None:
                        X = X.permute(0, 2, 1)
                        X = gauss_smoother(X)
                        X = X.permute(0, 2, 1)

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
                current_lr = optimizer.param_groups[0]['lr']

                endTime = time.time()
                print(
                    f"batch {step}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, lr: {current_lr:.6f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

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
                        print(f"  -> No improvement for {patience_counter}/{early_stopping_patience} evaluations (patience: {patience_counter * 100} batches)")
            
            testLoss.append(avgDayLoss)
            testCER.append(cer)
            
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered!")
                print(f"  Best CER: {best_cer:.6f} at batch {best_batch}")
                print(f"  Current CER: {cer:.6f}")
                print(f"  No improvement for {patience_counter} evaluations ({patience_counter * 100} batches)")
                print(f"  Training stopped at batch {step} / {args['nBatch']}")
                break

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
                    f.write("batch,ctc_loss,cer,lr,time_per_batch\n")
                f.write(f"{step},{avgDayLoss},{cer},{current_lr},{(endTime - startTime)/100}\n")


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