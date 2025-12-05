"""
Train with CR-CTC loss and sweep lambda_cr.

Based on scripts/train_model.py but targets the Transformer + CR-CTC setup.
"""

from copy import deepcopy
import os

from neural_decoder.neural_decoder_trainer import trainModel


def make_args(model_name, lambda_cr):
    args = {

        "outputDir": f"/home/tianlezheng/UCLA-ECE243-FALL2025/logs/speech_logs/{model_name}",
        "datasetPath": "/home/tianlezheng/UCLA-ECE243-FALL2025/data/ptDecoder_ctc",

        # Model / data basics
        "seed": 0,
        "batchSize": 64,
        "nBatch": 16000,
        "nClasses": 40,
        "nInputFeatures": 256,

        # Convolution / sequence geometry (used only for length math)
        "bidirectional": False,
        "strideLen": 4,   # CNN front-end uses stride 4; used in length computation
        "kernelLen": 0,   # if conv kernel > 0, adjust; used in length computation

        # Transformer architecture
        "use_transformer": True,
        "nhead": 6,
        "nUnits": 384,
        "nLayers": 8,
        "dim_feedforward": 1536,
        "use_rope": True,       # RoPE in transformer attention
        "use_layer_norm": True,
        "dropout": 0.3,

        "timeMasking": True,
        "timeMaskLen": 20,
        "timeMaskNum": 2,
        "featureMasking": True,
        "featureMaskLen": 20,
        "featureMaskNum": 2,
        "whiteNoiseSD": 0.8,
        "constantOffsetSD": 0.2,

        "gaussianSmoothWidth": 2.0,

        # time-based warps
        "augRampupFrac": 0.3,# ramp augment strength over first 30% of training
        "timeStretching": True,
        "timeStretchMinFactor": 0.9,
        "timeStretchMaxFactor": 1.1,
        "timeJittering": True,
        "timeJitterMaxShift": 5,

        # adversarial FGSM augmentation
        "useFGSM": True,
        "advEps": 0.02,# epsilon relative to feature scale
        "advProb": 0.3,

        # Optimizer & LR schedule
        "lrStart": 0.0008,
        "lrEnd": 0.00008,
        "l2_decay": 0.01,
        "scheduler_type": "cosine_warmup",  # "cosine_warmup", "cosine", or "linear"
        "warmup_steps": 500,

        # Gradient clipping
        "max_grad_norm": 5.0,
        "early_stopping_patience": 50,

        # CR-CTC loss vs label smoothing loss
        "use_cr_ctc": True,
        "lambda_cr": lambda_cr,
        "labelSmoothing": 0.0,  # 0 -> only CR-CTC; >0 -> label-smoothed CTC base
    }
    return args


def main():
    lambda_grid = [0.05, 0.1, 0.2]
    for lam in lambda_grid:
        model_name = f"speechTransformerCRCTC_ROPE_layernorm_gaussianSmooth_tier1_fgsm_lam{lam}"
        args = make_args(model_name, lam)
        os.makedirs(args["outputDir"], exist_ok=True)
        print(f"=== Starting run: {model_name} (lambda_cr={lam}) ===")
        trainModel(deepcopy(args))


if __name__ == "__main__":
    main()
