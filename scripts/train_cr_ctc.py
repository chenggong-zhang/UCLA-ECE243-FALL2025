"""
Train with CR-CTC loss and sweep lambda_cr.

Based on scripts/train_model.py but targets the Transformer + CR-CTC setup.
"""

from copy import deepcopy
import os

from neural_decoder.neural_decoder_trainer import trainModel


def make_args(model_name, lambda_cr):
    args = {
        "outputDir": f"/home/chenggong/UCLA-ECE243-FALL2025/logs/speech_logs/{model_name}",
        "datasetPath": "/home/chenggong/UCLA-ECE243-FALL2025/data/ptDecoder_ctc.pkl",

        # "seqLen": 150,
        # "maxTimeSeriesLen": 1200,

        "batchSize": 64,
        "lrStart": 0.0008,
        "lrEnd": 0.00008,
        "nUnits": 384,
        "nBatch": 16000,
        "nLayers": 8,
        "seed": 0,
        "nClasses": 40,
        "nInputFeatures": 256,
        "dropout": 0.3,
        "whiteNoiseSD": 0.8,
        "constantOffsetSD": 0.2,
        "gaussianSmoothWidth": 2.0,

        "strideLen": 4,
        "kernelLen": 0,

        "bidirectional": False,
        "l2_decay": 0.01,

        # Transformer specifics
        "use_transformer": True,
        "nhead": 6,
        "dim_feedforward": 1536,
        "timeMasking": True,
        "featureMasking": True,

        # Enhanced training controls
        "warmupSteps": 500,
        "labelSmoothing": 0.0,
        "earlyStoppingPatience": 50,
        "use_layer_norm": True,
        "gradClip": 5.0,
        # CR-CTC
        "use_cr_ctc": True,
        "lambda_cr": lambda_cr,
        "use_rope": True  # Enable RoPE in transformer attention
    }
    return args


def main():
    lambda_grid = [0.05, 0.1, 0.2]
    for lam in lambda_grid:
        model_name = f"speechTransformerCRCTC_ROPE_layernorm_gaussianSmooth_lam{lam}"
        args = make_args(model_name, lam)
        os.makedirs(args["outputDir"], exist_ok=True)
        print(f"=== Starting run: {model_name} (lambda_cr={lam}) ===")
        trainModel(deepcopy(args))


if __name__ == "__main__":
    main()
