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
        # not referenced anywhere in the trainer or model

        "bidirectional": False,
        # accepted by TransformerDecoder but never used inside it.

        "strideLen": 4, # stored, but CNN front-end is hardcoded to stride 4; only used later for length math.
        "kernelLen": 0, # not used in the CNN; only affects the length formula in the trainer.
        ##################################################### used and could be improved by grid search######################################
        #basic info about dataset and training
        "seed": 0,
        "batchSize": 64,
        "nBatch": 16000,
        "nClasses": 40,
        "nInputFeatures": 256,

        # Transformer architecture
        "use_transformer": True,
        "nhead": 6,
        "nUnits": 384,
        "nLayers": 8,
        "dim_feedforward": 1536,
        "timeMasking": True,
        "featureMasking": True,
        "use_rope": True,  # Enable RoPE in transformer attention
        "use_layer_norm": True,
        "dropout": 0.3,

        # augmentation
        "whiteNoiseSD": 0.8,
        "constantOffsetSD": 0.2,
        "gaussianSmoothWidth": 2.0,

        # optimizer and learning rate LR Schedule and decay
        "lrStart": 0.0008,
        "lrEnd": 0.00008,
        "l2_decay": 0.01,
        "warmupSteps": 500,
        "gradClip": 5.0, #Stability

        #Training Control
        "earlyStoppingPatience": 50,

        # CR-CTC loss vs label smoothing loss
        "use_cr_ctc": True,
        "lambda_cr": lambda_cr,
        "labelSmoothing": 0.0, # set 0 for not used, use cr_ctc

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
