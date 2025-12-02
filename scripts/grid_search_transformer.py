import os
from copy import deepcopy
from itertools import product

from neural_decoder.neural_decoder_trainer import trainModel


def make_model_name(args):
    """Create a unique model name from key hyperparameters."""
    parts = [
        "speechTransformer",
        f"ls{args['labelSmoothing']}",
        f"do{args['dropout']}",
        f"lr{args['lrStart']:.0e}-{args['lrEnd']:.0e}",
        f"wu{args['warmupSteps']}",
        f"hid{args['nUnits']}",
    ]
    return "_".join(parts)


def iter_configs(base_args, search_space):
    """Yield full args dicts for each point in the grid."""
    keys = list(search_space.keys())
    for values in product(*(search_space[k] for k in keys)):
        cfg = deepcopy(base_args)
        for k, v in zip(keys, values):
            if k == "lr_pair":
                cfg["lrStart"], cfg["lrEnd"] = v
            else:
                cfg[k] = v
        model_name = make_model_name(cfg)
        cfg["outputDir"] = os.path.join(base_args["log_root"], model_name)
        cfg.pop("log_root", None)
        yield cfg


def main():
    # Base (non-swept) arguments
    base_args = {
        "log_root": "/home/chenggong/UCLA-ECE243-FALL2025/logs/speech_logs",
        "datasetPath": "/home/chenggong/UCLA-ECE243-FALL2025/data/ptDecoder_ctc.pkl",
        "seqLen": 150,
        "maxTimeSeriesLen": 1200,
        "batchSize": 64,
        "lrStart": 0.0008,
        "lrEnd": 0.00008,
        "warmupSteps": 500,
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
        "labelSmoothing": 0.01,
        "earlyStoppingPatience": 50,
        "use_layer_norm": True,
        "gradClip": 5.0,
    }

    # Define the grid to sweep
    search_space = {
        "labelSmoothing": [0.01, 0.05, 0.1],
        "dropout": [0.2, 0.3],
        "warmupSteps": [250, 500],
        # Pair lrStart and lrEnd to keep ~10x decay
        "lr_pair": [(0.0006, 0.00006), (0.0008, 0.00008)],
    }

    # Run the grid
    for cfg in iter_configs(base_args, search_space):
        print(f"Starting run: {cfg['outputDir']}")
        trainModel(cfg)


if __name__ == "__main__":
    main()
