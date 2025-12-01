"""
Train GRU/Transformer speech decoder with CR-CTC (CTC + symmetric KL consistency).

Runs a simple sweep over lambda_cr values; each value writes to its own output directory.
Update DATASET_PATH/OUTPUT_ROOT to your environment before running.
"""
import argparse
from pathlib import Path

from neural_decoder.neural_decoder_trainer import trainModel


def main():
    parser = argparse.ArgumentParser(description="CR-CTC training sweep")
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="*",
        default=None,
        help="Optional list of lambda_cr values to sweep; if omitted uses base_args['lambda_cr'] for a single run.",
    )
    args_in = parser.parse_args()

    # Base paths (edit to match your setup)
    DATASET_PATH = Path("/home/alex/Downloads/UCLA-ECE243-FALL2025/data/ptDecoder_ctc.pkl")
    OUTPUT_ROOT = Path("/home/alex/Downloads/UCLA-ECE243-FALL2025/logs/speech_logs")

    base_args = {
        "outputDir": None,  # set per sweep
        "datasetPath": str(DATASET_PATH),
        "seqLen": 150,
        "maxTimeSeriesLen": 1200,
        "batchSize": 128,
        "lrStart": 0.05,
        "lrEnd": 0.02,
        "nUnits": 256,
        "nBatch": 10000,
        "nLayers": 5,
        "seed": 0,
        "nClasses": 40,
        "nInputFeatures": 256,
        "dropout": 0.2,
        "whiteNoiseSD": 0.8,
        "constantOffsetSD": 0.2,
        "gaussianSmoothWidth": 2.0,
        "strideLen": 4,
        "kernelLen": 32,
        "bidirectional": False,
        "l2_decay": 1e-5,
        # Optimizer/model toggles
        "use_transformer": False,
        "optimizer": "adamw",
        # Consistency regularization weight (can be edited here or via --lambdas)
        "lambda_cr": 0.2,
    }

    lambda_values = args_in.lambdas if args_in.lambdas is not None else [base_args["lambda_cr"]]
    base_run_name = "speechBaseline4_cr"

    for lam in lambda_values:
        run_name = f"{base_run_name}_{lam:.3f}"
        run_args = dict(base_args)
        run_args["lambda_cr"] = lam
        run_args["outputDir"] = str(OUTPUT_ROOT / run_name)
        print(f"Starting run lambda_cr={lam} -> {run_args['outputDir']}")
        trainModel(run_args)


if __name__ == "__main__":
    main()
