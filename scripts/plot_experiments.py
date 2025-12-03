import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

def plot_experiments(base_log_dir):
    # Find all stats.csv files in subdirectories
    stat_files = glob.glob(os.path.join(base_log_dir, "**", "stats.csv"), recursive=True)
    
    if not stat_files:
        print(f"No stats.csv files found in {base_log_dir}")
        return

    plt.figure(figsize=(15, 6))

    # Plot CER
    plt.subplot(1, 2, 1)
    for f in stat_files:
        try:
            df = pd.read_csv(f)
            # Extract experiment name from parent directory
            exp_name = os.path.basename(os.path.dirname(f))
            plt.plot(df['batch'], df['cer'], label=exp_name)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    plt.xlabel('Batch')
    plt.ylabel('Character Error Rate (CER)')
    plt.title('CER over Training')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.1)

    # Plot Loss
    plt.subplot(1, 2, 2)
    for f in stat_files:
        try:
            df = pd.read_csv(f)
            exp_name = os.path.basename(os.path.dirname(f))
            plt.plot(df['batch'], df['ctc_loss'], label=exp_name)
        except:
            pass
            
    plt.xlabel('Batch')
    plt.ylabel('CTC Loss')
    plt.title('Loss over Training')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('experiment_comparison.png')
    print("Plot saved to experiment_comparison.png")

if __name__ == "__main__":
    # Assuming logs are in logs/speech_logs/
    base_dir = '/home/chenggong/UCLA-ECE243-FALL2025/logs/speech_logs/'
    plot_experiments(base_dir)

