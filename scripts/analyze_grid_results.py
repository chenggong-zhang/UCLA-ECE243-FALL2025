#!/usr/bin/env python3
import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_FILE = '/home/tianlezheng/UCLA-ECE243-FALL2025/grid_search_batchsize_results.json'
BASE_LOG_DIR = '/home/tianlezheng/UCLA-ECE243-FALL2025/logs/speech_logs'

def load_results():
    """Load grid search results."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return None

def extract_cer_curve(log_dir):
    """Extract CER curve from training stats."""
    stats_path = os.path.join(log_dir, 'trainingStats')
    if not os.path.exists(stats_path):
        return None
    
    try:
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        if 'testCER' in stats and len(stats['testCER']) > 0:
            return {
                'cer': stats['testCER'].tolist(),
                'batches': list(range(0, len(stats['testCER']) * 100, 100))
            }
    except Exception as e:
        print(f"Error reading {stats_path}: {e}")
    return None

def analyze_results():
    """Analyze and display grid search results."""
    results = load_results()
    
    if not results:
        print("No grid search results found. Run grid_search_batchsize.py first.")
        return
    
    print("=" * 80)
    print("GRID SEARCH RESULTS ANALYSIS")
    print("=" * 80)
    
    successful = [r for r in results['results'] if r.get('best_cer') is not None]
    
    if not successful:
        print("No successful runs found.")
        return
    
    successful.sort(key=lambda x: x['best_cer'])
    
    print(f"\n{'Batch Size':<15} {'Best CER':<15} {'Status':<15} {'Log Dir':<50}")
    print("-" * 95)
    
    for r in results['results']:
        cer_str = f"{r['best_cer']:.6f}" if r.get('best_cer') else "N/A"
        log_dir = r.get('log_dir', 'N/A')
        if len(log_dir) > 47:
            log_dir = log_dir[:44] + "..."
        print(f"{r['batch_size']:<15} {cer_str:<15} {r['status']:<15} {log_dir:<50}")
    
    print("\n" + "=" * 80)
    print("RANKING (Best to Worst)")
    print("=" * 80)
    
    for i, r in enumerate(successful, 1):
        improvement = ""
        if i > 1:
            prev_cer = successful[i-2]['best_cer']
            diff = r['best_cer'] - prev_cer
            improvement = f" ({diff:+.4f} vs previous)"
        print(f"{i}. Batch Size {r['batch_size']:<5} → CER: {r['best_cer']:.6f}{improvement}")
    
    best = successful[0]
    print("\n" + "=" * 80)
    print(f"🏆 BEST CONFIGURATION")
    print("=" * 80)
    print(f"Batch Size: {best['batch_size']}")
    print(f"Best CER: {best['best_cer']:.6f}")
    print(f"Log Directory: {best.get('log_dir', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("Generating comparison plot...")
    
    plt.figure(figsize=(12, 6))
    
    for r in successful[:5]:
        log_dir = r.get('log_dir')
        if log_dir and os.path.exists(log_dir):
            curve = extract_cer_curve(log_dir)
            if curve:
                plt.plot(curve['batches'], curve['cer'], 
                        label=f"Batch {r['batch_size']} (best: {r['best_cer']:.4f})",
                        linewidth=2)
    
    plt.xlabel('Batch Number', fontsize=12)
    plt.ylabel('Character Error Rate (CER)', fontsize=12)
    plt.title('CER Comparison: Different Batch Sizes', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max([r['best_cer'] for r in successful]) * 1.2)
    
    plot_path = '/home/tianlezheng/UCLA-ECE243-FALL2025/batch_size_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    plt.close()
    
    print("=" * 80)

if __name__ == '__main__':
    analyze_results()

