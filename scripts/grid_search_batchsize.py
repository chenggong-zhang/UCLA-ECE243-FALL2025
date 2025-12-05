#!/usr/bin/env python3
"""
Grid search script for batch size optimization.
Tests different batch sizes and tracks best CER for each configuration.
"""

import os
import subprocess
import json
from datetime import datetime

BATCH_SIZES = [32, 48, 64, 96, 128]

BASE_CONFIG = {
    'lrStart': 0.0008,
    'lrEnd': 0.00008,
    'nUnits': 384,
    'nBatch': 20000,
    'nLayers': 8,
    'scheduler_type': 'cosine_warmup',
    'warmup_steps': 800,
    'early_stopping_patience': 50,
    'max_grad_norm': 1.0,
    'seed': 0,
    'nClasses': 40,
    'nInputFeatures': 256,
    'dropout': 0.3,
    'whiteNoiseSD': 0.8,
    'constantOffsetSD': 0.2,
    'gaussianSmoothWidth': 2.0,
    'strideLen': 4,
    'kernelLen': 0,
    'bidirectional': False,
    'l2_decay': 0.01,
    'use_transformer': True,
    'nhead': 6,
    'dim_feedforward': 1536,
    'timeMasking': True,
    'featureMasking': True,
}

BASE_OUTPUT_DIR = '/home/tianlezheng/UCLA-ECE243-FALL2025/logs/speech_logs'
DATASET_PATH = '/home/tianlezheng/UCLA-ECE243-FALL2025/data/ptDecoder_ctc'
RESULTS_FILE = '/home/tianlezheng/UCLA-ECE243-FALL2025/grid_search_batchsize_results.json'

def create_training_script(batch_size, config):
    """Create a temporary training script for this batch size."""
    script_content = f"""
modelName = 'grid_search_batch{batch_size}'

args = {{}}
args['outputDir'] = '{BASE_OUTPUT_DIR}/' + modelName
args['datasetPath'] = '{DATASET_PATH}'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = {batch_size}
args['lrStart'] = {config['lrStart']}
args['lrEnd'] = {config['lrEnd']}
args['nUnits'] = {config['nUnits']}
args['nBatch'] = {config['nBatch']}
args['nLayers'] = {config['nLayers']}
args['scheduler_type'] = '{config['scheduler_type']}'
args['warmup_steps'] = {config['warmup_steps']}
args['early_stopping_patience'] = {config['early_stopping_patience']}
args['max_grad_norm'] = {config['max_grad_norm']}
args['seed'] = {config['seed']}
args['nClasses'] = {config['nClasses']}
args['nInputFeatures'] = {config['nInputFeatures']}
args['dropout'] = {config['dropout']}
args['whiteNoiseSD'] = {config['whiteNoiseSD']}
args['constantOffsetSD'] = {config['constantOffsetSD']}
args['gaussianSmoothWidth'] = {config['gaussianSmoothWidth']}
args['strideLen'] = {config['strideLen']}
args['kernelLen'] = {config['kernelLen']}
args['bidirectional'] = {config['bidirectional']}
args['l2_decay'] = {config['l2_decay']}
args['use_transformer'] = {config['use_transformer']}
args['nhead'] = {config['nhead']}
args['dim_feedforward'] = {config['dim_feedforward']}
args['timeMasking'] = {config['timeMasking']}
args['featureMasking'] = {config['featureMasking']}

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
"""
    script_path = f'/tmp/train_batch{batch_size}.py'
    with open(script_path, 'w') as f:
        f.write(script_content)
    return script_path

def extract_best_cer(log_dir):
    """Extract the best CER from training stats."""
    import pickle
    import numpy as np
    
    stats_path = os.path.join(log_dir, 'trainingStats')
    if not os.path.exists(stats_path):
        return None
    
    try:
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        if 'testCER' in stats and len(stats['testCER']) > 0:
            best_cer = np.min(stats['testCER'])
            return float(best_cer)
    except Exception as e:
        print(f"Error reading stats: {e}")
    return None

def run_grid_search():
    """Run grid search over batch sizes."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': BASE_CONFIG,
        'batch_sizes': BATCH_SIZES,
        'results': []
    }
    
    print("=" * 80)
    print("GRID SEARCH: Batch Size Optimization")
    print("=" * 80)
    print(f"Testing batch sizes: {BATCH_SIZES}")
    print(f"Base config: LR={BASE_CONFIG['lrStart']}, nBatch={BASE_CONFIG['nBatch']}")
    print("=" * 80)
    
    for i, batch_size in enumerate(BATCH_SIZES, 1):
        print(f"\n[{i}/{len(BATCH_SIZES)}] Testing batch_size = {batch_size}")
        print("-" * 80)
        
        script_path = create_training_script(batch_size, BASE_CONFIG)
        log_dir = os.path.join(BASE_OUTPUT_DIR, f'grid_search_batch{batch_size}')
        
        try:
            print(f"Starting training... (output: {log_dir})")
            result = subprocess.run(
                ['python', script_path],
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                best_cer = extract_best_cer(log_dir)
                if best_cer is not None:
                    print(f"✓ Training completed. Best CER: {best_cer:.6f}")
                    results['results'].append({
                        'batch_size': batch_size,
                        'best_cer': best_cer,
                        'status': 'success',
                        'log_dir': log_dir
                    })
                else:
                    print(f"⚠ Training completed but couldn't extract CER")
                    results['results'].append({
                        'batch_size': batch_size,
                        'best_cer': None,
                        'status': 'no_stats',
                        'log_dir': log_dir
                    })
            else:
                print(f"✗ Training failed with return code {result.returncode}")
                results['results'].append({
                    'batch_size': batch_size,
                    'best_cer': None,
                    'status': 'failed',
                    'log_dir': log_dir
                })
        
        except Exception as e:
            print(f"✗ Error during training: {e}")
            results['results'].append({
                'batch_size': batch_size,
                'best_cer': None,
                'status': 'error',
                'error': str(e)
            })
        
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("GRID SEARCH SUMMARY")
    print("=" * 80)
    print(f"{'Batch Size':<15} {'Best CER':<15} {'Status':<15}")
    print("-" * 80)
    
    successful_results = [r for r in results['results'] if r.get('best_cer') is not None]
    if successful_results:
        successful_results.sort(key=lambda x: x['best_cer'])
        for r in results['results']:
            cer_str = f"{r['best_cer']:.6f}" if r.get('best_cer') else "N/A"
            print(f"{r['batch_size']:<15} {cer_str:<15} {r['status']:<15}")
        
        best = successful_results[0]
        print("-" * 80)
        print(f"🏆 BEST: batch_size={best['batch_size']}, CER={best['best_cer']:.6f}")
    else:
        print("No successful runs to compare.")
    
    print(f"\nFull results saved to: {RESULTS_FILE}")
    print("=" * 80)
    
    return results

if __name__ == '__main__':
    results = run_grid_search()

