#!/usr/bin/env python3
"""Run benchmark multiple times with different seeds."""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

def main():
    results = []
    base_seed_start = 4242
    num_runs = 10
    iterations = 200
    
    for run_idx in range(num_runs):
        seed = base_seed_start + run_idx * 100
        print(f'\n=== Run {run_idx + 1}/{num_runs} with seed {seed} ===', flush=True)
        
        subprocess.run([
            sys.executable, 'scripts/alns_mode_benchmark.py',
            '--sizes', 'small',
            '--modes', 'baseline', 'kisialiou',
            '--iterations', str(iterations),
            '--datasets-per-size', '1',
            '--base-seed', str(seed),
            '--results-path', f'output/benchmark_run_{run_idx}.json',
            '--log-path', f'output/benchmark_run_{run_idx}.log',
            '--log-dir', 'logs'
        ], cwd=REPO_ROOT)
        
        # Load and append results
        result_file = REPO_ROOT / f'output/benchmark_run_{run_idx}.json'
        with open(result_file, 'r') as f:
            run_results = json.load(f)
            for r in run_results:
                r['run_id'] = run_idx
            results.extend(run_results)
    
    # Save combined results
    output_path = REPO_ROOT / 'output/benchmark_10runs_200iter.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\n=== Combined results saved to {output_path} ===')
    print(f'Total runs: {len(results)}')
    
    # Print summary
    by_mode = {}
    for r in results:
        mode = r['mode']
        if mode not in by_mode:
            by_mode[mode] = {'costs': [], 'improvements': [], 'runtimes': []}
        by_mode[mode]['costs'].append(r['best_cost'])
        by_mode[mode]['improvements'].append(r['improvement_pct'])
        by_mode[mode]['runtimes'].append(r['runtime_seconds'])
    
    print('\n=== Summary ===')
    for mode, stats in by_mode.items():
        best_cost = min(stats['costs'])
        avg_cost = sum(stats['costs']) / len(stats['costs'])
        avg_impr = sum(stats['improvements']) / len(stats['improvements'])
        avg_time = sum(stats['runtimes']) / len(stats['runtimes'])
        print(f"{mode:24} | Best: {best_cost:12.2f} | Avg: {avg_cost:12.2f} | Avg Δ%: {avg_impr:6.2f} | Avg Time: {avg_time:6.2f}s")

if __name__ == '__main__':
    main()
