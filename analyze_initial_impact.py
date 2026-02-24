#!/usr/bin/env python3
"""Analyze impact of initial solution on baseline vs kisialiou."""

import json
from pathlib import Path

results_file = Path('output/benchmark_10runs_200iter.json')
with open(results_file, 'r') as f:
    results = json.load(f)

print('Baseline vs Kisialiou - Initial solution quality impact:')
print('='*80)
print(f"{'Run':<5} {'Mode':<12} {'Initial':<12} {'Best':<12} {'Δ Cost':<12} {'Δ%':<8}")
print('='*80)

for run_id in range(10):
    baseline = [r for r in results if r['mode'] == 'baseline' and r['run_id'] == run_id][0]
    kisialiou = [r for r in results if r['mode'] == 'kisialiou' and r['run_id'] == run_id][0]
    
    bl_init = baseline['initial_cost']
    bl_best = baseline['best_cost']
    bl_delta = bl_init - bl_best
    bl_pct = baseline['improvement_pct']
    
    ki_init = kisialiou['initial_cost']
    ki_best = kisialiou['best_cost']
    ki_delta = ki_init - ki_best
    ki_pct = kisialiou['improvement_pct']
    
    # Highlight runs where baseline got lucky with initial solution
    if bl_init < 350000:
        marker = " ⭐"
    else:
        marker = ""
    
    print(f"{run_id:<5} {'baseline':<12} {bl_init:<12.2f} {bl_best:<12.2f} {bl_delta:<12.2f} {bl_pct:<8.2f}{marker}")
    print(f"{'':5} {'kisialiou':<12} {ki_init:<12.2f} {ki_best:<12.2f} {ki_delta:<12.2f} {ki_pct:<8.2f}")
    print()

print("⭐ = Baseline got lucky with initial solution < 350k")
