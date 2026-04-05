#!/usr/bin/env python3
"""
Script to plot scalability of STATE_SEND, SCATTER_GATHER, and SINGLE_SERVER 
approaches as the number of servers/threads increases.
Evaluates all datasets, applying a specific target recall for text2image.
Aggregates runs by taking the average of QPS and Recall for the same L value.
"""

import os
import re
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Define legend name mapping
LEGEND_NAME_MAPPING = {
    'STATE_SEND': 'BatANN',
    'SCATTER_GATHER': 'ScatterGather',
    'SINGLE_SERVER': 'SingleServer',
    'DISTRIBUTED_ANN': 'DistributedANN'
}

# Define consistent colors and markers for each method
METHOD_STYLES = {
    'STATE_SEND': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'label': 'BatANN'},
    'SCATTER_GATHER': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-', 'label': 'ScatterGather'},
    'SINGLE_SERVER': {'color': '#2ca02c', 'marker': '^', 'linestyle': '--', 'label': 'SingleServer'},
    'DISTRIBUTED_ANN': {'color': 'red', 'marker': 'p', 'linestyle': '-', 'label': 'DistributedANN'},
}

# Mapping from raw log dataset names to desired display names
DATASET_NAME_MAPPING = {
    'bigann': 'bigann',
    'deep1b': 'deep',
    'MSSPACEV1B': 'msspacev1b',
    'msspacev1b': 'msspacev1b',
    'text2image1B' : 'text2image',
}


def parse_folder_name(folder_name):
    """Parse the log folder name to extract metadata."""
    if not folder_name.startswith('logs_'):
        return None
    
    parts = folder_name.split('_')
    if len(parts) < 6:
        return None
    
    try:
        if parts[1] == 'STATE' and parts[2] == 'SEND':
            method, start_idx = 'STATE_SEND', 3
        elif parts[1] == 'SCATTER' and parts[2] == 'GATHER':
            method, start_idx = 'SCATTER_GATHER', 3
        elif parts[1] == 'SINGLE' and parts[2] == 'SERVER':
            method, start_idx = 'SINGLE_SERVER', 3
        elif parts[1] == 'DISTRIBUTED' and parts[2] == 'ANN':
            method, start_idx = 'DISTRIBUTED_ANN', 3            
        else:
            return None
        
        dataset_name = parts[start_idx + 1] if start_idx + 1 < len(parts) else None
        dataset_size = parts[start_idx + 2] if start_idx + 2 < len(parts) else None
        
        num_servers = None
        num_threads = None
        
        for i in range(len(parts)):
            if parts[i] == 'THREADS' and i > 0 and parts[i-1] == 'SEARCH' and i > 1 and parts[i-2] == 'NUM':
                if i + 1 < len(parts) and parts[i+1].isdigit():
                    num_threads = int(parts[i+1])
                    break
        
        if method == 'SINGLE_SERVER':
            if num_threads is None: return None
            return {
                'method': method, 'num_servers': 1, 'num_threads': num_threads,
                'dataset_name': dataset_name, 'dataset_size': dataset_size, 'full_name': folder_name
            }
        else:
            for i in range(start_idx + 2, len(parts)):
                if parts[i].isdigit() and len(parts[i]) <= 2:
                    if i > 0 and any(char in parts[i-1] for char in ['M', 'K', 'G', 'B']):
                        num_servers = int(parts[i])
                        break
            
            if num_servers is None: return None
            return {
                'method': method, 'num_servers': num_servers, 'num_threads': num_threads if num_threads else 8,
                'dataset_name': dataset_name, 'dataset_size': dataset_size, 'full_name': folder_name
            }
    except (IndexError, ValueError):
        return None


def parse_client_log(log_file_path):
    """Parse client.log file and extract L, QPS and Recall data."""
    data_points = []
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
        
        header_found = False
        data_start_idx = 0
        
        for i, line in enumerate(lines):
            if 'L   I/O Width' in line and 'QPS' in line and 'Recall' in line:
                header_found = True
                data_start_idx = i + 2  
                break
        
        if not header_found: return data_points
        
        for line in lines[data_start_idx:]:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 9: continue
            
            try:
                l_val = int(parts[0])  # Extract L value for grouping
                qps = float(parts[2])
                avg_latency = float(parts[3])
                recall = float(parts[-1])
                data_points.append((l_val, qps, avg_latency, recall / 100.0))
            except (ValueError, IndexError):
                continue
    except Exception:
        pass
    
    return data_points


def get_qps_at_recall(data_points, target_recall):
    """Get QPS at target recall using linear interpolation."""
    if not data_points: return None
    # Elements in data_points are now expected to be (qps, latency, recall)
    sorted_points = sorted(data_points, key=lambda x: x[2])
    
    min_recall = sorted_points[0][2]
    max_recall = sorted_points[-1][2]
    
    if target_recall < min_recall or target_recall > max_recall:
        return None
    
    for i in range(len(sorted_points) - 1):
        recall_low = sorted_points[i][2]
        recall_high = sorted_points[i + 1][2]
        
        if recall_low <= target_recall <= recall_high:
            qps_low = sorted_points[i][0]
            qps_high = sorted_points[i + 1][0]
            
            if recall_high == recall_low:
                return qps_low
            else:
                weight = (target_recall - recall_low) / (recall_high - recall_low)
                return qps_low + weight * (qps_high - qps_low)
    
    for qps, _, recall in sorted_points:
        if abs(recall - target_recall) < 1e-6:
            return qps
            
    return None


def collect_data(logs_root_folder, default_recall, t2i_recall):
    """Collect QPS at target recall for methods, organized by dataset, aggregating by L."""
    datasets = defaultdict(lambda: {'data': defaultdict(dict), 'info': {'name': None, 'size': None, 'target_recall': None}})
    
    # Store raw data for averaging: dataset_key -> method -> config_key -> l_val -> [(qps, lat, rec), ...]
    raw_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    logs_root = Path(logs_root_folder)
    
    if not logs_root.exists():
        print(f"Error: Root folder '{logs_root_folder}' does not exist")
        return datasets
    
    # Pass 1: Gather all raw data points mapped by their L values
    for folder in logs_root.iterdir():
        if not folder.is_dir(): continue
        
        metadata = parse_folder_name(folder.name)
        if metadata is None: continue
        
        dataset_name = metadata['dataset_name']
        dataset_size = metadata['dataset_size']
        dataset_key = f"{dataset_name}_{dataset_size}"
        
        # Apply dataset-specific recall rule
        if 'text2image' in dataset_name.lower():
            current_target_recall = t2i_recall
        else:
            current_target_recall = default_recall
            
        if datasets[dataset_key]['info']['name'] is None:
            # Map name if exists, otherwise keep original
            mapped_name = DATASET_NAME_MAPPING.get(dataset_name, dataset_name)
            datasets[dataset_key]['info']['name'] = mapped_name
            datasets[dataset_key]['info']['size'] = dataset_size
            datasets[dataset_key]['info']['target_recall'] = current_target_recall
        
        client_log = folder / 'client.log'
        if not client_log.exists(): continue
        
        data_points = parse_client_log(client_log)
        
        method = metadata['method']
        key = metadata['num_threads'] if method == 'SINGLE_SERVER' else metadata['num_servers']
        
        # Store points by L value for averaging later
        for l_val, qps, lat, rec in data_points:
            raw_data[dataset_key][method][key][l_val].append((qps, lat, rec))
            
    # Pass 2: Average points for the same L value, then find QPS at target recall
    for dataset_key, methods_dict in raw_data.items():
        target_recall = datasets[dataset_key]['info']['target_recall']
        
        for method, config_dict in methods_dict.items():
            for key, l_dict in config_dict.items():
                averaged_points = []
                
                for l_val, points in l_dict.items():
                    avg_qps = np.mean([p[0] for p in points])
                    avg_lat = np.mean([p[1] for p in points])
                    avg_rec = np.mean([p[2] for p in points])
                    averaged_points.append((avg_qps, avg_lat, avg_rec))
                    
                qps = get_qps_at_recall(averaged_points, target_recall)
                
                if qps is not None:
                    datasets[dataset_key]['data'][method][key] = qps
    
    return {k: v for k, v in datasets.items() if any(v['data'].values())}


def print_summary_table(datasets):
    """Prints a formatted summary table of QPS and Scalability to the console."""
    print(f"\n{'='*75}")
    print(f" SUMMARY TABLE")
    print(f"{'='*75}")
    
    for dataset_key, dataset in sorted(datasets.items()):
        recall_used = dataset['info']['target_recall']
        display_name = dataset['info']['name']
        print(f"\nDataset: {display_name} (Evaluated @ {recall_used} Recall)")
        print(f"{'Method':<18} | {'Hardware (Nodes/Thr)':<20} | {'QPS':<10} | {'Scalability':<10}")
        print(f"{'-'*18}-+-{'-'*20}-+-{'-'*10}-+-{'-'*10}")
        
        data = dataset['data']
        baseline_qps = None
        
        if 'SINGLE_SERVER' in data and 8 in data['SINGLE_SERVER']:
            baseline_qps = data['SINGLE_SERVER'][8]
        elif 'SINGLE_SERVER' in data and data['SINGLE_SERVER']:
            baseline_qps = data['SINGLE_SERVER'][min(data['SINGLE_SERVER'].keys())]
            
        for method in ['SINGLE_SERVER', 'STATE_SEND', 'SCATTER_GATHER', 'DISTRIBUTED_ANN']:
            if method not in data: continue
            
            display_name = LEGEND_NAME_MAPPING.get(method, method)
            for key, qps in sorted(data[method].items()):
                hw_str = f"{key} threads" if method == 'SINGLE_SERVER' else f"{key} servers"
                speedup_str = f"{qps/baseline_qps:.2f}x" if baseline_qps else "N/A"
                print(f"{display_name:<18} | {hw_str:<20} | {qps:<10.2f} | {speedup_str:<10}")
    print(f"\n{'='*75}\n")


def plot_single_dataset(ax, data, dataset_info, threads_per_server=8, show_legend=True, show_ylabel=False):
    """Plot scalability comparison for a single dataset on a given axis."""
    if not data: return False
    
    baseline_qps = data.get('SINGLE_SERVER', {}).get(8, None)
    if not baseline_qps and data.get('SINGLE_SERVER'):
        baseline_qps = data['SINGLE_SERVER'][min(data['SINGLE_SERVER'].keys())]
        
    if not baseline_qps:
        ax.text(0.5, 0.5, 'No baseline data', ha='center', va='center', transform=ax.transAxes)
        return False
    
    max_x = 1
    for method in ['STATE_SEND', 'SCATTER_GATHER', 'SINGLE_SERVER', 'DISTRIBUTED_ANN']:
        if method not in data or not data[method]: continue
        if method == 'SINGLE_SERVER':
            max_x = max(max_x, max(data[method].keys()) / threads_per_server)
        else:
            max_x = max(max_x, max(data[method].keys()))
    
    ax.plot([1, max_x], [1, max_x], 'k--', linewidth=1.5, alpha=0.5, label='Optimal')
    
    for method in ['STATE_SEND', 'SCATTER_GATHER', 'SINGLE_SERVER', 'DISTRIBUTED_ANN']:
        if method not in data or not data[method]: continue
        
        style = METHOD_STYLES[method]
        x_values, y_values = [], []
        
        if method == 'SINGLE_SERVER':
            for num_threads, qps in sorted(data[method].items()):
                x_values.append(num_threads / threads_per_server)
                y_values.append(qps / baseline_qps)
        else:
            x_values.append(1)
            y_values.append(1)
            for num_servers, qps in sorted(data[method].items()):
                x_values.append(num_servers)
                y_values.append(qps / baseline_qps)
        
        ax.plot(x_values, y_values, label=style['label'], color=style['color'],
                marker=style['marker'], linestyle=style['linestyle'],
                linewidth=2.5, markersize=10)
    
    if show_ylabel:
        ax.set_ylabel('Scalability (Speedup)', fontsize=20)
    
    title = f"{dataset_info['name']} @ {dataset_info['target_recall']} Recall"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    if show_legend:
        ax.legend(fontsize=10, loc='upper left')
    
    # Restrict X-axis ticks to exactly [1, 5, 10]
    ax.set_xticks([1, 5, 10])
    
    # Increase the tick label size on both axes
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    ax.set_ylim(bottom=0)
    return True


def plot_scalability_grid(datasets, threads_per_server=8):
    """Plot scalability comparison for multiple datasets in a single horizontal row."""
    n_datasets = len(datasets)
    if n_datasets == 0: return None
    
    # Create 1 row and 'n_datasets' columns
    fig, axes = plt.subplots(1, n_datasets, figsize=(3.5 * n_datasets, 4))
    
    if n_datasets == 1:
        axes = [axes]
        
    sorted_dataset_keys = sorted(datasets.keys(), key=lambda x: x.lower())
    
    for idx, dataset_key in enumerate(sorted_dataset_keys):
        dataset = datasets[dataset_key]
        ax = axes[idx]
        
        # Show legend and ylabel ONLY on the first plot
        is_first = (idx == 0)
        plot_single_dataset(ax, dataset['data'], dataset['info'], threads_per_server, show_legend=is_first, show_ylabel=is_first)
        
    # Set a single, global X-axis title at the bottom of the figure
    fig.supxlabel('Number of Servers (or Equivalent Threads)', fontsize=20, y=+0.05)
        
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Plot scalability for ANN Search approaches')
    parser.add_argument('logs_folder', type=str, help='Path to the root folder containing all log subfolders')
    parser.add_argument('--default-target-recall', type=float, default=0.95, help='Default target recall (default: 0.95)')
    parser.add_argument('--t2i-target-recall', type=float, default=0.70, help='Target recall specifically for text2image datasets (default: 0.70)')
    parser.add_argument('--threads-per-server', type=int, default=8, help='Threads per server (default: 8)')
    parser.add_argument('--output', type=str, default='scalability_comparison.png', help='Output filename')
    args = parser.parse_args()
    
    print(f"Collecting data from: {args.logs_folder}")
    print(f"Default target recall:    {args.default_target_recall}")
    print(f"Text2Image target recall: {args.t2i_target_recall}")
    
    datasets = collect_data(args.logs_folder, args.default_target_recall, args.t2i_target_recall)
    
    if not datasets:
        print("\nNo data collected! Check if your recall targets are achievable in the logs.")
        return
    
    print_summary_table(datasets)
    
    fig = plot_scalability_grid(datasets, args.threads_per_server)
    if fig:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {args.output}")

if __name__ == '__main__':
    main()
