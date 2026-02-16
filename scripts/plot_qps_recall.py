#!/usr/bin/env python3
"""
Script to plot QPS vs Recall curves comparing STATE_SEND and SCATTER_GATHER 
approaches across different server configurations.
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Define consistent colors for each method
METHOD_COLORS = {
    'STATE_SEND': '#1f77b4',      # Blue
    'SCATTER_GATHER': '#ff7f0e',   # Orange
}


def parse_folder_name(folder_name):
    """
    Parse the log folder name to extract metadata.
    Expected format: logs_${DIST_SEARCH_MODE}_${MODE}_${DATASET_NAME}_${DATASET_SIZE}_${NUM_SERVERS}_${COUNTER_SLEEP_MS}_MS_NUM_SEARCH_THREADS_${NUM_SEARCH_THREADS}_MAX_BATCH_SIZE_${MAX_BATCH_SIZE}_K_${K_VALUE}_OVERLAP_${OVERLAP}_BEAMWIDTH_${BEAM_WIDTH}
    
    Returns a dict with extracted fields or None if parsing fails.
    """
    # Only process folders that start with 'logs_'
    if not folder_name.startswith('logs_'):
        return None
    
    # Split by underscore
    parts = folder_name.split('_')
    
    try:
        # Extract DIST_SEARCH_MODE (indices 1-2)
        if len(parts) < 8:  # Minimum: logs + 2(mode) + mode + dataset + size + numservers + counter + MS
            return None
            
        dist_search_mode = f"{parts[1]}_{parts[2]}"  # STATE_SEND or SCATTER_GATHER
        
        # Validate it's one of the expected modes
        if dist_search_mode not in ['STATE_SEND', 'SCATTER_GATHER']:
            return None
        
        # Extract remaining fields
        mode = parts[3]  # distributed, centralized, etc.
        dataset_name = parts[4]  # bigann, etc.
        dataset_size = parts[5]  # 1B, 100M, etc.
        num_servers = int(parts[6])  # Should be an integer
        counter_sleep_ms = int(parts[7])  # Should be an integer before "MS"
        
        # Verify the "MS" marker is present
        if len(parts) <= 8 or parts[8] != 'MS':
            return None
            
        return {
            'dist_search_mode': dist_search_mode,
            'mode': mode,
            'dataset_name': dataset_name,
            'dataset_size': dataset_size,
            'num_servers': num_servers,
            'counter_sleep_ms': counter_sleep_ms,
            'full_name': folder_name
        }
    except (IndexError, ValueError):
        return None

def parse_client_log(log_file_path):
    """
    Parse client.log file and extract QPS and Recall data.
    
    Returns a list of tuples: (qps, latency, recall)
    """
    data_points = []
    
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
        
        # Find the header line
        header_found = False
        data_start_idx = 0
        
        for i, line in enumerate(lines):
            if 'L   I/O Width' in line and 'QPS' in line and 'Recall' in line:
                header_found = True
                data_start_idx = i + 2  # Skip header and separator line
                break
        
        if not header_found:
            print(f"Warning: Header not found in {log_file_path}")
            return data_points
        
        # Parse data lines
        for line in lines[data_start_idx:]:
            line = line.strip()
            if not line:
                continue
            
            # Split by whitespace and extract relevant columns
            parts = line.split()
            if len(parts) < 10:
                continue
            
            try:
                # Extract QPS (column 3), and Recall (column 9)
                qps = float(parts[2])
                avg_latency = float(parts[3])
                # p99_latency = float(parts[4])
                recall = float(parts[-1])
                
                data_points.append((qps, avg_latency, recall / 100.0))  # Convert recall to 0-1 range
            except (ValueError, IndexError):
                continue
    
    except FileNotFoundError:
        print(f"Warning: File not found: {log_file_path}")
    except Exception as e:
        print(f"Error parsing {log_file_path}: {e}")
    
    return data_points


def collect_data(logs_root_folder):
    """
    Collect all data from log folders.
    
    Returns a tuple: (data_dict, dataset_info)
    where data_dict is {num_servers: {dist_search_mode: [(qps, latency, recall), ...]}}
    and dataset_info is {'name': str, 'size': str}
    """
    data = defaultdict(lambda: defaultdict(list))
    dataset_info = {'name': None, 'size': None}
    logs_root = Path(logs_root_folder)
    
    if not logs_root.exists():
        print(f"Error: Root folder '{logs_root_folder}' does not exist")
        return data, dataset_info
    
    # Iterate through all subdirectories
    for folder in logs_root.iterdir():
        if not folder.is_dir():
            continue
        
        # Parse folder name
        metadata = parse_folder_name(folder.name)
        if metadata is None:
            print(f"Skipping folder (couldn't parse): {folder.name}")
            continue
        print(metadata)
        # Store dataset info from first valid folder
        if dataset_info['name'] is None and metadata['dataset_name']:
            dataset_info['name'] = metadata['dataset_name']
            dataset_info['size'] = metadata['dataset_size']
        
        # Look for client.log file
        client_log = folder / 'client.log'
        if not client_log.exists():
            print(f"Warning: client.log not found in {folder.name}")
            continue
        
        # Parse the log file
        data_points = parse_client_log(client_log)
        if not data_points:
            print(f"Warning: No data extracted from {client_log}")
            continue
        
        # Store data
        num_servers = metadata['num_servers']
        mode = metadata['dist_search_mode']
        data[num_servers][mode] = data_points
        
        print(f"Loaded {len(data_points)} data points from {folder.name}")
    
    return data, dataset_info


def plot_tput_acc(data, dataset_info, min_recall):
    """
    Plot throughput vs accuracy for different server configurations.
    
    data: {num_servers: {dist_search_mode: [(qps, latency, recall), ...]}}
    dataset_info: {'name': str, 'size': str}
    """
    print(data)
    num_configs = len(data)
    if num_configs == 0:
        print("No data to plot!")
        return None
    
    fig, axes = plt.subplots(nrows=1, ncols=num_configs, 
                            figsize=(5 * num_configs, 4), squeeze=False)
    axes_flat = axes.flatten()
    
    # Sort by number of servers
    sorted_configs = sorted(data.items())
    
    # First pass: find global min/max QPS values for aligned y-axis
    all_qps_values = []
    for num_servers, methods_data in sorted_configs:
        for method_name, data_points in methods_data.items():
            y_values = [point[0] for point in data_points if len(point) >= 3]
            all_qps_values.extend(y_values)
    
    if all_qps_values:
        global_min_qps = min(all_qps_values)
        global_max_qps = max(all_qps_values)
        # Add 5% padding
        qps_range = global_max_qps - global_min_qps
        y_min = global_min_qps - 0.05 * qps_range
        y_max = global_max_qps + 0.05 * qps_range
    else:
        y_min, y_max = None, None
    
    # Second pass: plot with aligned y-axis
    # Plot in specific order to control legend order: STATE_SEND first, then SCATTER_GATHER
    method_order = ['STATE_SEND', 'SCATTER_GATHER']
    
    for i, (num_servers, methods_data) in enumerate(sorted_configs):
        ax = axes_flat[i]
        
        # Plot methods in specified order
        for method_name in method_order:
            if method_name not in methods_data:
                continue
                
            data_points = methods_data[method_name]
            
            # Extract recall (x) and QPS (y) values
            x_values = [point[2] for point in data_points if len(point) >= 3]
            y_values = [point[0] for point in data_points if len(point) >= 3]
            
            if x_values:
                # Sort by recall
                sorted_points = sorted(zip(x_values, y_values))
                x_values_sorted, y_values_sorted = zip(*sorted_points)
                
                # Get consistent color for this method
                color = METHOD_COLORS.get(method_name, None)
                
                ax.plot(x_values_sorted, y_values_sorted, 
                       label=method_name, marker='o', linestyle='-', 
                       linewidth=2, color=color)
            else:
                print(f"No valid points for {method_name} with {num_servers} servers")
        
        # Create title with dataset info and number of servers
        if dataset_info['name'] and dataset_info['size']:
            title = f"{dataset_info['name']} ({dataset_info['size']}) - {num_servers} Servers"
        else:
            title = f'{num_servers} Servers'
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Recall@10', fontsize=11)
        ax.set_xlim(min_recall, 1.01)
        ax.set_ylabel('Throughput (QPS)', fontsize=11)
        
        # Set aligned y-axis limits
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot QPS vs Recall curves comparing STATE_SEND and SCATTER_GATHER approaches'
    )
    parser.add_argument(
        'logs_folder',
        type=str,
        help='Path to the root folder containing all log subfolders'
    )
    parser.add_argument(
        '--min-recall',
        type=float,
        default=0.8,
        help='Minimum recall value to display on x-axis (default: 0.8)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='qps_recall_comparison.png',
        help='Output filename for the plot (default: qps_recall_comparison.png)'
    )
    
    args = parser.parse_args()
    
    print(f"Collecting data from: {args.logs_folder}")
    data, dataset_info = collect_data(args.logs_folder)
    
    if not data:
        print("No data collected. Please check your log folder structure.")
        return
    
    print(f"\nFound data for {len(data)} server configuration(s)")
    print(f"Dataset: {dataset_info['name']} ({dataset_info['size']})")
    for num_servers in sorted(data.keys()):
        print(f"  {num_servers} servers: {list(data[num_servers].keys())}")
    
    print(f"\nGenerating plot...")
    fig = plot_tput_acc(data, dataset_info, args.min_recall)
    
    if fig:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {args.output}")
    else:
        print("Failed to generate plot.")

if __name__ == '__main__':
    main()
