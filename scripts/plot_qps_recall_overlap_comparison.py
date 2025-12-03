#!/usr/bin/env python3
"""
Script to plot QPS vs Recall curves comparing STATE_SEND with and without overlap
across different server configurations.
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Define consistent colors for overlap vs non-overlap
METHOD_COLORS = {
    'OVERLAP_true': '#1f77b4',      # Blue
    'OVERLAP_false': '#ff7f0e',     # Orange
}

METHOD_LABELS = {
    'OVERLAP_true': 'With Overlap',
    'OVERLAP_false': 'No Overlap',
}


def parse_folder_name(folder_name):
    """
    Parse the log folder name to extract metadata.
    Expected format: 
    logs_STATE_SEND_distributed_bigann_100M_2_10_MS_NUM_SEARCH_THREADS_8_MAX_BATCH_SIZE_8_K_10_OVERLAP_true_LVEC_...
    
    Returns a dict with extracted fields or None if parsing fails.
    """
    # Only process folders that start with 'logs_STATE_SEND'
    if not folder_name.startswith('logs_STATE_SEND'):
        return None
    
    # Split by underscore
    parts = folder_name.split('_')
    
    try:
        # Find number of servers (should be after 100M and before 10)
        num_servers = None
        for i in range(len(parts)):
            if '100M' in parts[i] and i + 1 < len(parts):
                # Next part should be the number of servers
                if parts[i + 1].isdigit() and len(parts[i + 1]) <= 2:
                    num_servers = int(parts[i + 1])
                    break
        
        if num_servers is None:
            return None
        
        # Find OVERLAP value
        overlap = None
        for i in range(len(parts)):
            if parts[i] == 'OVERLAP' and i + 1 < len(parts):
                overlap_str = parts[i + 1].lower()
                if overlap_str in ['true', 'false']:
                    overlap = overlap_str
                    break
        
        if overlap is None:
            return None
        
        return {
            'num_servers': num_servers,
            'overlap': overlap,
            'full_name': folder_name
        }
    
    except (IndexError, ValueError) as e:
        print(f"Error parsing {folder_name}: {e}")
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
            if len(parts) < 8:
                continue
            
            try:
                # Extract QPS (column 3), and Recall (column 8)
                qps = float(parts[2])
                avg_latency = float(parts[3])
                recall = float(parts[7])
                
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
    
    Returns a nested dict: {num_servers: {overlap_config: [(qps, latency, recall), ...]}}
    """
    data = defaultdict(lambda: defaultdict(list))
    logs_root = Path(logs_root_folder)
    
    if not logs_root.exists():
        print(f"Error: Root folder '{logs_root_folder}' does not exist")
        return data
    
    # Iterate through all subdirectories
    for folder in logs_root.iterdir():
        if not folder.is_dir():
            continue
        
        # Parse folder name
        metadata = parse_folder_name(folder.name)
        if metadata is None:
            print(f"Skipping folder (couldn't parse): {folder.name}")
            continue
        
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
        overlap_key = f"OVERLAP_{metadata['overlap']}"
        data[num_servers][overlap_key] = data_points
        
        print(f"Loaded {len(data_points)} data points from {folder.name}")
    
    return data


def plot_tput_acc(data, min_recall):
    """
    Plot throughput vs accuracy for different server configurations.
    
    data: {num_servers: {overlap_config: [(qps, latency, recall), ...]}}
    """
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
    for num_servers, overlap_data in sorted_configs:
        for overlap_config, data_points in overlap_data.items():
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
    # Plot in specific order: no overlap first, then with overlap
    overlap_order = ['OVERLAP_false', 'OVERLAP_true']
    
    for i, (num_servers, overlap_data) in enumerate(sorted_configs):
        ax = axes_flat[i]
        
        # Plot configurations in specified order
        for overlap_config in overlap_order:
            if overlap_config not in overlap_data:
                continue
                
            data_points = overlap_data[overlap_config]
            
            # Extract recall (x) and QPS (y) values
            x_values = [point[2] for point in data_points if len(point) >= 3]
            y_values = [point[0] for point in data_points if len(point) >= 3]
            
            if x_values:
                # Sort by recall
                sorted_points = sorted(zip(x_values, y_values))
                x_values_sorted, y_values_sorted = zip(*sorted_points)
                
                # Get consistent color and label for this configuration
                color = METHOD_COLORS.get(overlap_config, None)
                label = METHOD_LABELS.get(overlap_config, overlap_config)
                
                ax.plot(x_values_sorted, y_values_sorted, 
                       label=label, marker='o', linestyle='-', 
                       linewidth=2, color=color)
            else:
                print(f"No valid points for {overlap_config} with {num_servers} servers")
        
        ax.set_title(f'{num_servers} Servers', fontsize=12, fontweight='bold')
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
        description='Plot QPS vs Recall curves comparing STATE_SEND with and without overlap'
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
        default='overlap_qps_recall.png',
        help='Output filename for the plot (default: overlap_qps_recall.png)'
    )
    
    args = parser.parse_args()
    
    print(f"Collecting data from: {args.logs_folder}")
    data = collect_data(args.logs_folder)
    
    if not data:
        print("No data collected. Please check your log folder structure.")
        return
    
    print(f"\nFound data for {len(data)} server configuration(s)")
    for num_servers in sorted(data.keys()):
        print(f"  {num_servers} servers: {list(data[num_servers].keys())}")
    
    print(f"\nGenerating plot...")
    fig = plot_tput_acc(data, args.min_recall)
    
    if fig:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {args.output}")
    else:
        print("Failed to generate plot.")

if __name__ == '__main__':
    main()
