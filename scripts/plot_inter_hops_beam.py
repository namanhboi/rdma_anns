#!/usr/bin/env python3
"""
Script to plot inter-partition hops comparison across different beamwidths.
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Define consistent colors for different beamwidths
BEAMWIDTH_COLORS = {
    1: '#1f77b4',   # Blue
    2: '#ff7f0e',   # Orange
    4: '#2ca02c',   # Green
    8: '#d62728',   # Red
    16: '#9467bd',  # Purple
    32: '#8c564b',  # Brown
}


def parse_folder_name(folder_name):
    """
    Parse the log folder name to extract metadata.
    Expected format:
    - logs_STATE_SEND_distributed_${DATASET_NAME}_${DATASET_SIZE}_${NUM_SERVERS}_...BEAMWIDTH_${BEAM_WIDTH}_timestamp
    
    Returns a dict with extracted fields or None if parsing fails.
    """
    # Determine the method
    method = None
    if folder_name.startswith('logs_STATE_SEND'):
        method = 'STATE_SEND'
    elif folder_name.startswith('logs_SCATTER_GATHER'):
        method = 'SCATTER_GATHER'
    elif folder_name.startswith('logs_SINGLE_SERVER'):
        method = 'SINGLE_SERVER'
    else:
        return None
    
    # Extract beamwidth using regex
    beamwidth_match = re.search(r'BEAMWIDTH_(\d+)', folder_name)
    if not beamwidth_match:
        return None
    beamwidth = int(beamwidth_match.group(1))
    
    # Extract dataset name and size
    if method in ['STATE_SEND', 'SCATTER_GATHER']:
        dataset_match = re.search(r'distributed_(\w+)_(\d+[BKMG])', folder_name)
        if not dataset_match:
            return None
        dataset_name = dataset_match.group(1)
        dataset_size = dataset_match.group(2)
        
        # Extract number of servers
        num_servers_match = re.search(r'_(\d+[BKMG])_(\d+)_\d+_MS', folder_name)
        if not num_servers_match:
            return None
        num_servers = int(num_servers_match.group(2))
    else:  # SINGLE_SERVER
        dataset_match = re.search(r'logs_SINGLE_SERVER_(\w+)_(\d+[BKMG])', folder_name)
        if not dataset_match:
            return None
        dataset_name = dataset_match.group(1)
        dataset_size = dataset_match.group(2)
        
        # Extract number of search threads for SINGLE_SERVER
        num_threads_match = re.search(r'NUM_SEARCH_THREADS_(\d+)', folder_name)
        if not num_threads_match:
            return None
        num_threads = int(num_threads_match.group(1))
        
        # Map threads to equivalent number of servers (8 threads = 1 server)
        num_servers = num_threads // 8
        if num_servers == 0:
            return None
    
    return {
        'method': method,
        'beamwidth': beamwidth,
        'num_servers': num_servers,
        'dataset_name': dataset_name,
        'dataset_size': dataset_size,
        'full_name': folder_name
    }


def parse_client_log(log_file_path):
    """
    Parse client.log file and extract QPS, Recall, and Mean Inter-partition Hops data.
    
    Returns a list of tuples: (qps, latency, recall, mean_inter_partition_hops)
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
                # Extract QPS (column 3), Recall (column 9), and Mean Inter-partition Hops (column 6)
                qps = float(parts[2])
                avg_latency = float(parts[3])
                mean_inter_partition_hops = float(parts[6])
                recall = float(parts[-1])
                
                data_points.append((qps, avg_latency, recall / 100.0, mean_inter_partition_hops))
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line '{line}': {e}")
                continue
    
    except FileNotFoundError:
        print(f"Warning: File not found: {log_file_path}")
    except Exception as e:
        print(f"Error parsing {log_file_path}: {e}")
    
    return data_points


def collect_data(logs_folders):
    """
    Collect all data from log folders.
    
    Args:
        logs_folders: List of paths to root folders containing log subfolders
    
    Returns a tuple: (data_dict, dataset_info)
    where data_dict is {num_servers: {beamwidth: [(qps, latency, recall, mean_hops), ...]}}
    and dataset_info is {'name': str, 'size': str}
    """
    data = defaultdict(lambda: defaultdict(list))
    dataset_info = {'name': None, 'size': None}
    
    for logs_root_folder in logs_folders:
        logs_root = Path(logs_root_folder)
        
        if not logs_root.exists():
            print(f"Warning: Root folder '{logs_root_folder}' does not exist, skipping")
            continue
        
        print(f"Scanning folder: {logs_root_folder}")
        
        # Iterate through all subdirectories (including symlinks)
        for folder in logs_root.iterdir():
            if not folder.is_dir():
                continue
            
            # Parse folder name
            metadata = parse_folder_name(folder.name)
            if metadata is None:
                print(f"  Skipping folder (couldn't parse): {folder.name}")
                continue
            
            # Store dataset info from first valid folder
            if dataset_info['name'] is None and metadata['dataset_name']:
                dataset_info['name'] = metadata['dataset_name']
                dataset_info['size'] = metadata['dataset_size']
            
            # Look for client.log file
            client_log = folder / 'client.log'
            if not client_log.exists():
                print(f"  Warning: client.log not found in {folder.name}")
                continue
            
            # Parse the log file
            data_points = parse_client_log(client_log)
            if not data_points:
                print(f"  Warning: No data extracted from {client_log}")
                continue
            
            # Store data with beamwidth as key
            num_servers = metadata['num_servers']
            beamwidth = metadata['beamwidth']
            method = metadata['method']
            
            data[num_servers][beamwidth] = data_points
            print(f"  Loaded {len(data_points)} data points from {folder.name} (method={method}, beamwidth={beamwidth}, servers={num_servers})")
    
    return data, dataset_info


def interpolate_hops_at_recall(x_values, y_values, target_recall):
    """
    Interpolate to find the number of hops at a specific recall value.
    
    Args:
        x_values: list of recall values (sorted)
        y_values: list of hop values (corresponding to recalls)
        target_recall: the recall value to interpolate at
    
    Returns:
        Interpolated hop value or None if target_recall is out of range
    """
    if not x_values or not y_values:
        return None
    
    # Check if target is in range
    if target_recall < min(x_values) or target_recall > max(x_values):
        return None
    
    # Use numpy interpolation
    return np.interp(target_recall, x_values, y_values)


def plot_inter_partition_hops(data, dataset_info, min_recall):
    """
    Plot inter-partition hops vs recall for different beamwidths.
    
    data: {num_servers: {beamwidth: [(qps, latency, recall, mean_hops), ...]}}
    dataset_info: {'name': str, 'size': str}
    """
    num_configs = len(data)
    if num_configs == 0:
        print("No data to plot!")
        return None
    
    # Print interpolated values at recall 0.90 and 0.95
    print("\n" + "="*80)
    print("Inter-partition Hops at Specific Recall Values")
    print("="*80)
    
    for num_servers in sorted(data.keys()):
        print(f"\n{num_servers} Server{'s' if num_servers > 1 else ''}:")
        print(f"{'Beamwidth':<12} {'Hops @ 0.90':<15} {'Hops @ 0.95':<15}")
        print("-" * 42)
        
        for beamwidth in sorted(data[num_servers].keys()):
            data_points = data[num_servers][beamwidth]
            
            # Extract recall and hops
            x_values = [point[2] for point in data_points if len(point) >= 4]
            y_values = [point[3] for point in data_points if len(point) >= 4]
            
            if x_values:
                # Sort by recall
                sorted_points = sorted(zip(x_values, y_values))
                x_sorted, y_sorted = zip(*sorted_points)
                
                # Interpolate at 0.90 and 0.95
                hops_at_90 = interpolate_hops_at_recall(x_sorted, y_sorted, 0.90)
                hops_at_95 = interpolate_hops_at_recall(x_sorted, y_sorted, 0.95)
                
                hops_90_str = f"{hops_at_90:.4f}" if hops_at_90 is not None else "N/A"
                hops_95_str = f"{hops_at_95:.4f}" if hops_at_95 is not None else "N/A"
                
                print(f"{beamwidth:<12} {hops_90_str:<15} {hops_95_str:<15}")
    
    print("\n" + "="*80 + "\n")
    
    fig, axes = plt.subplots(nrows=1, ncols=num_configs, 
                            figsize=(3 * num_configs, 4.5), squeeze=False)
    axes_flat = axes.flatten()
    
    # Sort by number of servers
    sorted_configs = sorted(data.items())
    
    # First pass: find global min/max hop values for aligned y-axis
    all_hop_values = []
    for num_servers, beamwidths_data in sorted_configs:
        for beamwidth, data_points in beamwidths_data.items():
            y_values = [point[3] for point in data_points if len(point) >= 4]
            all_hop_values.extend(y_values)
    
    if all_hop_values:
        global_min_hops = min(all_hop_values)
        global_max_hops = max(all_hop_values)
        # Add 5% padding
        hops_range = global_max_hops - global_min_hops
        y_min = max(0, global_min_hops - 0.05 * hops_range)  # Don't go below 0
        y_max = global_max_hops + 0.05 * hops_range
    else:
        y_min, y_max = None, None
    
    # Second pass: plot with aligned y-axis
    for i, (num_servers, beamwidths_data) in enumerate(sorted_configs):
        ax = axes_flat[i]
        
        # Sort by beamwidth
        sorted_beamwidths = sorted(beamwidths_data.items())
        
        for beamwidth, data_points in sorted_beamwidths:
            # Extract recall (x) and mean inter-partition hops (y) values
            x_values = [point[2] for point in data_points if len(point) >= 4]
            y_values = [point[3] for point in data_points if len(point) >= 4]
            
            if x_values:
                # Sort by recall
                sorted_points = sorted(zip(x_values, y_values))
                x_values_sorted, y_values_sorted = zip(*sorted_points)
                
                # Get consistent color for this beamwidth
                color = BEAMWIDTH_COLORS.get(beamwidth, None)
                
                # Plot
                ax.plot(x_values_sorted, y_values_sorted, 
                       marker='o', linestyle='-', 
                       linewidth=2, markersize=5, color=color,
                       label=f'BW={beamwidth}')
            else:
                print(f"No valid points for beamwidth {beamwidth} with {num_servers} servers")
        
        # Create title with dataset info and number of servers
        if dataset_info['name'] and dataset_info['size']:
            title = f"{dataset_info['name']} ({dataset_info['size']}) - {num_servers} Server{'s' if num_servers > 1 else ''}"
        else:
            title = f'{num_servers} Server{"s" if num_servers > 1 else ""}'
        
        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.set_xlabel('Recall@10', fontsize=11)
        ax.set_xlim(min_recall, 1.01)
        
        # Only show y-axis label and ticks for the first subplot
        if i == 0:
            ax.set_ylabel('Mean Inter-partition Hops', fontsize=11)
        else:
            ax.set_yticklabels([])
        
        # Set aligned y-axis limits
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot inter-partition hops vs recall for different beamwidths'
    )
    parser.add_argument(
        'logs_folders',
        type=str,
        nargs='+',
        help='Path(s) to the root folder(s) containing log subfolders.'
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
        default='inter_partition_hops_comparison.png',
        help='Output filename for the plot (default: inter_partition_hops_comparison.png)'
    )
    
    args = parser.parse_args()
    
    print(f"Collecting data from {len(args.logs_folders)} folder(s)")
    data, dataset_info = collect_data(args.logs_folders)
    
    if not data:
        print("No data collected. Please check your log folder structure.")
        return
    
    print(f"\nFound data for {len(data)} server configuration(s)")
    print(f"Dataset: {dataset_info['name']} ({dataset_info['size']})")
    for num_servers in sorted(data.keys()):
        beamwidths = sorted(data[num_servers].keys())
        print(f"  {num_servers} server(s): beamwidths={beamwidths}")
    
    print(f"\nGenerating plot...")
    fig = plot_inter_partition_hops(data, dataset_info, args.min_recall)
    
    if fig:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {args.output}")
    else:
        print("Failed to generate plot.")

if __name__ == '__main__':
    main()
