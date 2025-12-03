#!/usr/bin/env python3
"""
Script to plot total hops and inter-partition hops comparison across different beamwidths.
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
    Parse client.log file and extract QPS, Recall, Mean Hops, and Mean Inter-partition Hops data.
    
    Returns a list of tuples: (qps, latency, recall, mean_hops, mean_inter_partition_hops)
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
                # Extract QPS (column 2), Mean Hops (column 5), Mean Inter-partition Hops (column 6), Recall (last column)
                qps = float(parts[2])
                avg_latency = float(parts[3])
                mean_hops = float(parts[5])
                mean_inter_partition_hops = float(parts[6])
                recall = float(parts[-1])
                
                data_points.append((qps, avg_latency, recall / 100.0, mean_hops, mean_inter_partition_hops))
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
    where data_dict is {num_servers: {beamwidth: [(qps, latency, recall, mean_hops, mean_inter_hops), ...]}}
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


def interpolate_at_recall(x_values, y_values, target_recall):
    """
    Interpolate to find the value at a specific recall.
    
    Args:
        x_values: list of recall values (sorted)
        y_values: list of values (corresponding to recalls)
        target_recall: the recall value to interpolate at
    
    Returns:
        Interpolated value or None if target_recall is out of range
    """
    if not x_values or not y_values:
        return None
    
    # Check if target is in range
    if target_recall < min(x_values) or target_recall > max(x_values):
        return None
    
    # Use numpy interpolation
    return np.interp(target_recall, x_values, y_values)


def plot_hops_comparison(data, dataset_info, min_recall):
    """
    Plot both total hops and inter-partition hops vs recall for different beamwidths.
    
    data: {num_servers: {beamwidth: [(qps, latency, recall, mean_hops, mean_inter_hops), ...]}}
    dataset_info: {'name': str, 'size': str}
    """
    num_configs = len(data)
    if num_configs == 0:
        print("No data to plot!")
        return None
    
    # Print percentage of inter-partition hops at recall >= 0.9
    print("\n" + "="*80)
    print("Percentage of Inter-partition Hops at Recall >= 0.95")
    print("="*80)
    
    for num_servers in sorted(data.keys()):
        print(f"\n{num_servers} Server{'s' if num_servers > 1 else ''}:")
        print(f"{'Beamwidth':<12} {'Total Hops':<15} {'Inter-part Hops':<15} {'% Inter-part':<15}")
        print("-" * 57)
        
        for beamwidth in sorted(data[num_servers].keys()):
            data_points = data[num_servers][beamwidth]
            
            # Extract recall, total hops, and inter-partition hops
            recalls = [point[2] for point in data_points if len(point) >= 5]
            total_hops = [point[3] for point in data_points if len(point) >= 5]
            inter_hops = [point[4] for point in data_points if len(point) >= 5]
            
            if recalls:
                # Sort by recall
                sorted_data = sorted(zip(recalls, total_hops, inter_hops))
                recalls_sorted, total_hops_sorted, inter_hops_sorted = zip(*sorted_data)
                
                # Interpolate at 0.95
                total_at_95 = interpolate_at_recall(recalls_sorted, total_hops_sorted, 0.95)
                inter_at_95 = interpolate_at_recall(recalls_sorted, inter_hops_sorted, 0.95)
                
                if total_at_95 is not None and inter_at_95 is not None and total_at_95 > 0:
                    percentage = (inter_at_95 / total_at_95) * 100
                    print(f"{beamwidth:<12} {total_at_95:<15.4f} {inter_at_95:<15.4f} {percentage:<15.2f}%")
                else:
                    print(f"{beamwidth:<12} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    
    print("\n" + "="*80 + "\n")
    
    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=num_configs, 
                            figsize=(5 * num_configs, 4), squeeze=False)
    axes_flat = axes.flatten()
    
    # Sort by number of servers
    sorted_configs = sorted(data.items())
    
    # First pass: find global min/max hop values for aligned y-axis
    all_hop_values = []
    for num_servers, beamwidths_data in sorted_configs:
        for beamwidth, data_points in beamwidths_data.items():
            total_hops_vals = [point[3] for point in data_points if len(point) >= 5]
            inter_hops_vals = [point[4] for point in data_points if len(point) >= 5]
            all_hop_values.extend(total_hops_vals)
            all_hop_values.extend(inter_hops_vals)
    
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
    # Collect legend elements from first subplot only
    legend_elements = []
    
    for i, (num_servers, beamwidths_data) in enumerate(sorted_configs):
        ax = axes_flat[i]
        
        # Track what we've plotted for custom legend
        plotted_beamwidths = set()
        has_total = False
        has_inter = False
        
        # Sort by beamwidth
        sorted_beamwidths = sorted(beamwidths_data.items())
        
        for beamwidth, data_points in sorted_beamwidths:
            # Extract recall, total hops, and inter-partition hops
            recalls = [point[2] for point in data_points if len(point) >= 5]
            total_hops = [point[3] for point in data_points if len(point) >= 5]
            inter_hops = [point[4] for point in data_points if len(point) >= 5]
            
            if recalls:
                # Sort by recall
                sorted_data = sorted(zip(recalls, total_hops, inter_hops))
                recalls_sorted, total_hops_sorted, inter_hops_sorted = zip(*sorted_data)
                
                # Get consistent color for this beamwidth
                color = BEAMWIDTH_COLORS.get(beamwidth, None)
                
                # Plot total hops (solid line) - no label in plot
                ax.plot(recalls_sorted, total_hops_sorted, 
                       marker='+', linestyle='-', 
                        linewidth=1.25, markersize=6.5, color=color)
                
                # Plot inter-partition hops (dashed line, same color) - no label in plot
                ax.plot(recalls_sorted, inter_hops_sorted, 
                       marker='x', linestyle='--', 
                        linewidth=1.25, markersize=6.5, color=color)
                
                plotted_beamwidths.add(beamwidth)
                has_total = True
                has_inter = True
            else:
                print(f"No valid points for beamwidth {beamwidth} with {num_servers} servers")
        
        # Create custom legend elements only for the first subplot
        if i == 0:
            from matplotlib.lines import Line2D
            
            # Section 1: Beamwidth (colors)
            if plotted_beamwidths:
                for bw in sorted(plotted_beamwidths):
                    color = BEAMWIDTH_COLORS.get(bw)
                    legend_elements.append(Line2D([0], [0], color=color, linewidth=2, 
                                                 label=f'BW={bw}'))
            
            # Add a separator (invisible line)
            if plotted_beamwidths and (has_total or has_inter):
                legend_elements.append(Line2D([0], [0], color='none', label=''))
            
            # Section 2: Hop types (line styles)
            if has_total:
                legend_elements.append(Line2D([0], [0], color='black', linewidth=2, 
                                              linestyle='-', label='+ Hops'))
            if has_inter:
                legend_elements.append(Line2D([0], [0], color='black', linewidth=2, 
                                              linestyle='--', label='x Inter-partition Hops'))
        
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
            ax.set_ylabel('Mean Hops', fontsize=11)
        else:
            ax.set_yticklabels([])
        
        # Set aligned y-axis limits
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        
        ax.grid(True, alpha=0.3)
        
        # Only show legend in the first subplot
        if i == 0 and legend_elements:
            ax.legend(handles=legend_elements, fontsize=9, loc='best')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot total hops and inter-partition hops vs recall for different beamwidths'
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
        default='hops_comparison.png',
        help='Output filename for the plot (default: hops_comparison.png)'
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
    fig = plot_hops_comparison(data, dataset_info, args.min_recall)
    
    if fig:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {args.output}")
    else:
        print("Failed to generate plot.")

if __name__ == '__main__':
    main()
