#!/usr/bin/env python3
"""
Script to plot total hops and inter-partition hops comparison across different server configurations.
All data is shown in a single plot with different colors for each server configuration.
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

# Use matplotlib's default color cycle
def get_default_colors():
    """Get matplotlib's default color cycle."""
    prop_cycle = plt.rcParams['axes.prop_cycle']
    return prop_cycle.by_key()['color']

# Define marker styles for different server configurations
def get_marker_styles():
    """Get a list of distinct marker styles with fillstyles."""
    # Returns list of tuples: (marker, fillstyle)
    return [
        ('+', 'full'),      # Plus
        ('x', 'full'),      # X
        ('o', 'none'),      # Hollow circle
        ('s', 'none'),      # Hollow square
    ]


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


def plot_hops_comparison_single(data, dataset_info, min_recall, beamwidth_filter):
    """
    Plot both total hops and inter-partition hops vs recall for different server configurations
    in a single plot.
    
    data: {num_servers: {beamwidth: [(qps, latency, recall, mean_hops, mean_inter_hops), ...]}}
    dataset_info: {'name': str, 'size': str}
    beamwidth_filter: Only plot data for this beamwidth (or None for all)
    """
    if not data:
        print("No data to plot!")
        return None
    
    # Print percentage of inter-partition hops at recall >= 0.95
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
    
    # Create a single plot
    fig, ax = plt.subplots(figsize=(3, 3))
    
    # Track what we've plotted for custom legend
    plotted_servers = set()
    has_total = False
    has_inter = False
    
    # Get default matplotlib colors
    default_colors = get_default_colors()
    
    # Get marker styles
    marker_styles = get_marker_styles()
    
    # Sort by number of servers
    sorted_configs = sorted(data.items())
    
    # Create a mapping from num_servers to color index and marker index
    server_to_color_idx = {num_servers: idx for idx, (num_servers, _) in enumerate(sorted_configs)}
    server_to_marker_idx = {num_servers: idx for idx, (num_servers, _) in enumerate(sorted_configs)}
    
    for num_servers, beamwidths_data in sorted_configs:
        # If beamwidth_filter is specified, only plot that beamwidth
        if beamwidth_filter is not None:
            if beamwidth_filter not in beamwidths_data:
                print(f"Warning: Beamwidth {beamwidth_filter} not found for {num_servers} server(s)")
                continue
            beamwidths_to_plot = {beamwidth_filter: beamwidths_data[beamwidth_filter]}
        else:
            # Plot all beamwidths (will combine/overlay them)
            beamwidths_to_plot = beamwidths_data
        
        # Get consistent color and marker for this server configuration
        color_idx = server_to_color_idx[num_servers]
        color = default_colors[color_idx % len(default_colors)]
        
        marker_idx = server_to_marker_idx[num_servers]
        marker, fillstyle = marker_styles[marker_idx % len(marker_styles)]
        
        # Collect all data points for this server configuration across all beamwidths
        all_recalls = []
        all_total_hops = []
        all_inter_hops = []
        
        for beamwidth, data_points in beamwidths_to_plot.items():
            # Extract recall, total hops, and inter-partition hops
            recalls = [point[2] for point in data_points if len(point) >= 5]
            total_hops = [point[3] for point in data_points if len(point) >= 5]
            inter_hops = [point[4] for point in data_points if len(point) >= 5]
            
            all_recalls.extend(recalls)
            all_total_hops.extend(total_hops)
            all_inter_hops.extend(inter_hops)
        
        if all_recalls:
            # Sort total hops data by recall
            sorted_total = sorted(zip(all_recalls, all_total_hops))
            recalls_sorted_total, total_hops_sorted = zip(*sorted_total)
            
            # Sort inter-partition hops data by recall
            sorted_inter = sorted(zip(all_recalls, all_inter_hops))
            recalls_sorted_inter, inter_hops_sorted = zip(*sorted_inter)
            
            # Plot total hops (solid line)
            ax.plot(recalls_sorted_total, total_hops_sorted, 
                   marker=marker, fillstyle=fillstyle, linestyle='-', 
                    linewidth=1.25, markersize=5, color=color, alpha=1)
            
            # Plot inter-partition hops (dashed line, same color)
            ax.plot(recalls_sorted_inter, inter_hops_sorted, 
                   marker=marker, fillstyle=fillstyle, linestyle='--', 
                    linewidth=1.25, markersize=5, color=color, alpha=1)
            
            plotted_servers.add(num_servers)
            has_total = True
            has_inter = True
        else:
            print(f"No valid points for {num_servers} server(s)")
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = []
    
    # Section 1: Server configurations (colors and markers)
    if plotted_servers:
        for num_servers in sorted(plotted_servers):
            color_idx = server_to_color_idx[num_servers]
            color = default_colors[color_idx % len(default_colors)]
            marker_idx = server_to_marker_idx[num_servers]
            marker, fillstyle = marker_styles[marker_idx % len(marker_styles)]
            server_label = f'{num_servers} Server{"s" if num_servers > 1 else ""}'
            legend_elements.append(Line2D([0], [0], color=color, marker=marker, 
                                         fillstyle=fillstyle, linewidth=2, markersize=8,
                                         label=server_label))
    
    # Add a separator (invisible line)
    if plotted_servers and (has_total or has_inter):
        legend_elements.append(Line2D([0], [0], color='none', label=''))
    
    # Section 2: Hop types (line styles)
    if has_total:
        legend_elements.append(Line2D([0], [0], color='black', linewidth=2, 
                                      linestyle='-', label='Total Hops'))
    if has_inter:
        legend_elements.append(Line2D([0], [0], color='black', linewidth=2, 
                                      linestyle='--', label='Inter-partition Hops'))
    
    # Create title with dataset info
    if dataset_info['name'] and dataset_info['size']:
        title = f"{dataset_info['name']} ({dataset_info['size']}) - Hops vs Recall"
        if beamwidth_filter is not None:
            title += f" (Beamwidth={beamwidth_filter})"
    else:
        title = 'Hops vs Recall Comparison'
        if beamwidth_filter is not None:
            title += f" (Beamwidth={beamwidth_filter})"
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Recall@10', fontsize=11)
    ax.set_ylabel('Mean Hops', fontsize=11)
    ax.set_xlim(min_recall, 1.01)
    ax.grid(True, alpha=0.3)
    
    if legend_elements:
        ax.legend(handles=legend_elements, fontsize=9, loc='best')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot total hops and inter-partition hops vs recall for different server configurations in a single plot'
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
        '--beamwidth',
        type=int,
        default=None,
        help='Only plot data for this specific beamwidth (default: plot all beamwidths)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='hops_comparison_single.png',
        help='Output filename for the plot (default: hops_comparison_single.png)'
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
    
    if args.beamwidth is not None:
        print(f"\nFiltering for beamwidth={args.beamwidth}")
    
    print(f"\nGenerating plot...")
    fig = plot_hops_comparison_single(data, dataset_info, args.min_recall, args.beamwidth)
    
    if fig:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {args.output}")
    else:
        print("Failed to generate plot.")

if __name__ == '__main__':
    main()
