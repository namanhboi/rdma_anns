#!/usr/bin/env python3
"""
Script to plot distance comparisons and disk I/O comparison across different beamwidths.
All data is shown in a single plot with different colors for each beamwidth.
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

# Define marker styles for different beamwidths
def get_marker_styles():
    """Get a list of distinct marker styles."""
    return ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'P', 'X']


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
    Parse client.log file and extract QPS, Recall, Distance Comparisons, and Mean I/O data.
    
    Returns a list of tuples: (qps, latency, recall, distance_comparisons, mean_io)
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
                # Extract QPS (column 2), Mean I/O (parts[-3]), Distance Comparisons (parts[-2]), Recall (parts[-1])
                qps = float(parts[2])
                avg_latency = float(parts[3])
                mean_io = float(parts[-3])
                distance_comparisons = float(parts[-2])
                recall = float(parts[-1])
                
                data_points.append((qps, avg_latency, recall / 100.0, distance_comparisons, mean_io))
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line '{line}': {e}")
                continue
    
    except FileNotFoundError:
        print(f"Warning: File not found: {log_file_path}")
    except Exception as e:
        print(f"Error parsing {log_file_path}: {e}")
    
    return data_points


def collect_data(logs_folders, num_servers_filter):
    """
    Collect all data from log folders, organized by beamwidth.
    
    Args:
        logs_folders: List of paths to root folders containing log subfolders
        num_servers_filter: Only collect data for this number of servers (or None for all)
    
    Returns a tuple: (data_dict, dataset_info)
    where data_dict is {beamwidth: {num_servers: [(qps, latency, recall, distance_comparisons, mean_io), ...]}}
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
            
            # Filter by number of servers if specified
            if num_servers_filter is not None and metadata['num_servers'] != num_servers_filter:
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
            
            # Store data with beamwidth as primary key
            beamwidth = metadata['beamwidth']
            num_servers = metadata['num_servers']
            method = metadata['method']
            
            data[beamwidth][num_servers] = data_points
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


def plot_both_metrics(data, dataset_info, min_recall, num_servers_filter):
    """
    Plot both distance comparisons and disk I/O vs recall for different beamwidths in a single figure.
    
    data: {beamwidth: {num_servers: [(qps, latency, recall, distance_comparisons, mean_io), ...]}}
    dataset_info: {'name': str, 'size': str}
    num_servers_filter: Number of servers being compared (for title)
    """
    if not data:
        print("No data to plot!")
        return None
    
    # Print values at recall >= 0.95 for both metrics
    print("\n" + "="*80)
    print("Distance Comparisons and I/O at Recall >= 0.95")
    print("="*80)
    
    for beamwidth in sorted(data.keys()):
        print(f"\W {beamwidth}:")
        print(f"{'Servers':<12} {'Distance Comp':<20} {'Mean I/O':<20}")
        print("-" * 52)
        
        for num_servers in sorted(data[beamwidth].keys()):
            data_points = data[beamwidth][num_servers]
            
            # Extract recall and metric values
            recalls = [point[2] for point in data_points if len(point) >= 5]
            distance_values = [point[3] for point in data_points if len(point) >= 5]
            io_values = [point[4] for point in data_points if len(point) >= 5]
            
            if recalls:
                # Sort by recall
                sorted_dist = sorted(zip(recalls, distance_values))
                recalls_sorted_dist, distance_sorted = zip(*sorted_dist)
                
                sorted_io = sorted(zip(recalls, io_values))
                recalls_sorted_io, io_sorted = zip(*sorted_io)
                
                # Interpolate at 0.95
                dist_at_95 = interpolate_at_recall(recalls_sorted_dist, distance_sorted, 0.95)
                io_at_95 = interpolate_at_recall(recalls_sorted_io, io_sorted, 0.95)
                
                dist_str = f"{dist_at_95:.4f}" if dist_at_95 is not None else "N/A"
                io_str = f"{io_at_95:.4f}" if io_at_95 is not None else "N/A"
                
                print(f"{num_servers:<12} {dist_str:<20} {io_str:<20}")
            else:
                print(f"{num_servers:<12} {'N/A':<20} {'N/A':<20}")
    
    print("\n" + "="*80 + "\n")
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Get default matplotlib colors
    default_colors = get_default_colors()
    
    # Get marker styles
    marker_styles = get_marker_styles()
    
    # Sort by beamwidth
    sorted_configs = sorted(data.items())
    
    # Create a mapping from beamwidth to color index and marker index
    # Manually map beamwidths to specific colors: bw1=blue(0), bw8=red(3)
    # For other beamwidths, use sequential colors
    def get_color_idx_for_beamwidth(bw):
        if bw == 1:
            return 0  # Blue (first color in default cycle)
        elif bw == 8:
            return 3  # Red (fourth color in default cycle)
        else:
            # For other beamwidths, use sequential indices, skipping 0 and 3
            all_beamwidths = sorted([b for b, _ in sorted_configs])
            other_bws = [b for b in all_beamwidths if b not in [1, 8]]
            if bw in other_bws:
                idx = other_bws.index(bw)
                # Skip indices 0 and 3 (used by bw1 and bw8)
                available_indices = [i for i in range(10) if i not in [0, 3]]
                return available_indices[idx % len(available_indices)]
            return 0
    
    beamwidth_to_color_idx = {beamwidth: get_color_idx_for_beamwidth(beamwidth) for beamwidth, _ in sorted_configs}
    beamwidth_to_marker_idx = {beamwidth: idx for idx, (beamwidth, _) in enumerate(sorted_configs)}
    
    plotted_beamwidths = set()
    
    # Plot both metrics
    for beamwidth, servers_data in sorted_configs:
        # Get consistent color and marker for this beamwidth
        color_idx = beamwidth_to_color_idx[beamwidth]
        color = default_colors[color_idx % len(default_colors)]
        
        marker_idx = beamwidth_to_marker_idx[beamwidth]
        marker = marker_styles[marker_idx % len(marker_styles)]
        
        # Collect all data points for this beamwidth across all server configs
        all_recalls = []
        all_distance_values = []
        all_io_values = []
        
        for num_servers, data_points in servers_data.items():
            # Extract recall and metric values
            recalls = [point[2] for point in data_points if len(point) >= 5]
            distance_values = [point[3] for point in data_points if len(point) >= 5]
            io_values = [point[4] for point in data_points if len(point) >= 5]
            
            all_recalls.extend(recalls)
            all_distance_values.extend(distance_values)
            all_io_values.extend(io_values)
        
        if all_recalls:
            # Sort data by recall for distance comparisons
            sorted_dist = sorted(zip(all_recalls, all_distance_values))
            recalls_sorted_dist, distance_sorted = zip(*sorted_dist)
            
            # Sort data by recall for I/O
            sorted_io = sorted(zip(all_recalls, all_io_values))
            recalls_sorted_io, io_sorted = zip(*sorted_io)
            
            # Plot distance comparisons on left subplot
            beamwidth_label = f'W {beamwidth}'
            ax1.plot(recalls_sorted_dist, distance_sorted, 
                    marker=marker, linestyle='-', 
                    linewidth=1.25, markersize=4, color=color, alpha=0.9,
                    label=beamwidth_label)
            
            # Plot I/O on right subplot
            ax2.plot(recalls_sorted_io, io_sorted, 
                    marker=marker, linestyle='-', 
                    linewidth=1.25, markersize=4, color=color, alpha=0.9,
                    label=beamwidth_label)
            
            plotted_beamwidths.add(beamwidth)
        else:
            print(f"No valid points for beamwidth {beamwidth}")
    
    # Configure left subplot (Distance Comparisons)
    ax1.set_title('Distance Comparisons vs Recall', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Recall@10', fontsize=10)
    ax1.set_ylabel('Distance Comparisons', fontsize=10)
    ax1.set_xlim(min_recall, 1.01)
    ax1.grid(True, alpha=0.3)
    if plotted_beamwidths:
        ax1.legend(fontsize=8, loc='best')
    
    # Configure right subplot (I/O)
    ax2.set_title('Mean I/O vs Recall', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Recall@10', fontsize=10)
    ax2.set_ylabel('Mean I/O Operations', fontsize=10)
    ax2.set_xlim(min_recall, 1.01)
    ax2.grid(True, alpha=0.3)
    if plotted_beamwidths:
        ax2.legend(fontsize=8, loc='best')
    
    # Create overall title with dataset info
    title_parts = []
    if dataset_info['name'] and dataset_info['size']:
        title_parts.append(f"{dataset_info['name']} ({dataset_info['size']})")
    
    if num_servers_filter is not None:
        title_parts.append(f"{num_servers_filter} Server{'s' if num_servers_filter > 1 else ''}")
    
    # if title_parts:
        # fig.suptitle(' - '.join(title_parts), fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot distance comparisons and disk I/O vs recall for different beamwidths in a single figure'
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
        '--num-servers',
        type=int,
        default=None,
        help='Only plot data for this specific number of servers (default: combine all server configs)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='distance_io_by_beamwidth.png',
        help='Output filename for the combined plot (default: distance_io_by_beamwidth.png)'
    )
    
    args = parser.parse_args()
    
    print(f"Collecting data from {len(args.logs_folders)} folder(s)")
    if args.num_servers is not None:
        print(f"Filtering for {args.num_servers} server(s)")
    
    data, dataset_info = collect_data(args.logs_folders, args.num_servers)
    
    if not data:
        print("No data collected. Please check your log folder structure.")
        return
    
    print(f"\nFound data for {len(data)} beamwidth(s)")
    print(f"Dataset: {dataset_info['name']} ({dataset_info['size']})")
    for beamwidth in sorted(data.keys()):
        server_configs = sorted(data[beamwidth].keys())
        print(f"  Beamwidth {beamwidth}: server configs={server_configs}")
    
    # Generate combined plot
    print(f"\nGenerating combined plot...")
    fig = plot_both_metrics(data, dataset_info, args.min_recall, args.num_servers)
    
    if fig:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {args.output}")
    else:
        print("Failed to generate plot.")

if __name__ == '__main__':
    main()
