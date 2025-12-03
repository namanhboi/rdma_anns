#!/usr/bin/env python3
"""
Script to plot Mean IOs vs Recall and Mean Inter vs Recall curves comparing different datasets and server configurations.
Creates two plots per dataset (IOs and Inter) with all server configs combined.
Line styles: 5 servers = solid, 10 servers = dashed, SINGLE_SERVER = plotted once
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Define consistent colors for different methods
METHOD_COLORS = {
    'STATE_SEND': '#1f77b4',      # Blue
    'SCATTER_GATHER': '#ff7f0e',  # Orange
    'SINGLE_SERVER': '#2ca02c',   # Green
}

# Define markers for different methods
METHOD_MARKERS = {
    'STATE_SEND': 'x',           # X marker
    'SCATTER_GATHER': '.',       # Point marker
    'SINGLE_SERVER': '+',        # Plus marker
}

# Define line styles for different server counts
SERVER_LINE_STYLES = {
    5: '-',      # Solid for 5 servers
    10: '--',    # Dashed for 10 servers
}


def parse_folder_name(folder_name):
    """
    Parse the log folder name to extract metadata.
    Expected formats (all use distributed format now):
    - logs_STATE_SEND_distributed_${DATASET_NAME}_${DATASET_SIZE}_${NUM_SERVERS}_${TIMEOUT}_MS_...BEAMWIDTH_${BEAM_WIDTH}_timestamp
    - logs_SCATTER_GATHER_distributed_${DATASET_NAME}_${DATASET_SIZE}_${NUM_SERVERS}_${TIMEOUT}_MS_...BEAMWIDTH_${BEAM_WIDTH}_timestamp
    - logs_SINGLE_SERVER_distributed_${DATASET_NAME}_${DATASET_SIZE}_1_${TIMEOUT}_MS_...BEAMWIDTH_${BEAM_WIDTH}_timestamp
    
    Note: SINGLE_SERVER always has num_servers=1, but will be added to all plots since distance comparisons are the same.
    
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
    # All methods now use distributed_DATASET_SIZE format
    dataset_match = re.search(r'distributed_(\w+)_(\d+[BKMG])', folder_name)
    if not dataset_match:
        return None
    dataset_name = dataset_match.group(1)
    dataset_size = dataset_match.group(2)
    
    # Extract number of servers - look for pattern after dataset size
    # Pattern: ...100M_NUM_SERVERS_... or ...1B_NUM_SERVERS_...
    num_servers_match = re.search(r'_(\d+[BKMG])_(\d+)_\d+_MS', folder_name)
    if not num_servers_match:
        return None
    num_servers = int(num_servers_match.group(2))
    
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
    Parse client.log file and extract Mean Hops, Mean Inter, Mean IOs, and Recall data.
    
    Returns a tuple of three lists: (ios_data_points, hops_data_points, inter_data_points)
    where each list contains tuples of (value, recall)
    """
    ios_data_points = []
    hops_data_points = []
    inter_data_points = []
    
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
        
        # Find the header line
        header_found = False
        data_start_idx = 0
        
        for i, line in enumerate(lines):
            if 'L   I/O Width' in line and 'Mean cmps' in line and 'Recall' in line:
                header_found = True
                data_start_idx = i + 2  # Skip header and separator line
                break
        
        if not header_found:
            print(f"Warning: Header not found in {log_file_path}")
            return ios_data_points, hops_data_points, inter_data_points
        
        # Parse data lines
        for line in lines[data_start_idx:]:
            line = line.strip()
            if not line:
                continue
            
            # Split by whitespace and extract relevant columns
            parts = line.split()
            if len(parts) < 9:
                continue
            
            try:
                # Extract Mean Hops (column 5), Mean Inter (column 6), Mean IOs, and Recall
                mean_hops = float(parts[5])
                mean_inter = float(parts[6])
                mean_ios = float(parts[-3])
                recall = float(parts[-1])
                
                ios_data_points.append((mean_ios, recall / 100.0))  # Convert recall to 0-1 range
                hops_data_points.append((mean_hops, recall / 100.0))
                inter_data_points.append((mean_inter, recall / 100.0))
            except (ValueError, IndexError):
                continue
    
    except FileNotFoundError:
        print(f"Warning: File not found: {log_file_path}")
    except Exception as e:
        print(f"Error parsing {log_file_path}: {e}")
    
    return ios_data_points, hops_data_points, inter_data_points


def collect_data(logs_folder):
    """
    Collect all data from log folder.
    
    Args:
        logs_folder: Path to root folder containing log subfolders
    
    Returns a tuple: (ios_data, hops_data, inter_data, single_server_ios_data, single_server_hops_data, single_server_inter_data)
    - ios_data: {dataset_name: {num_servers: {(method, beamwidth): [(mean_ios, recall), ...]}}}
    - hops_data: {dataset_name: {num_servers: {(method, beamwidth): [(mean_hops, recall), ...]}}}
    - inter_data: {dataset_name: {num_servers: {(method, beamwidth): [(mean_inter, recall), ...]}}}
    - single_server_ios_data: {dataset_name: {beamwidth: [(mean_ios, recall), ...]}}
    - single_server_hops_data: {dataset_name: {beamwidth: [(mean_hops, recall), ...]}}
    - single_server_inter_data: {dataset_name: {beamwidth: [(mean_inter, recall), ...]}}
    """
    ios_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    hops_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    inter_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    single_server_ios_data = defaultdict(lambda: defaultdict(list))
    single_server_hops_data = defaultdict(lambda: defaultdict(list))
    single_server_inter_data = defaultdict(lambda: defaultdict(list))
    
    logs_root = Path(logs_folder)
    
    if not logs_root.exists():
        print(f"Error: Root folder '{logs_folder}' does not exist")
        return ios_data, hops_data, inter_data, single_server_ios_data, single_server_hops_data, single_server_inter_data
    
    print(f"Scanning folder: {logs_folder}")
    
    # Iterate through all subdirectories (including symlinks)
    for folder in logs_root.iterdir():
        if not folder.is_dir():
            continue
        
        # Parse folder name
        metadata = parse_folder_name(folder.name)
        if metadata is None:
            print(f"  Skipping folder (couldn't parse): {folder.name}")
            continue
        
        # Look for client.log file
        client_log = folder / 'client.log'
        if not client_log.exists():
            print(f"  Warning: client.log not found in {folder.name}")
            continue
        
        # Parse the log file
        ios_data_points, hops_data_points, inter_data_points = parse_client_log(client_log)
        if not ios_data_points or not hops_data_points or not inter_data_points:
            print(f"  Warning: No data extracted from {client_log}")
            continue
        
        # Store data with dataset -> num_servers -> (method, beamwidth) hierarchy
        dataset_name = metadata['dataset_name']
        num_servers = metadata['num_servers']
        method = metadata['method']
        beamwidth = metadata['beamwidth']
        
        # Store SINGLE_SERVER separately (will be plotted once per dataset)
        if method == 'SINGLE_SERVER':
            single_server_ios_data[dataset_name][beamwidth] = ios_data_points
            single_server_hops_data[dataset_name][beamwidth] = hops_data_points
            single_server_inter_data[dataset_name][beamwidth] = inter_data_points
            print(f"  Loaded {len(ios_data_points)} data points from {folder.name} "
                  f"(dataset={dataset_name}, method={method}, beamwidth={beamwidth}) - will plot once")
        else:
            ios_data[dataset_name][num_servers][(method, beamwidth)] = ios_data_points
            hops_data[dataset_name][num_servers][(method, beamwidth)] = hops_data_points
            inter_data[dataset_name][num_servers][(method, beamwidth)] = inter_data_points
            print(f"  Loaded {len(ios_data_points)} data points from {folder.name} "
                  f"(dataset={dataset_name}, method={method}, beamwidth={beamwidth}, servers={num_servers})")
    
    print(f"\n=== Data Collection Summary ===")
    print(f"Distributed data: {len(ios_data)} datasets")
    for dataset_name in ios_data:
        print(f"  {dataset_name}: {list(ios_data[dataset_name].keys())} server configs")
    print(f"Single server data: {len(single_server_ios_data)} datasets")
    for dataset_name in single_server_ios_data:
        print(f"  {dataset_name}: beamwidths {list(single_server_ios_data[dataset_name].keys())}")
    
    return ios_data, hops_data, inter_data, single_server_ios_data, single_server_hops_data, single_server_inter_data


def interpolate_at_recall(data_points, target_recall):
    """
    Interpolate to find mean_ios at a specific recall value.
    
    Args:
        data_points: List of (mean_ios, recall) tuples
        target_recall: Target recall value (e.g., 0.9, 0.95)
    
    Returns:
        Interpolated mean_ios value or None if target_recall is out of range
    """
    if not data_points:
        return None
    
    # Extract and sort by recall
    sorted_points = sorted(data_points, key=lambda x: x[1])
    recalls = [p[1] for p in sorted_points]
    ios = [p[0] for p in sorted_points]
    
    # Check if target_recall is within range
    if target_recall < recalls[0] or target_recall > recalls[-1]:
        return None
    
    # Use numpy's interp function (linear interpolation)
    return np.interp(target_recall, recalls, ios)


def print_recall_comparisons(ios_data, hops_data, inter_data, single_server_ios_data, single_server_hops_data, single_server_inter_data, target_recalls=[0.9, 0.95]):
    """
    Print IOs, Hops, and Inter at specific recall values for all methods.
    
    Args:
        ios_data: {dataset_name: {num_servers: {(method, beamwidth): [(mean_ios, recall), ...]}}}
        hops_data: {dataset_name: {num_servers: {(method, beamwidth): [(mean_hops, recall), ...]}}}
        inter_data: {dataset_name: {num_servers: {(method, beamwidth): [(mean_inter, recall), ...]}}}
        single_server_ios_data: {dataset_name: {beamwidth: [(mean_ios, recall), ...]}}
        single_server_hops_data: {dataset_name: {beamwidth: [(mean_hops, recall), ...]}}
        single_server_inter_data: {dataset_name: {beamwidth: [(mean_inter, recall), ...]}}
        target_recalls: List of recall values to interpolate at
    """
    print("\n" + "="*80)
    print("MEAN IOs, HOPS, AND INTER AT SPECIFIC RECALL VALUES")
    print("="*80)
    
    # Get sorted lists of datasets and server configs
    datasets = sorted(ios_data.keys())
    
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # Get all server configs for this dataset
        server_configs = sorted(ios_data[dataset_name].keys())
        
        for num_servers in server_configs:
            print(f"\n--- {num_servers} Server{'s' if num_servers > 1 else ''} ---")
            
            # Combine distributed and single server data for this plot
            ios_methods_data = ios_data[dataset_name][num_servers].copy()
            hops_methods_data = hops_data[dataset_name][num_servers].copy()
            inter_methods_data = inter_data[dataset_name][num_servers].copy()
            
            # Add SINGLE_SERVER data
            if dataset_name in single_server_ios_data:
                for beamwidth, data_points in single_server_ios_data[dataset_name].items():
                    ios_methods_data[('SINGLE_SERVER', beamwidth)] = data_points
                for beamwidth, data_points in single_server_hops_data[dataset_name].items():
                    hops_methods_data[('SINGLE_SERVER', beamwidth)] = data_points
                for beamwidth, data_points in single_server_inter_data[dataset_name].items():
                    inter_methods_data[('SINGLE_SERVER', beamwidth)] = data_points
            
            # Group by method for cleaner output
            method_results = defaultdict(dict)
            
            for (method, beamwidth), ios_data_points in ios_methods_data.items():
                hops_data_points = hops_methods_data[(method, beamwidth)]
                inter_data_points = inter_methods_data[(method, beamwidth)]
                for target_recall in target_recalls:
                    ios_value = interpolate_at_recall(ios_data_points, target_recall)
                    hops_value = interpolate_at_recall(hops_data_points, target_recall)
                    inter_value = interpolate_at_recall(inter_data_points, target_recall)
                    if ios_value is not None and hops_value is not None and inter_value is not None:
                        if beamwidth not in method_results[method]:
                            method_results[method][beamwidth] = {}
                        method_results[method][beamwidth][target_recall] = (ios_value, hops_value, inter_value)
            
            # Print results organized by method
            method_order = ['STATE_SEND', 'SINGLE_SERVER']
            for method in method_order:
                if method not in method_results:
                    continue
                
                print(f"\n  {method}:")
                for beamwidth in sorted(method_results[method].keys()):
                    print(f"    Beamwidth {beamwidth}:")
                    for target_recall in target_recalls:
                        if target_recall in method_results[method][beamwidth]:
                            ios_value, hops_value, inter_value = method_results[method][beamwidth][target_recall]
                            print(f"      Recall@{target_recall:.2f}: {ios_value:,.1f} IOs, {hops_value:,.1f} Hops, {inter_value:,.1f} Inter")
                        else:
                            print(f"      Recall@{target_recall:.2f}: N/A (out of range)")


def plot_comparison_grid(ios_data, hops_data, inter_data, single_server_ios_data, single_server_hops_data, single_server_inter_data, min_recall, dataset_sizes):
    """
    Plot three plots per dataset (IOs, Hops, and Inter) showing all server configs combined.
    
    Args:
        ios_data: {dataset_name: {num_servers: {(method, beamwidth): [(mean_ios, recall), ...]}}}
        hops_data: {dataset_name: {num_servers: {(method, beamwidth): [(mean_hops, recall), ...]}}}
        inter_data: {dataset_name: {num_servers: {(method, beamwidth): [(mean_inter, recall), ...]}}}
        single_server_ios_data: {dataset_name: {beamwidth: [(mean_ios, recall), ...]}} - plotted once per dataset
        single_server_hops_data: {dataset_name: {beamwidth: [(mean_hops, recall), ...]}} - plotted once per dataset
        single_server_inter_data: {dataset_name: {beamwidth: [(mean_inter, recall), ...]}} - plotted once per dataset
        min_recall: Minimum recall value for x-axis
        dataset_sizes: {dataset_name: size_string} mapping
    
    Returns:
        matplotlib figure
    """
    if not ios_data:
        print("No data to plot!")
        return None
    
    # Get sorted list of datasets
    datasets = sorted(ios_data.keys())
    
    if not datasets:
        print("No valid data configuration found!")
        return None
    
    num_datasets = len(datasets)
    
    print(f"\nCreating {num_datasets} row(s) with 2 plots each (IOs and Inter arranged horizontally)")
    print(f"Datasets: {datasets}")
    
    # Create figure with subplots (num_datasets rows, 2 columns)
    # Each dataset gets 1 row with 2 columns: IOs, Inter
    fig, axes = plt.subplots(nrows=num_datasets, ncols=2,
                            figsize=(8, 4 * num_datasets),
                            squeeze=False)
    
    # Process each dataset
    for dataset_idx, dataset_name in enumerate(datasets):
        # Process IOs and Inter for this dataset (skip Hops)
        for metric_idx, (data_dict, single_server_dict, ylabel, plot_type) in enumerate([
            (ios_data, single_server_ios_data, 'Mean Disk IOs', 'Disk IOs'),
            (inter_data, single_server_inter_data, 'Mean Inter-partition Hops', 'Inter-partition Hops')
        ]):
            # Column index is the metric_idx (0=IOs, 1=Inter)
            ax = axes[dataset_idx, metric_idx]
            
            # Find min/max for this metric across all datasets for consistent y-axis
            all_values = []
            for ds_name in datasets:
                for num_servers in data_dict[ds_name].keys():
                    for (method, beamwidth), data_points in data_dict[ds_name][num_servers].items():
                        y_values = [point[0] for point in data_points if len(point) >= 2]
                        all_values.extend(y_values)
                
                # Also include single server data
                if ds_name in single_server_dict:
                    for beamwidth, data_points in single_server_dict[ds_name].items():
                        y_values = [point[0] for point in data_points if len(point) >= 2]
                        all_values.extend(y_values)
            
            if all_values:
                global_min = min(all_values)
                global_max = max(all_values)
                value_range = global_max - global_min
                y_min = global_min - 0.05 * value_range
                y_max = global_max + 0.05 * value_range
            else:
                y_min, y_max = None, None
            
            # Collect all data for this dataset
            all_methods_data = {}
            
            # Add distributed server data (5, 10, etc.)
            for num_servers in sorted(data_dict[dataset_name].keys()):
                for (method, beamwidth), data_points in data_dict[dataset_name][num_servers].items():
                    key = (method, beamwidth, num_servers)
                    all_methods_data[key] = data_points
            
            # Add SINGLE_SERVER data once
            if dataset_name in single_server_dict:
                print(f"  Adding SINGLE_SERVER {plot_type} data for {dataset_name} (row {dataset_idx}, col {metric_idx})")
                for beamwidth, data_points in single_server_dict[dataset_name].items():
                    key = ('SINGLE_SERVER', beamwidth, 1)
                    all_methods_data[key] = data_points
                    print(f"    Added SINGLE_SERVER beamwidth={beamwidth} with {len(data_points)} points")
            
            # Sort keys for consistent plotting order
            method_order = ['STATE_SEND', 'SINGLE_SERVER']
            sorted_keys = sorted(all_methods_data.keys(),
                               key=lambda x: (method_order.index(x[0]) if x[0] in method_order else 999, x[2], x[1]))
            
            # Plot each method/beamwidth/server combination (excluding SCATTER_GATHER)
            for (method, beamwidth, num_servers) in sorted_keys:
                if method == 'SCATTER_GATHER':
                    continue
                data_points = all_methods_data[(method, beamwidth, num_servers)]
                
                # Extract recall (x) and value (y)
                x_values = [point[1] for point in data_points if len(point) >= 2]
                y_values = [point[0] for point in data_points if len(point) >= 2]
                
                if x_values:
                    # Sort by recall
                    sorted_points = sorted(zip(x_values, y_values))
                    x_values_sorted, y_values_sorted = zip(*sorted_points)
                    
                    # Get color for method
                    color = METHOD_COLORS.get(method, '#000000')
                    
                    # Get marker for method
                    marker = METHOD_MARKERS.get(method, 'o')
                    
                    # Adjust marker size based on marker type
                    if marker == '.':
                        markersize = 8
                    elif marker == 'x' or marker == '+':
                        markersize = 6
                    else:
                        markersize = 4
                    
                    # Get line style based on server count
                    if method == 'SINGLE_SERVER':
                        linestyle = '-'
                    else:
                        linestyle = SERVER_LINE_STYLES.get(num_servers, '-')
                    
                    # Plot
                    ax.plot(x_values_sorted, y_values_sorted,
                           marker=marker, linestyle=linestyle,
                           linewidth=2, markersize=markersize, color=color,
                           alpha=0.8)
            
            # Set title with dataset name and size
            dataset_size_str = dataset_sizes.get(dataset_name, '')
            title = f"{plot_type}"
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            ax.set_xlim(min_recall, 1.01)
            
            # Set y-axis limits
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
            
            ax.grid(True, alpha=0.3)
            
            # Create custom legend (only on the first plot - leftmost IOs plot)
            if metric_idx == 0:  # IOs plot (leftmost)
                from matplotlib.lines import Line2D
                legend_handles = []
                
                # Section 1: Line styles (server configurations)
                legend_handles.append(Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='5 servers'))
                legend_handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='10 servers'))
                
                # Section 2: Markers (methods with their colors)
                legend_handles.append(Line2D([0], [0], color=METHOD_COLORS['STATE_SEND'], marker='x', 
                                            linestyle='None', markersize=6, label='STATE_SEND'))
                # legend_handles.append(Line2D([0], [0], color=METHOD_COLORS['SINGLE_SERVER'], marker='+', 
                                            # linestyle='None', markersize=6, label='SINGLE_SERVER'))
                
                # Add legend
                ax.legend(handles=legend_handles, fontsize=6, loc='best', framealpha=0.9)
            
            # Labels
            ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
            # Add x-axis label to both plots in the bottom row
            if dataset_idx == num_datasets - 1:
                ax.set_xlabel('Recall@10', fontsize=12)
            
            # Set tick label sizes
            ax.tick_params(axis='both', which='major', labelsize=13)
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot Mean IOs and Mean Inter vs Recall - two plots per dataset with all server configs combined'
    )
    parser.add_argument(
        'logs_folder',
        type=str,
        help='Path to the root folder containing log subfolders'
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
        default='mean_ios_inter_recall_comparison.png',
        help='Output filename for the plot (default: mean_ios_inter_recall_comparison.png)'
    )
    
    args = parser.parse_args()
    
    print(f"Collecting data from: {args.logs_folder}")
    ios_data, hops_data, inter_data, single_server_ios_data, single_server_hops_data, single_server_inter_data = collect_data(args.logs_folder)
    
    if not ios_data:
        print("No data collected. Please check your log folder structure.")
        return
    
    # Extract dataset sizes for titles
    dataset_sizes = {}
    for dataset_name in ios_data.keys():
        for num_servers in ios_data[dataset_name].keys():
            for (method, beamwidth), _ in ios_data[dataset_name][num_servers].items():
                # Find any folder with this dataset to get its size
                logs_root = Path(args.logs_folder)
                for folder in logs_root.iterdir():
                    if not folder.is_dir():
                        continue
                    metadata = parse_folder_name(folder.name)
                    if metadata and metadata['dataset_name'] == dataset_name:
                        dataset_sizes[dataset_name] = metadata['dataset_size']
                        break
                if dataset_name in dataset_sizes:
                    break
            if dataset_name in dataset_sizes:
                break
    
    print(f"\nFound data for:")
    for dataset_name in sorted(ios_data.keys()):
        print(f"  Dataset: {dataset_name} ({dataset_sizes.get(dataset_name, 'unknown size')})")
        for num_servers in sorted(ios_data[dataset_name].keys()):
            print(f"    {num_servers} server(s): {len(ios_data[dataset_name][num_servers])} configurations")
    
    # Print interpolated IOs and Inter at specific recall values
    print_recall_comparisons(ios_data, hops_data, inter_data, single_server_ios_data, single_server_hops_data, single_server_inter_data, target_recalls=[0.9, 0.95])
    
    print(f"\nGenerating plot...")
    fig = plot_comparison_grid(ios_data, hops_data, inter_data, single_server_ios_data, single_server_hops_data, single_server_inter_data, args.min_recall, dataset_sizes)
    
    if fig:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {args.output}")
    else:
        print("Failed to generate plot.")


if __name__ == '__main__':
    main()
