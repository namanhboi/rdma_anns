#!/usr/bin/env python3
"""
Script to plot distance comparisons and disk I/O comparison across different server configurations.
Both metrics are shown side by side in a single plot.
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

# Define consistent colors for different methods
METHOD_COLORS = {
    'STATE_SEND': '#1f77b4',      # Blue
    'SCATTER_GATHER': '#ff7f0e',  # Orange
    'SINGLE_SERVER': '#2ca02c',   # Green
}

# Define legend name mapping
LEGEND_NAME_MAPPING = {
    'STATE_SEND': 'BatANN',
    'SCATTER_GATHER': 'ScatterGather',
    'SINGLE_SERVER': 'SingleServer'
}

# Define markers for different methods
METHOD_MARKERS = {
    'STATE_SEND': 'x',           # X marker
    'SCATTER_GATHER': '.',       # Point marker
    'SINGLE_SERVER': '+',        # Plus marker
}

# Define line styles for different server counts
SERVER_LINE_STYLES = {
    1: ':',      # Dotted for 1 server (SINGLE_SERVER)
    5: '-',      # Solid for 5 servers
    10: '--',    # Dashed for 10 servers
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
            if len(parts) < 9:
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


def collect_data(logs_folders):
    """
    Collect all data from log folders.
    
    Args:
        logs_folders: List of paths to root folders containing log subfolders
    
    Returns a tuple: (data_dict, dataset_info)
    where data_dict is {num_servers: {(method, beamwidth): [(qps, latency, recall, distance_comparisons, mean_io), ...]}}
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
            
            # Store data with (method, beamwidth) as key within num_servers
            num_servers = metadata['num_servers']
            beamwidth = metadata['beamwidth']
            method = metadata['method']
            
            data[num_servers][(method, beamwidth)] = data_points
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


def plot_combined_metrics(data, dataset_info, min_recall, beamwidth_filter):
    """
    Plot both distance comparisons and disk I/O vs recall side by side.
    
    data: {num_servers: {(method, beamwidth): [(qps, latency, recall, distance_comparisons, mean_io), ...]}}
    dataset_info: {'name': str, 'size': str}
    beamwidth_filter: Only plot data for this beamwidth (or None for all)
    """
    if not data:
        print("No data to plot!")
        return None
    
    # Print values at recall >= 0.95 for both metrics
    for metric_idx, metric_name in [(3, 'Distance Comparisons'), (4, 'Mean I/O')]:
        print("\n" + "="*80)
        print(f"{metric_name} at Recall >= 0.95")
        print("="*80)
        
        for num_servers in sorted(data.keys()):
            print(f"\n{num_servers} Server{'s' if num_servers > 1 else ''}:")
            print(f"{'Method':<20} {'Beamwidth':<12} {metric_name:<20}")
            print("-" * 52)
            
            for (method, beamwidth) in sorted(data[num_servers].keys()):
                data_points = data[num_servers][(method, beamwidth)]
                
                # Extract recall and metric values
                recalls = [point[2] for point in data_points if len(point) >= 5]
                metric_values = [point[metric_idx] for point in data_points if len(point) >= 5]
                
                if recalls:
                    # Sort by recall
                    sorted_data = sorted(zip(recalls, metric_values))
                    recalls_sorted, metric_sorted = zip(*sorted_data)
                    
                    # Interpolate at 0.95
                    value_at_95 = interpolate_at_recall(recalls_sorted, metric_sorted, 0.95)
                    
                    if value_at_95 is not None:
                        # Use display name in output
                        display_name = LEGEND_NAME_MAPPING.get(method, method)
                        print(f"{display_name:<20} {beamwidth:<12} {value_at_95:<20.4f}")
                    else:
                        display_name = LEGEND_NAME_MAPPING.get(method, method)
                        print(f"{display_name:<20} {beamwidth:<12} {'N/A':<20}")
        
        print("\n" + "="*80 + "\n")
    
    # Create a figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))
    
    # Sort by number of servers
    sorted_configs = sorted(data.items())
    
    # Plot both metrics
    for metric_idx, ax, ylabel, metric_title in [
        (3, ax1, 'Distance Comparisons', 'Distance Comparisons vs Recall'),
        (4, ax2, 'Mean I/O Operations', 'Mean I/O vs Recall')
    ]:
        plotted_items = []
        
        for num_servers, methods_data in sorted_configs:
            # If beamwidth_filter is specified, only plot that beamwidth
            if beamwidth_filter is not None:
                methods_to_plot = {(method, bw): dp for (method, bw), dp in methods_data.items() if bw == beamwidth_filter}
                if not methods_to_plot:
                    continue
            else:
                # Plot all beamwidths
                methods_to_plot = methods_data
            
            # Sort by method for consistent ordering
            method_order = ['STATE_SEND', 'SCATTER_GATHER', 'SINGLE_SERVER']
            sorted_keys = sorted(methods_to_plot.keys(),
                               key=lambda x: (method_order.index(x[0]) if x[0] in method_order else 999, x[1]))
            
            for (method, beamwidth) in sorted_keys:
                data_points = methods_to_plot[(method, beamwidth)]
                
                # Extract recall and metric values
                recalls = [point[2] for point in data_points if len(point) >= 5]
                metric_values = [point[metric_idx] for point in data_points if len(point) >= 5]
                
                if recalls:
                    # Sort data by recall
                    sorted_data = sorted(zip(recalls, metric_values))
                    recalls_sorted, metric_sorted = zip(*sorted_data)
                    
                    # Get color for method
                    color = METHOD_COLORS.get(method, '#000000')
                    
                    # Get marker for method
                    marker = METHOD_MARKERS.get(method, 'o')
                    
                    # Adjust marker size based on marker type
                    if marker == '.':
                        markersize = 8  # Larger for point marker
                    elif marker == 'x' or marker == '+':
                        markersize = 6  # Medium for x and + markers
                    else:
                        markersize = 4  # Default
                    
                    # Get line style based on server count
                    linestyle = SERVER_LINE_STYLES.get(num_servers, '-')
                    
                    # Create label with display name
                    display_name = LEGEND_NAME_MAPPING.get(method, method)
                    if method == 'SINGLE_SERVER':
                        label = f"{display_name}"
                    else:
                        label = f"{display_name} ({num_servers}s)"
                    
                    # Plot metric
                    ax.plot(recalls_sorted, metric_sorted, 
                           marker=marker, linestyle=linestyle, 
                           linewidth=2, markersize=markersize, color=color, alpha=0.8)
                    
                    # Track for legend
                    plotted_items.append((method, num_servers, color, marker, linestyle, markersize, label))
        
        # Set subplot title
        ax.set_title(metric_title, fontsize=8, fontweight='bold')
        ax.set_xlabel('Recall@10', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(min_recall, 1.01)
        ax.grid(True, alpha=0.3)
        
        # Create custom legend
        if plotted_items:
            from matplotlib.lines import Line2D
            legend_handles = []
            
            # Add line style legend (server configurations)
            unique_servers = sorted(set(item[1] for item in plotted_items))
            for num_servers in unique_servers:
                linestyle = SERVER_LINE_STYLES.get(num_servers, '-')
                if num_servers == 1:
                    label = '1 server'
                else:
                    label = f'{num_servers} servers'
                legend_handles.append(Line2D([0], [0], color='black', linestyle=linestyle, 
                                            linewidth=2, label=label))
            
            # Add method legend (colors and markers) with display names
            unique_methods = []
            seen_methods = set()
            for item in plotted_items:
                if item[0] not in seen_methods:
                    unique_methods.append(item[0])
                    seen_methods.add(item[0])
            
            for method in unique_methods:
                color = METHOD_COLORS.get(method, '#000000')
                marker = METHOD_MARKERS.get(method, 'o')
                markersize = 8 if marker == '.' else (8 if marker in ['x', '+'] else 4)
                display_name = LEGEND_NAME_MAPPING.get(method, method)
                legend_handles.append(Line2D([0], [0], color=color, marker=marker,
                                            linestyle='None', markersize=markersize, label=display_name))
            
            ax.legend(handles=legend_handles, fontsize=8, loc='best', framealpha=0.9)
    
    # Create overall figure title
    if dataset_info['name'] and dataset_info['size']:
        title = f"{dataset_info['name']} ({dataset_info['size']}) - Performance Metrics"
        if beamwidth_filter is not None:
            title += f" (Beamwidth={beamwidth_filter})"
    else:
        title = 'Performance Metrics Comparison'
        if beamwidth_filter is not None:
            title += f" (Beamwidth={beamwidth_filter})"
    
    # fig.suptitle(title, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot distance comparisons and disk I/O vs recall for different server configurations (combined in one figure)'
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
        default='combined_metrics.png',
        help='Output filename for the combined plot (default: combined_metrics.png)'
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
        methods_beamwidths = sorted(data[num_servers].keys())
        print(f"  {num_servers} server(s):")
        for method, beamwidth in methods_beamwidths:
            display_name = LEGEND_NAME_MAPPING.get(method, method)
            print(f"    {display_name}, beamwidth={beamwidth}")
    
    if args.beamwidth is not None:
        print(f"\nFiltering for beamwidth={args.beamwidth}")
    
    # Generate combined plot
    print(f"\nGenerating combined metrics plot...")
    fig = plot_combined_metrics(data, dataset_info, args.min_recall, args.beamwidth)
    
    if fig:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Combined metrics plot saved to: {args.output}")
    else:
        print("Failed to generate combined metrics plot.")
    
    plt.close()

if __name__ == '__main__':
    main()
