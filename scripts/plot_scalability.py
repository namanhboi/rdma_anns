#!/usr/bin/env python3
"""
Script to plot scalability of STATE_SEND, SCATTER_GATHER, and SINGLE_SERVER 
approaches as the number of servers/threads increases.
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Define legend name mapping
LEGEND_NAME_MAPPING = {
    'STATE_SEND': 'BatANN',
    'SCATTER_GATHER': 'ScatterGather',
    'SINGLE_SERVER': 'SingleServer'
}

# Define consistent colors and markers for each method
METHOD_STYLES = {
    'STATE_SEND': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'label': 'BatANN'},
    'SCATTER_GATHER': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-', 'label': 'ScatterGather'},
    'SINGLE_SERVER': {'color': '#2ca02c', 'marker': '^', 'linestyle': '--', 'label': 'SingleServer'},
}


def parse_folder_name(folder_name):
    """
    Parse the log folder name to extract metadata.
    Expected formats:
los_${DIST_SEARCH_MODE}_${MODE}_${DATASET_NAME}_${DATASET_SIZE}_${NUM_SERVERS}_${COUNTER_SLEEP_MS}_MS_NUM_SEARCH_THREADS_${NUM_SEARCH_THREADS}_MAX_BATCH_SIZE_${MAX_BATCH_SIZE}_K_${K_VALUE}_OVERLAP_${OVERLAP}_BEAMWIDTH_${BEAM_WIDTH}
    
    Returns a dict with extracted fields or None if parsing fails.
    """
    # Only process folders that start with 'logs_'
    if not folder_name.startswith('logs_'):
        return None
    
    # Split by underscore
    parts = folder_name.split('_')
    
    if len(parts) < 6:
        return None
    
    try:
        # Extract method (STATE_SEND, SCATTER_GATHER, or SINGLE_SERVER)
        if parts[1] == 'STATE' and parts[2] == 'SEND':
            method = 'STATE_SEND'
            start_idx = 3
        elif parts[1] == 'SCATTER' and parts[2] == 'GATHER':
            method = 'SCATTER_GATHER'
            start_idx = 3
        elif parts[1] == 'SINGLE' and parts[2] == 'SERVER':
            method = 'SINGLE_SERVER'
            start_idx = 3
        else:
            return None
        
        # Extract dataset name and size
        dataset_name = parts[start_idx + 1] if start_idx + 1 < len(parts) else None
        dataset_size = parts[start_idx + 2] if start_idx + 2 < len(parts) else None
        
        # Find NUM_SERVERS (for distributed) or num_threads (for single server)
        num_servers = None
        num_threads = None
        
        # Look for NUM_SEARCH_THREADS pattern
        for i in range(len(parts)):
            if parts[i] == 'THREADS' and i > 0 and parts[i-1] == 'SEARCH' and i > 1 and parts[i-2] == 'NUM':
                if i + 1 < len(parts) and parts[i+1].isdigit():
                    num_threads = int(parts[i+1])
                    break
        
        if method == 'SINGLE_SERVER':
            # For SINGLE_SERVER, we use num_threads as the key metric
            if num_threads is None:
                return None
            return {
                'method': method,
                'num_servers': 1,
                'num_threads': num_threads,
                'dataset_name': dataset_name,
                'dataset_size': dataset_size,
                'full_name': folder_name
            }
        else:
            # For STATE_SEND and SCATTER_GATHER, find NUM_SERVERS
            # It should be a small integer (2-5) after the dataset size
            for i in range(start_idx + 2, len(parts)):
                if parts[i].isdigit() and len(parts[i]) <= 2:
                    # Check that previous part might be dataset size (contains M, K, etc.)
                    if i > 0 and any(char in parts[i-1] for char in ['M', 'K', 'G']):
                        num_servers = int(parts[i])
                        break
            
            if num_servers is None:
                return None
            
            return {
                'method': method,
                'num_servers': num_servers,
                'num_threads': num_threads if num_threads else 8,  # Default to 8
                'dataset_name': dataset_name,
                'dataset_size': dataset_size,
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
            if len(parts) < 9:
                continue
            
            try:
                # Extract QPS (column 3), and Recall (column 8)
                qps = float(parts[2])
                avg_latency = float(parts[3])
                recall = float(parts[-1])
                
                data_points.append((qps, avg_latency, recall / 100.0))  # Convert recall to 0-1 range
            except (ValueError, IndexError):
                continue
    
    except FileNotFoundError:
        print(f"Warning: File not found: {log_file_path}")
    except Exception as e:
        print(f"Error parsing {log_file_path}: {e}")
    
    return data_points


def get_qps_at_recall(data_points, target_recall=0.95):
    """
    Get QPS at target recall using linear interpolation.
    
    Returns QPS value or None if target_recall is out of range.
    """
    if not data_points:
        return None
    
    # Sort by recall
    sorted_points = sorted(data_points, key=lambda x: x[2])
    
    # Check if target_recall is within the range of available data
    min_recall = sorted_points[0][2]
    max_recall = sorted_points[-1][2]
    
    if target_recall < min_recall or target_recall > max_recall:
        print(f"Warning: Target recall {target_recall} is outside available range [{min_recall:.3f}, {max_recall:.3f}]")
        return None
    
    # Find the two points to interpolate between
    for i in range(len(sorted_points) - 1):
        recall_low = sorted_points[i][2]
        recall_high = sorted_points[i + 1][2]
        
        if recall_low <= target_recall <= recall_high:
            qps_low = sorted_points[i][0]
            qps_high = sorted_points[i + 1][0]
            
            # Linear interpolation
            if recall_high == recall_low:
                # Avoid division by zero
                interpolated_qps = qps_low
            else:
                weight = (target_recall - recall_low) / (recall_high - recall_low)
                interpolated_qps = qps_low + weight * (qps_high - qps_low)
            
            return interpolated_qps
    
    # If we get here, target_recall exactly matches one of the points
    for qps, _, recall in sorted_points:
        if abs(recall - target_recall) < 1e-6:
            return qps
    
    return None


def collect_data(logs_root_folder, target_recall=0.95):
    """
    Collect QPS at target recall for all methods, organized by dataset.
    
    Returns a dict: {dataset_key: {'data': {method: {num_servers or num_threads: qps}}, 'info': {'name': str, 'size': str}}}
    """
    datasets = defaultdict(lambda: {'data': defaultdict(dict), 'info': {'name': None, 'size': None}})
    logs_root = Path(logs_root_folder)
    
    if not logs_root.exists():
        print(f"Error: Root folder '{logs_root_folder}' does not exist")
        return datasets
    
    # Iterate through all subdirectories
    for folder in logs_root.iterdir():
        if not folder.is_dir():
            continue
        
        # Parse folder name
        metadata = parse_folder_name(folder.name)
        if metadata is None:
            continue
        
        # Create dataset key from name and size
        dataset_name = metadata['dataset_name']
        dataset_size = metadata['dataset_size']
        dataset_key = f"{dataset_name}_{dataset_size}"
        
        # Store dataset info
        if datasets[dataset_key]['info']['name'] is None:
            datasets[dataset_key]['info']['name'] = dataset_name
            datasets[dataset_key]['info']['size'] = dataset_size
        
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
        
        # Get QPS at target recall
        qps = get_qps_at_recall(data_points, target_recall)
        if qps is None:
            print(f"Warning: Could not find QPS at recall={target_recall} in {folder.name}")
            continue
        
        method = metadata['method']
        
        # For SINGLE_SERVER, key by num_threads; for others, key by num_servers
        if method == 'SINGLE_SERVER':
            key = metadata['num_threads']
        else:
            key = metadata['num_servers']
        
        datasets[dataset_key]['data'][method][key] = qps
        
        # Use display name in output
        display_name = LEGEND_NAME_MAPPING.get(method, method)
        print(f"Loaded {folder.name}: {display_name}, dataset={dataset_key}, key={key}, QPS@{target_recall}={qps:.2f}")
    
    return dict(datasets)


def plot_single_dataset(ax, data, dataset_info, target_recall, threads_per_server=8, show_legend=True):
    """
    Plot scalability comparison for a single dataset on a given axis.
    
    ax: matplotlib axis to plot on
    data: {method: {num_servers or num_threads: qps}}
    dataset_info: {'name': str, 'size': str}
    show_legend: whether to show the legend on this subplot
    """
    if not data:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return False
    
    # Get baseline QPS from SINGLE_SERVER with 8 threads
    baseline_qps = None
    if 'SINGLE_SERVER' in data and 8 in data['SINGLE_SERVER']:
        baseline_qps = data['SINGLE_SERVER'][8]
    else:
        # Try to find any single server data as fallback
        if 'SINGLE_SERVER' in data and data['SINGLE_SERVER']:
            min_threads = min(data['SINGLE_SERVER'].keys())
            baseline_qps = data['SINGLE_SERVER'][min_threads]
        else:
            ax.text(0.5, 0.5, 'No baseline data', ha='center', va='center', transform=ax.transAxes)
            return False
    
    # Plot optimal scalability line (y = x)
    max_x = 1
    for method in ['STATE_SEND', 'SCATTER_GATHER', 'SINGLE_SERVER']:
        if method not in data or not data[method]:
            continue
        if method == 'SINGLE_SERVER':
            max_x = max(max_x, max(data[method].keys()) / threads_per_server)
        else:
            max_x = max(max_x, max(data[method].keys()))
    
    ax.plot([1, max_x], [1, max_x], 'k--', linewidth=1.5, alpha=0.5, label='Optimal')
    
    # Plot each method
    for method in ['STATE_SEND', 'SCATTER_GATHER', 'SINGLE_SERVER']:
        if method not in data or not data[method]:
            continue
        
        style = METHOD_STYLES[method]
        
        x_values = []
        y_values = []
        
        if method == 'SINGLE_SERVER':
            # For SINGLE_SERVER, x-axis is num_threads / threads_per_server
            for num_threads, qps in sorted(data[method].items()):
                equiv_servers = num_threads / threads_per_server
                speedup = qps / baseline_qps
                x_values.append(equiv_servers)
                y_values.append(speedup)
        else:
            # For distributed methods, add origin point (1, 1) to connect to baseline
            x_values.append(1)
            y_values.append(1)
            
            # Then add actual data points
            for num_servers, qps in sorted(data[method].items()):
                speedup = qps / baseline_qps
                x_values.append(num_servers)
                y_values.append(speedup)
        
        ax.plot(x_values, y_values, 
               label=style['label'], 
               color=style['color'],
               marker=style['marker'], 
               linestyle=style['linestyle'],
               linewidth=2.5, 
               markersize=10)
    
    ax.set_xlabel('Number of Servers (or Equivalent)', fontsize=14)
    ax.set_ylabel('Scalability', fontsize=14)
    
    # Create title with dataset info
    if dataset_info['name'] and dataset_info['size']:
        title = f"{dataset_info['name']} ({dataset_info['size']})"
    else:
        title = 'Scalability Comparison'
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Only show legend if requested
    if show_legend:
        ax.legend(fontsize=15, loc='upper left')
    
    # Set x-axis to show integer values
    all_x = []
    for method in ['STATE_SEND', 'SCATTER_GATHER', 'SINGLE_SERVER']:
        if method not in data or not data[method]:
            continue
        if method == 'SINGLE_SERVER':
            all_x.extend([t / threads_per_server for t in data[method].keys()])
        else:
            all_x.extend(data[method].keys())
    if all_x:
        min_x = int(min(all_x))
        max_x = int(max(all_x)) + 1
        ax.set_xticks(range(min_x, max_x))
    
    # Set y-axis to start from a reasonable value
    ax.set_ylim(bottom=0)
    
    return True


def plot_scalability_grid(datasets, target_recall, threads_per_server=8):
    """
    Plot scalability comparison for multiple datasets in a grid layout.
    2 plots per row, centered if odd number of datasets.
    Legend appears only on the first subplot.
    
    datasets: {dataset_key: {'data': {method: {num_servers or num_threads: qps}}, 'info': {'name': str, 'size': str}}}
    """
    if not datasets:
        print("No data to plot!")
        return None
    
    n_datasets = len(datasets)
    
    # Calculate grid dimensions: 2 columns, rows as needed
    n_cols = 2
    n_rows = (n_datasets + 1) // 2  # Ceiling division
    
    # Sort datasets alphabetically (case-insensitive)
    sorted_dataset_keys = sorted(datasets.keys(), key=lambda x: x.lower())
    
    # For odd number of datasets, use subplot2grid for all to allow centering
    if n_datasets % 2 == 1:
        # Create figure
        fig = plt.figure(figsize=(6 * n_cols, 6 * n_rows))
        
        for idx, dataset_key in enumerate(sorted_dataset_keys):
            dataset = datasets[dataset_key]
            data = dataset['data']
            dataset_info = dataset['info']
            
            if idx == n_datasets - 1:
                # Last plot - center it by using colspan and offset
                ax = plt.subplot2grid((n_rows, n_cols * 2), (n_rows - 1, 1), colspan=2, fig=fig)
            else:
                # Regular plots - 2 per row
                row = idx // 2
                col = idx % 2
                ax = plt.subplot2grid((n_rows, n_cols * 2), (row, col * 2), colspan=2, fig=fig)
            
            # Show legend only on first subplot
            show_legend = (idx == 0)
            plot_single_dataset(ax, data, dataset_info, target_recall, threads_per_server, show_legend=show_legend)
            
            # Print baseline info
            if 'SINGLE_SERVER' in data and 8 in data['SINGLE_SERVER']:
                baseline_qps = data['SINGLE_SERVER'][8]
                print(f"{dataset_key} - Baseline QPS (SingleServer, 8 threads): {baseline_qps:.2f}")
    else:
        # Even number - use regular subplot
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 6 * n_rows))
        
        # Make axes always 2D for consistent indexing
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, dataset_key in enumerate(sorted_dataset_keys):
            dataset = datasets[dataset_key]
            data = dataset['data']
            dataset_info = dataset['info']
            
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Show legend only on first subplot
            show_legend = (idx == 0)
            plot_single_dataset(ax, data, dataset_info, target_recall, threads_per_server, show_legend=show_legend)
            
            # Print baseline info
            if 'SINGLE_SERVER' in data and 8 in data['SINGLE_SERVER']:
                baseline_qps = data['SINGLE_SERVER'][8]
                print(f"{dataset_key} - Baseline QPS (SingleServer, 8 threads): {baseline_qps:.2f}")
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot scalability of STATE_SEND, SCATTER_GATHER, and SINGLE_SERVER approaches'
    )
    parser.add_argument(
        'logs_folder',
        type=str,
        help='Path to the root folder containing all log subfolders'
    )
    parser.add_argument(
        '--target-recall',
        type=float,
        default=0.95,
        help='Target recall value to extract QPS for (default: 0.95)'
    )
    parser.add_argument(
        '--threads-per-server',
        type=int,
        default=8,
        help='Number of threads per server for distributed methods (default: 8)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='scalability_comparison.png',
        help='Output filename for the plot (default: scalability_comparison.png)'
    )
    
    args = parser.parse_args()
    
    print(f"Collecting data from: {args.logs_folder}")
    print(f"Target recall: {args.target_recall}")
    print(f"Threads per server: {args.threads_per_server}")
    
    datasets = collect_data(args.logs_folder, args.target_recall)
    
    if not datasets:
        print("No data collected. Please check your log folder structure.")
        return
    
    print(f"\nFound {len(datasets)} dataset(s):")
    for dataset_key, dataset in sorted(datasets.items()):
        info = dataset['info']
        data = dataset['data']
        print(f"\n  {dataset_key} - {info['name']} ({info['size']}):")
        for method, values in data.items():
            display_name = LEGEND_NAME_MAPPING.get(method, method)
            print(f"    {display_name}: {sorted(values.keys())}")
    
    print(f"\nGenerating plot(s)...")
    fig = plot_scalability_grid(datasets, args.target_recall, args.threads_per_server)
    
    if fig:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {args.output}")
    else:
        print("Failed to generate plot.")


if __name__ == '__main__':
    main()
