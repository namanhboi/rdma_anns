#!/usr/bin/env python3
"""
Script to plot QPS vs Recall curves comparing different datasets and methods.
Creates a grid with dataset sizes (100M, 1B) as columns, dataset names as rows.
All server configurations are plotted in the same subplot with different line/marker styles.
Adds speedup annotations for STATE_SEND vs SCATTER_GATHER.
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Define consistent colors for different methods
METHOD_COLORS = {
    'STATE_SEND': '#1f77b4',      # Blue
    'SCATTER_GATHER': '#ff7f0e',  # Orange
    'SINGLE_SERVER': '#2ca02c',   # Green
}

# Legend name mapping
LEGEND_NAME_MAPPING = {
    'STATE_SEND': 'BatANN',
    'SCATTER_GATHER': 'ScatterGather',
    'SINGLE_SERVER': 'SingleServer'
}

# Define line styles for different server configurations
SERVER_LINE_STYLES = {
    1: (0, (1, 1)),         # Densely dotted
    2: ':',                 # Dotted
    3: (0, (3, 1, 1, 1)),   # Densely dashdotted
    4: '-.',                # Dash-dot
    5: '--',                # Dashed
    6: (0, (5, 5)),         # Dashed (longer)
    7: (0, (3, 1, 1, 1, 1, 1)),  # Densely dashdotdotted
    8: (0, (5, 1)),         # Densely dashed
    9: (0, (3, 5, 1, 5)),   # Dashdotted (longer)
    10: '-',                # Solid
}

# Define marker styles for different server configurations
SERVER_MARKERS = {
    1: 'o',      # Circle
    2: 's',      # Square
    3: '^',      # Triangle up
    4: 'D',      # Diamond
    5: 'o',      # Hollow circle (will use fillstyle='none')
    6: 'p',      # Pentagon
    7: '*',      # Star
    8: 'v',      # Triangle down
    9: 'P',      # Plus (filled)
    10: '^',     # Hollow triangle (will use fillstyle='none')
}

# Define which markers should be hollow
HOLLOW_MARKERS = {5, 10}  # Server configs that should have hollow markers


def get_display_name(dataset_name):
    """
    Convert dataset name to display format.
    
    Args:
        dataset_name: Internal dataset name (e.g., 'bigann', 'deep1b')
    
    Returns:
        Display name (e.g., 'BIGANN', 'DEEP')
    """
    name_mapping = {
        'bigann': 'BIGANN',
        'deep1b': 'DEEP',
        'MSSPACEV1B': 'MSSPACEV1B',
        'msspacev1b': 'MSSPACEV1B',
    }
    
    return name_mapping.get(dataset_name, dataset_name.upper())


def parse_folder_name(folder_name):
    """
    Parse the log folder name to extract metadata.
    Expected formats:
    - logs_STATE_SEND_distributed_${DATASET_NAME}_${DATASET_SIZE}_${NUM_SERVERS}_...BEAMWIDTH_${BEAM_WIDTH}_timestamp
    - logs_SCATTER_GATHER_distributed_${DATASET_NAME}_${DATASET_SIZE}_${NUM_SERVERS}_...BEAMWIDTH_${BEAM_WIDTH}_timestamp
    - logs_SINGLE_SERVER_${DATASET_NAME}_${DATASET_SIZE}_...NUM_SEARCH_THREADS_${NUM_THREADS}_...BEAMWIDTH_${BEAM_WIDTH}_timestamp
    
    For SINGLE_SERVER: num_threads / 8 = equivalent number of servers
    (8 threads = 1 server, 16 threads = 2 servers, 24 threads = 3 servers, 32 threads = 4 servers, etc.)
    
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
    # For distributed modes: distributed_DATASET_SIZE
    # For single server: logs_SINGLE_SERVER_DATASET_SIZE
    if method in ['STATE_SEND', 'SCATTER_GATHER']:
        dataset_match = re.search(r'distributed_([A-Za-z0-9]+)_(\d+[BKMG])', folder_name)
        if not dataset_match:
            return None
        dataset_name_with_size = dataset_match.group(1)
        dataset_size = dataset_match.group(2)
        
        # Remove size suffix from dataset name if present
        # e.g., "bigann_100M" -> "bigann", "deep1b" -> "deep1b", "MSSPACEV1B" -> "MSSPACEV1B"
        # Look for common patterns where size is embedded
        dataset_name = re.sub(r'_?\d+[BKMG]$', '', dataset_name_with_size)
        # Also handle cases like "deep1b" where "1b" is part of the name
        # Keep it as-is if no underscore+size pattern found
        if not dataset_name:
            dataset_name = dataset_name_with_size
        
        # Extract number of servers - look for pattern after dataset size
        # Pattern: ...100M_NUM_SERVERS_... or ...1B_NUM_SERVERS_...
        num_servers_match = re.search(r'_(\d+[BKMG])_(\d+)_\d+_MS', folder_name)
        if not num_servers_match:
            return None
        num_servers = int(num_servers_match.group(2))
    else:  # SINGLE_SERVER
        dataset_match = re.search(r'logs_SINGLE_SERVER_([A-Za-z0-9]+)_(\d+[BKMG])', folder_name)
        if not dataset_match:
            return None
        dataset_name_with_size = dataset_match.group(1)
        dataset_size = dataset_match.group(2)
        
        # Remove size suffix from dataset name if present
        dataset_name = re.sub(r'_?\d+[BKMG]$', '', dataset_name_with_size)
        if not dataset_name:
            dataset_name = dataset_name_with_size
        
        # Extract number of search threads for SINGLE_SERVER
        # Pattern: NUM_SEARCH_THREADS_${NUM_THREADS}
        num_threads_match = re.search(r'NUM_SEARCH_THREADS_(\d+)', folder_name)
        if not num_threads_match:
            return None
        num_threads = int(num_threads_match.group(1))
        
        # Map threads to equivalent number of servers (8 threads = 1 server)
        num_servers = num_threads // 8
        if num_servers == 0:
            return None  # Invalid configuration
    
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
                # Extract QPS (column 3), and Recall (column 9)
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


def collect_data(logs_folder):
    """
    Collect all data from log folder.
    
    Args:
        logs_folder: Path to root folder containing log subfolders
    
    Returns a nested dict: 
    {
        dataset_name: {
            num_servers: {
                (method, beamwidth): [(qps, latency, recall), ...]
            }
        }
    }
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    single_server_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    logs_root = Path(logs_folder)
    
    if not logs_root.exists():
        print(f"Error: Root folder '{logs_folder}' does not exist")
        return data
    
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
        data_points = parse_client_log(client_log)
        if not data_points:
            print(f"  Warning: No data extracted from {client_log}")
            continue
        
        # Store data with dataset -> num_servers -> (method, beamwidth) hierarchy
        # Use base dataset name and store size separately for later organization
        dataset_name = metadata['dataset_name']
        dataset_size = metadata['dataset_size']
        num_servers = metadata['num_servers']
        method = metadata['method']
        beamwidth = metadata['beamwidth']
        
        # Create a unique key combining dataset name and size for storage
        dataset_key = f"{dataset_name}_{dataset_size}"
        
        # Store SINGLE_SERVER data separately for now
        if method == 'SINGLE_SERVER':
            single_server_data[dataset_key][num_servers][(method, beamwidth)] = data_points
            print(f"  Loaded {len(data_points)} data points from {folder.name} "
                  f"(dataset={dataset_key}, method={method}, beamwidth={beamwidth}, equiv_servers={num_servers})")
        else:
            data[dataset_key][num_servers][(method, beamwidth)] = data_points
            print(f"  Loaded {len(data_points)} data points from {folder.name} "
                  f"(dataset={dataset_key}, method={method}, beamwidth={beamwidth}, servers={num_servers})")
    
    # Merge SINGLE_SERVER data only for configs that have distributed methods
    for dataset_key in list(single_server_data.keys()):
        for num_servers in list(single_server_data[dataset_key].keys()):
            if dataset_key in data and num_servers in data[dataset_key]:
                # This dataset+server config has distributed methods, so include SINGLE_SERVER data
                for key, value in single_server_data[dataset_key][num_servers].items():
                    data[dataset_key][num_servers][key] = value
                print(f"Including SINGLE_SERVER data for {dataset_key} with {num_servers} servers")
            else:
                print(f"Excluding SINGLE_SERVER data for {dataset_key} with {num_servers} servers (no distributed methods)")
    
    return data


def plot_comparison_grid(data, min_recall, dataset_sizes):
    """
    Plot comparison grid with dataset sizes (100M, 1B) as columns, dataset names as rows.
    All server configurations plotted in same subplot with different line/marker styles.
    
    Args:
        data: {dataset_name: {num_servers: {(method, beamwidth): [(qps, latency, recall), ...]}}}
        min_recall: Minimum recall value for x-axis
        dataset_sizes: {dataset_name: size_string} mapping
    
    Returns:
        matplotlib figure
    """
    if not data:
        print("No data to plot!")
        return None
    
    # Reorganize data by (dataset_base_name, dataset_size)
    # Data structure: {dataset_base_name: {dataset_size: {num_servers: {(method, beamwidth): [...]}}}}
    reorganized_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for dataset_key in data.keys():
        # Split dataset_key which is in format "name_size" (e.g., "bigann_100M", "deep1b_1B")
        # The size is always at the end after underscore
        match = re.match(r'(.+)_(\d+[BKMG])$', dataset_key)
        if match:
            base_name = match.group(1)
            dataset_size = match.group(2)
        else:
            # Fallback: use the whole key as base name and get size from dataset_sizes
            base_name = dataset_key
            dataset_size = dataset_sizes.get(dataset_key, 'unknown')
        
        reorganized_data[base_name][dataset_size] = data[dataset_key]
    
    # Get sorted lists
    dataset_base_names = sorted(reorganized_data.keys())
    all_dataset_sizes = set()
    
    for base_name in dataset_base_names:
        all_dataset_sizes.update(reorganized_data[base_name].keys())
    
    # Sort sizes: 100M before 1B
    def size_sort_key(size_str):
        if 'M' in size_str:
            return (0, int(size_str.replace('M', '')))
        elif 'B' in size_str:
            return (1, int(size_str.replace('B', '')))
        return (2, 0)
    
    dataset_size_list = sorted(all_dataset_sizes, key=size_sort_key)
    
    if not dataset_base_names or not dataset_size_list:
        print("No valid data configuration found!")
        return None
    
    num_rows = len(dataset_size_list)
    num_cols = len(dataset_base_names)
    
    print(f"\nCreating grid: {num_rows} rows (dataset sizes) × {num_cols} cols (dataset types)")
    print(f"Dataset types: {dataset_base_names}")
    print(f"Dataset sizes: {dataset_size_list}")
    
    # Create figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows))
    
    # Make sure axes is always 2D
    if num_rows == 1 and num_cols == 1:
        axes = [[axes]]
    elif num_rows == 1:
        axes = [axes]
    elif num_cols == 1:
        axes = [[ax] for ax in axes]
    
    # Find min/max QPS for each dataset (row) separately, only considering points with recall >= min_recall
    dataset_qps_limits = {}
    for base_name in dataset_base_names:
        for dataset_size in dataset_size_list:
            if dataset_size not in reorganized_data[base_name]:
                continue
            
            qps_values = []
            for num_servers in reorganized_data[base_name][dataset_size].keys():
                for (method, beamwidth), data_points in reorganized_data[base_name][dataset_size][num_servers].items():
                    # Only consider points where recall >= min_recall
                    filtered_points = [(qps, recall) for qps, latency, recall in data_points if recall >= min_recall]
                    y_values = [qps for qps, recall in filtered_points]
                    qps_values.extend(y_values)
            
            if qps_values:
                min_qps = min(qps_values)
                max_qps = max(qps_values)
                qps_range = max_qps - min_qps
                y_min = min_qps - 0.05 * qps_range
                y_max = max_qps + 0.05 * qps_range
                dataset_qps_limits[(base_name, dataset_size)] = (y_min, y_max)
                print(f"  Y-axis limits for {base_name} ({dataset_size}): [{y_min:.2f}, {y_max:.2f}]")
            else:
                dataset_qps_limits[(base_name, dataset_size)] = (None, None)
    
    # Plot each cell
    for row_idx, dataset_size in enumerate(dataset_size_list):
        for col_idx, base_name in enumerate(dataset_base_names):
            ax = axes[row_idx][col_idx]
            
            y_min, y_max = dataset_qps_limits.get((base_name, dataset_size), (None, None))
            
            # Check if we have data for this combination
            if dataset_size not in reorganized_data[base_name]:
                # Hide the subplot completely if no data
                ax.set_visible(False)
                continue
            
            dataset_data = reorganized_data[base_name][dataset_size]
            
            # Get all server configs for this dataset+size combination
            server_configs = sorted(dataset_data.keys())
            
            # Track what we've plotted for legend
            plotted_methods = {}  # method -> num_servers list
            
            # Store data for speedup calculation (per server config)
            state_send_data = {}  # (num_servers, beamwidth) -> [(recall, qps), ...]
            scatter_gather_data = {}  # (num_servers, beamwidth) -> [(recall, qps), ...]
            
            # Plot each server config and method/beamwidth combination
            for num_servers in server_configs:
                methods_data = dataset_data[num_servers]
                
                # Sort by method first, then by beamwidth
                method_order = ['STATE_SEND', 'SCATTER_GATHER', 'SINGLE_SERVER']
                sorted_keys = sorted(methods_data.keys(),
                                   key=lambda x: (method_order.index(x[0]) if x[0] in method_order else 999, x[1]))
                
                # Plot each method/beamwidth combination
                for (method, beamwidth) in sorted_keys:
                    data_points = methods_data[(method, beamwidth)]
                    
                    # Extract recall (x) and QPS (y) values
                    x_values = [point[2] for point in data_points if len(point) >= 3]
                    y_values = [point[0] for point in data_points if len(point) >= 3]
                    
                    if x_values:
                        # Sort by recall
                        sorted_points = sorted(zip(x_values, y_values))
                        x_values_sorted, y_values_sorted = zip(*sorted_points)
                        
                        # Store for speedup calculation
                        if method == 'STATE_SEND':
                            state_send_data[(num_servers, beamwidth)] = list(zip(x_values_sorted, y_values_sorted))
                        elif method == 'SCATTER_GATHER':
                            scatter_gather_data[(num_servers, beamwidth)] = list(zip(x_values_sorted, y_values_sorted))
                        
                        # Get consistent color for method
                        color = METHOD_COLORS.get(method, '#000000')
                        
                        # Get line style and marker for this server config
                        linestyle = SERVER_LINE_STYLES.get(num_servers, '-')
                        marker = SERVER_MARKERS.get(num_servers, 'o')
                        fillstyle = 'none' if num_servers in HOLLOW_MARKERS else 'full'
                        
                        # Plot
                        ax.plot(x_values_sorted, y_values_sorted,
                               marker=marker, linestyle=linestyle,
                               linewidth=2, markersize=8, markeredgewidth=2,
                               fillstyle=fillstyle,
                               color=color,
                               label=f"{method} ({num_servers} srv)")
                        
                        # Track plotted methods and servers
                        if method not in plotted_methods:
                            plotted_methods[method] = []
                        if num_servers not in plotted_methods[method]:
                            plotted_methods[method].append(num_servers)
            
            # Add speedup annotations for STATE_SEND vs SCATTER_GATHER
            # Only for specific recall values: 0.9, 0.95, 0.975
            target_recalls = [0.9, 0.95, 0.975]
            
            for (num_servers, beamwidth) in state_send_data.keys():
                if (num_servers, beamwidth) in scatter_gather_data:
                    state_send_points = state_send_data[(num_servers, beamwidth)]
                    scatter_gather_points = scatter_gather_data[(num_servers, beamwidth)]
                    
                    # Sort points by recall for interpolation
                    ss_sorted = sorted(state_send_points, key=lambda p: p[0])
                    sg_sorted = sorted(scatter_gather_points, key=lambda p: p[0])
                    
                    # For each target recall value, interpolate both STATE_SEND and SCATTER_GATHER
                    for target_recall in target_recalls:
                        ss_qps = None
                        sg_qps = None
                        
                        # Interpolate STATE_SEND QPS at target_recall
                        if len(ss_sorted) >= 2:
                            for i in range(len(ss_sorted) - 1):
                                r1, q1 = ss_sorted[i]
                                r2, q2 = ss_sorted[i + 1]
                                
                                if r1 <= target_recall <= r2:
                                    if r2 != r1:
                                        weight = (target_recall - r1) / (r2 - r1)
                                        ss_qps = q1 + weight * (q2 - q1)
                                    else:
                                        ss_qps = q1
                                    break
                            
                            # Handle edge cases
                            if ss_qps is None:
                                if target_recall < ss_sorted[0][0]:
                                    ss_qps = ss_sorted[0][1]
                                elif target_recall > ss_sorted[-1][0]:
                                    ss_qps = ss_sorted[-1][1]
                        elif len(ss_sorted) == 1:
                            ss_qps = ss_sorted[0][1]
                        
                        # Interpolate SCATTER_GATHER QPS at target_recall
                        if len(sg_sorted) >= 2:
                            for i in range(len(sg_sorted) - 1):
                                r1, q1 = sg_sorted[i]
                                r2, q2 = sg_sorted[i + 1]
                                
                                if r1 <= target_recall <= r2:
                                    if r2 != r1:
                                        weight = (target_recall - r1) / (r2 - r1)
                                        sg_qps = q1 + weight * (q2 - q1)
                                    else:
                                        sg_qps = q1
                                    break
                            
                            # Handle edge cases
                            if sg_qps is None:
                                if target_recall < sg_sorted[0][0]:
                                    sg_qps = sg_sorted[0][1]
                                elif target_recall > sg_sorted[-1][0]:
                                    sg_qps = sg_sorted[-1][1]
                        elif len(sg_sorted) == 1:
                            sg_qps = sg_sorted[0][1]
                        
                        # Calculate and annotate speedup
                        if ss_qps is not None and sg_qps is not None and sg_qps > 0:
                            speedup = ss_qps / sg_qps
                            
                            # Print speedup to console
                            print(f"  {base_name} ({dataset_size}) - {num_servers} server(s) - beamwidth {beamwidth} - Recall@{target_recall}: {speedup:.3f}× speedup (SS: {ss_qps:.2f} QPS, SG: {sg_qps:.2f} QPS)")
            
            # Set title with dataset name (on top row only)
            if row_idx == 0:
                display_name = get_display_name(base_name)
                ax.set_title(display_name, fontsize=14, fontweight='bold')
            
            # Add dataset size label on the left
            if col_idx == 0:
                ax.set_ylabel(f"{dataset_size}\nQPS", fontsize=13, fontweight='bold')
            else:
                ax.set_ylabel('QPS', fontsize=12)
            
            ax.set_xlim(min_recall, 1.01)
            
            # Set consistent x-axis ticks across all plots
            import numpy as np
            if min_recall <= 0.8:
                x_ticks = [0.80, 0.85, 0.90, 0.95, 1.00]
            elif min_recall <= 0.85:
                x_ticks = [0.85, 0.90, 0.95, 1.00]
            elif min_recall <= 0.90:
                x_ticks = [0.90, 0.95, 1.00]
            else:
                x_ticks = [0.95, 1.00]
            ax.set_xticks(x_ticks)
            
            # Set y-axis limits based on the dataset and size
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
            
            ax.grid(True, alpha=0.3)
            
            # Increase tick label size
            ax.tick_params(axis='both', which='major', labelsize=11)
            
            # Add legend - show on first plot of each row
            if col_idx == 0:
                # Create custom legend with better organization
                from matplotlib.lines import Line2D
                legend_elements = []
                
                # First section: Methods (colors)
                method_display_order = ['STATE_SEND', 'SCATTER_GATHER', 'SINGLE_SERVER']
                methods_shown = set()
                for method in method_display_order:
                    if method in plotted_methods:
                        color = METHOD_COLORS.get(method)
                        display_name = LEGEND_NAME_MAPPING.get(method, method)  # Use mapping
                        legend_elements.append(
                            Line2D([0], [0], color=color, linewidth=2, 
                                  linestyle='-', marker='',
                                  label=display_name)  # Use display_name instead of method
                        )
                        methods_shown.add(method)
                
                # Add separator (empty line)
                if methods_shown:
                    legend_elements.append(
                        Line2D([0], [0], color='none', linewidth=0,
                              label='')
                    )
                
                # Second section: Server configurations (line styles and markers)
                all_servers = set()
                for method in plotted_methods:
                    all_servers.update(plotted_methods[method])
                
                for num_servers in sorted(all_servers):
                    linestyle = SERVER_LINE_STYLES.get(num_servers, '-')
                    marker = SERVER_MARKERS.get(num_servers, 'o')
                    fillstyle = 'none' if num_servers in HOLLOW_MARKERS else 'full'
                    legend_elements.append(
                        Line2D([0], [0], color='black', linewidth=2, 
                              linestyle=linestyle, marker=marker, markersize=8,
                              markeredgewidth=2, fillstyle=fillstyle,
                              label=f"{num_servers} server{'s' if num_servers > 1 else ''}")
                    )
                
                ax.legend(handles=legend_elements, fontsize=14, loc='best')
            
            # X-axis label only on bottom row
            if row_idx == num_rows - 1:
                ax.set_xlabel('Recall@10', fontsize=12)
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot QPS vs Recall comparison grid with all server configs in same plot'
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
        default='dataset_comparison_combined.png',
        help='Output filename for the plot (default: dataset_comparison_combined.png)'
    )
    
    args = parser.parse_args()
    
    print(f"Collecting data from: {args.logs_folder}")
    data = collect_data(args.logs_folder)
    
    if not data:
        print("No data collected. Please check your log folder structure.")
        return
    
    # Extract dataset sizes for titles
    dataset_sizes = {}
    for dataset_key in data.keys():
        # Extract size from dataset_key (format: "name_size")
        match = re.match(r'(.+)_(\d+[BKMG])$', dataset_key)
        if match:
            dataset_sizes[dataset_key] = match.group(2)
        else:
            # Fallback: look in folder metadata
            for num_servers in data[dataset_key].keys():
                for (method, beamwidth), _ in data[dataset_key][num_servers].items():
                    # Find any folder with this dataset to get its size
                    logs_root = Path(args.logs_folder)
                    for folder in logs_root.iterdir():
                        if not folder.is_dir():
                            continue
                        metadata = parse_folder_name(folder.name)
                        if metadata:
                            check_key = f"{metadata['dataset_name']}_{metadata['dataset_size']}"
                            if check_key == dataset_key:
                                dataset_sizes[dataset_key] = metadata['dataset_size']
                                break
                    if dataset_key in dataset_sizes:
                        break
                if dataset_key in dataset_sizes:
                    break
    
    print(f"\nFound data for:")
    for dataset_key in sorted(data.keys()):
        print(f"  Dataset: {dataset_key} ({dataset_sizes.get(dataset_key, 'unknown size')})")
        for num_servers in sorted(data[dataset_key].keys()):
            print(f"    {num_servers} server(s): {len(data[dataset_key][num_servers])} configurations")
    
    print(f"\nGenerating plot...")
    fig = plot_comparison_grid(data, args.min_recall, dataset_sizes)
    
    if fig:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {args.output}")
    else:
        print("Failed to generate plot.")


if __name__ == '__main__':
    main()
