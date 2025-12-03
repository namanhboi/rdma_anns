#!/usr/bin/env python3
"""
Script to plot QPS vs Recall curves comparing STATE_SEND, SCATTER_GATHER, 
and SINGLE_SERVER performance across different beamwidth values.
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Define consistent colors for different beamwidths
BEAMWIDTH_COLORS = {
    1: '#1f77b4',   # Blue
    2: '#ff7f0e',   # Orange
    4: '#2ca02c',   # Green
    8: '#d62728',   # Red
    16: '#9467bd',  # Purple
    32: '#8c564b',  # Brown
}

# Define line styles for different methods
METHOD_STYLES = {
    'STATE_SEND': '-',        # Solid line
    'SCATTER_GATHER': '--',   # Dashed line
    'SINGLE_SERVER': ':',     # Dotted line
}


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
    else:  # SINGLE_SERVER
        dataset_match = re.search(r'logs_SINGLE_SERVER_(\w+)_(\d+[BKMG])', folder_name)
        if not dataset_match:
            return None
        dataset_name = dataset_match.group(1)
        dataset_size = dataset_match.group(2)
        
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


def collect_data(logs_folders):
    """
    Collect all data from log folders.
    
    Args:
        logs_folders: List of paths to root folders containing log subfolders
    
    Returns a tuple: (data_dict, dataset_info)
    where data_dict is {num_servers: {(method, beamwidth): [(qps, latency, recall), ...]}}
    and dataset_info is {'name': str, 'size': str}
    
    Note: SINGLE_SERVER data is only included for server configurations that also have
    STATE_SEND or SCATTER_GATHER data.
    """
    data = defaultdict(lambda: defaultdict(list))
    single_server_data = defaultdict(lambda: defaultdict(list))  # Store SINGLE_SERVER separately
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
            
            # Store data with (method, beamwidth) as key
            num_servers = metadata['num_servers']
            method = metadata['method']
            beamwidth = metadata['beamwidth']
            
            # Store SINGLE_SERVER data separately for now
            if method == 'SINGLE_SERVER':
                single_server_data[num_servers][(method, beamwidth)] = data_points
                print(f"  Loaded {len(data_points)} data points from {folder.name} (method={method}, beamwidth={beamwidth}, equiv_servers={num_servers})")
            else:
                data[num_servers][(method, beamwidth)] = data_points
                print(f"  Loaded {len(data_points)} data points from {folder.name} (method={method}, beamwidth={beamwidth}, servers={num_servers})")
    
    # Now merge SINGLE_SERVER data only for server configs that have distributed methods
    for num_servers in list(single_server_data.keys()):
        if num_servers in data:
            # This server config has distributed methods, so include SINGLE_SERVER data
            for key, value in single_server_data[num_servers].items():
                data[num_servers][key] = value
            print(f"Including SINGLE_SERVER data for {num_servers} servers (has distributed methods)")
        else:
            print(f"Excluding SINGLE_SERVER data for {num_servers} servers (no distributed methods)")
    
    return data, dataset_info


def plot_tput_acc(data, dataset_info, min_recall):
    """
    Plot throughput vs accuracy for different server configurations, methods, and beamwidths.
    
    data: {num_servers: {(method, beamwidth): [(qps, latency, recall), ...]}}
    dataset_info: {'name': str, 'size': str}
    
    Legend scheme:
    - Color = Beamwidth (blue=1, orange=2, green=4, red=8)
    - Line style = Method (solid=STATE_SEND, dashed=SCATTER_GATHER, dotted=SINGLE_SERVER)
    """
    num_configs = len(data)
    if num_configs == 0:
        print("No data to plot!")
        return None
    
    fig, axes = plt.subplots(nrows=1, ncols=num_configs, 
                            figsize=(6 * num_configs, 4.5), squeeze=False)
    axes_flat = axes.flatten()
    
    # Sort by number of servers
    sorted_configs = sorted(data.items())
    
    # First pass: find global min/max QPS values for aligned y-axis
    all_qps_values = []
    for num_servers, methods_data in sorted_configs:
        for (method, beamwidth), data_points in methods_data.items():
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
    for i, (num_servers, methods_data) in enumerate(sorted_configs):
        ax = axes_flat[i]
        
        # Track what we've plotted for custom legend
        plotted_beamwidths = set()
        plotted_methods = set()
        
        # Group by method and beamwidth, then sort
        # Sort by beamwidth first, then by method (STATE_SEND, SCATTER_GATHER, SINGLE_SERVER)
        method_order = ['STATE_SEND', 'SCATTER_GATHER', 'SINGLE_SERVER']
        sorted_keys = sorted(methods_data.keys(), 
                           key=lambda x: (x[1], method_order.index(x[0]) if x[0] in method_order else 999))
        
        for (method, beamwidth) in sorted_keys:
            data_points = methods_data[(method, beamwidth)]
            
            # Extract recall (x) and QPS (y) values
            x_values = [point[2] for point in data_points if len(point) >= 3]
            y_values = [point[0] for point in data_points if len(point) >= 3]
            
            if x_values:
                # Sort by recall
                sorted_points = sorted(zip(x_values, y_values))
                x_values_sorted, y_values_sorted = zip(*sorted_points)
                
                # Get consistent color for this beamwidth and line style for this method
                color = BEAMWIDTH_COLORS.get(beamwidth, None)
                linestyle = METHOD_STYLES.get(method, '-')
                
                # Plot without label - we'll create custom legend
                ax.plot(x_values_sorted, y_values_sorted, 
                       marker='o', linestyle=linestyle, 
                       linewidth=2, markersize=5, color=color)
                
                plotted_beamwidths.add(beamwidth)
                plotted_methods.add(method)
            else:
                print(f"No valid points for {method} beamwidth {beamwidth} with {num_servers} servers")
        
        # Create custom legend with two sections
        legend_elements = []
        
        # Section 1: Beamwidth (colors)
        if plotted_beamwidths:
            for bw in sorted(plotted_beamwidths):
                color = BEAMWIDTH_COLORS.get(bw)
                from matplotlib.lines import Line2D
                legend_elements.append(Line2D([0], [0], color=color, linewidth=2, 
                                             label=f'BW={bw}'))
        
        # Add a separator (invisible line)
        if plotted_beamwidths and plotted_methods:
            legend_elements.append(Line2D([0], [0], color='none', label=''))
        
        # Section 2: Methods (line styles)
        if plotted_methods:
            method_display_order = ['STATE_SEND', 'SCATTER_GATHER', 'SINGLE_SERVER']
            for method in method_display_order:
                if method in plotted_methods:
                    linestyle = METHOD_STYLES.get(method)
                    from matplotlib.lines import Line2D
                    legend_elements.append(Line2D([0], [0], color='black', linewidth=2, 
                                                 linestyle=linestyle, label=method))
        
        # Create title with dataset info and number of servers
        if dataset_info['name'] and dataset_info['size']:
            title = f"{dataset_info['name']} ({dataset_info['size']}) - {num_servers} Server{'s' if num_servers > 1 else ''}"
        else:
            title = f'{num_servers} Server{"s" if num_servers > 1 else ""}'
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Recall@10', fontsize=11)
        ax.set_xlim(min_recall, 1.01)
        ax.set_ylabel('Throughput (QPS)', fontsize=11)
        
        # Set aligned y-axis limits
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        
        ax.grid(True, alpha=0.3)
        ax.legend(handles=legend_elements, fontsize=9, loc='best')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot QPS vs Recall curves comparing STATE_SEND, SCATTER_GATHER, and SINGLE_SERVER performance across different beamwidth values'
    )
    parser.add_argument(
        'logs_folders',
        type=str,
        nargs='+',
        help='Path(s) to the root folder(s) containing log subfolders. Can specify multiple folders for different methods.'
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
        default='method_beamwidth_comparison.png',
        help='Output filename for the plot (default: method_beamwidth_comparison.png)'
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
            print(f"    {method} with beamwidth={beamwidth}")
    
    print(f"\nGenerating plot...")
    fig = plot_tput_acc(data, dataset_info, args.min_recall)
    
    if fig:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {args.output}")
    else:
        print("Failed to generate plot.")

if __name__ == '__main__':
    main()
