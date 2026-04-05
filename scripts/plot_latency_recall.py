#!/usr/bin/env python3
"""
Script to plot Latency vs Recall curves comparing STATE_SEND, SCATTER_GATHER, 
and SINGLE_SERVER performance across different beamwidth values.
All server configurations are combined in a single plot.
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
    'SINGLE_SERVER': 'SingleServer',
    'DISTRIBUTED_ANN': 'DistributedANN'
}

# Define consistent colors for different methods
METHOD_COLORS = {
    'STATE_SEND': '#1f77b4',        # Matplotlib default blue
    'SCATTER_GATHER': '#ff7f0e',    # Matplotlib default orange
    'SINGLE_SERVER': '#2ca02c',     # Matplotlib default green
    'DISTRIBUTED_ANN': 'red',       # Matplotlib default red
}

# Define markers for different methods
METHOD_MARKERS = {
    'STATE_SEND': 'x',        # X marker
    'SCATTER_GATHER': '.',    # Dot marker
    'SINGLE_SERVER': 'o',     # Circle marker
    'DISTRIBUTED_ANN': '+',   # Plus marker
}

# Define line styles for different server counts
SERVER_STYLES = {
    5: ':',     # Dotted line for 5 servers
    10: '-',    # Solid line for 10 servers
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
    elif folder_name.startswith('logs_DISTRIBUTED_ANN'):
        method = 'DISTRIBUTED_ANN'        
    else:
        return None
    
    # Extract beamwidth using regex
    beamwidth_match = re.search(r'BEAMWIDTH_(\d+)', folder_name)
    if not beamwidth_match:
        return None
    beamwidth = int(beamwidth_match.group(1))
    
    # Extract dataset name and size
    if method in ['STATE_SEND', 'SCATTER_GATHER', 'DISTRIBUTED_ANN']:
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
    Parse client.log file and extract QPS, Latency, and Recall data.
    
    Returns a list of tuples: (qps, avg_latency, recall)
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
                # Extract QPS (column 3), Avg Latency (column 4), and Recall (column 9)
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
    """
    data = {}
    single_server_data = {}
    dataset_info = {'name': None, 'size': None}
    distributed_server_counts = set()
    
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
            
            # Store data with (num_servers, method, beamwidth) as key
            num_servers = metadata['num_servers']
            method = metadata['method']
            beamwidth = metadata['beamwidth']
            key = (num_servers, method, beamwidth)
            
            # Use display name in output
            display_name = LEGEND_NAME_MAPPING.get(method, method)
            
            # Store SINGLE_SERVER data separately for now
            if method == 'SINGLE_SERVER':
                single_server_data[key] = data_points
                print(f"  Loaded {len(data_points)} data points from {folder.name} (method={display_name}, beamwidth={beamwidth}, equiv_servers={num_servers})")
            else:
                data[key] = data_points
                distributed_server_counts.add(num_servers)
                print(f"  Loaded {len(data_points)} data points from {folder.name} (method={display_name}, beamwidth={beamwidth}, servers={num_servers})")
    
    # Now merge SINGLE_SERVER data only for server configs that have distributed methods
    for (num_servers, method, beamwidth), value in single_server_data.items():
        if num_servers in distributed_server_counts:
            data[(num_servers, method, beamwidth)] = value
            print(f"Including SingleServer data for {num_servers} servers (has distributed methods)")
    
    return data, dataset_info


def get_latency_at_target_recall(data_points, target_recall=0.95):
    """
    Linearly interpolates latency for a specific target recall.
    Assumes data_points is a list of tuples: (qps, avg_latency, recall)
    """
    if not data_points:
        return "No data"

    # Extract just recall and latency, and sort by recall
    points = sorted([(pt[2], pt[1]) for pt in data_points])
    
    # Check boundaries
    min_recall = points[0][0]
    max_recall = points[-1][0]
    
    if target_recall < min_recall:
        return f"< {points[0][1]:.2f} μs (min recall reached: {min_recall:.4f})"
    if target_recall > max_recall:
        return f"> {points[-1][1]:.2f} μs (max recall reached: {max_recall:.4f})"
        
    # Find exact match or interpolate
    for i in range(len(points) - 1):
        r1, l1 = points[i]
        r2, l2 = points[i+1]
        
        if r1 == target_recall:
            return f"{l1:.2f} μs"
            
        if r1 < target_recall < r2:
            # Linear interpolation
            interpolated_latency = l1 + (l2 - l1) * ((target_recall - r1) / (r2 - r1))
            return f"{interpolated_latency:.2f} μs"
            
    # Handle the case where the very last point is an exact match
    if points[-1][0] == target_recall:
        return f"{points[-1][1]:.2f} μs"
        
    return "N/A"


def plot_lat_acc(data, dataset_info, min_recall):
    """
    Plot latency vs accuracy combining all server configurations in a single plot.
    """
    if not data:
        print("No data to plot!")
        return None
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    # Track what we've plotted for custom legend
    plotted_beamwidths = set()
    plotted_methods = set()
    plotted_servers = set()
    
    # Sort keys by beamwidth, then server count, then method
    method_order = ['STATE_SEND', 'SCATTER_GATHER', 'SINGLE_SERVER', "DISTRIBUTED_ANN"]
    sorted_keys = sorted(data.keys(), 
                         key=lambda x: (x[2], x[0], method_order.index(x[1]) if x[1] in method_order else 999))
    
    for (num_servers, method, beamwidth) in sorted_keys:
        data_points = data[(num_servers, method, beamwidth)]
        
        # Extract recall (x) and latency (y) values
        x_values = [point[2] for point in data_points if len(point) >= 3]  # recall is at index 2
        y_values = [point[1] for point in data_points if len(point) >= 3]  # avg_latency is at index 1
        
        if x_values:
            # Sort by recall
            sorted_points = sorted(zip(x_values, y_values))
            x_values_sorted, y_values_sorted = zip(*sorted_points)
            
            # Get styling
            color = METHOD_COLORS.get(method, '#000000')
            linestyle = SERVER_STYLES.get(num_servers, '-')
            marker = METHOD_MARKERS.get(method, 'o')
            
            # Plot
            ax.plot(x_values_sorted, y_values_sorted, 
                   marker=marker, linestyle=linestyle, 
                   linewidth=2, markersize=6, color=color)
            
            plotted_beamwidths.add(beamwidth)
            plotted_methods.add(method)
            plotted_servers.add(num_servers)
        else:
            display_name = LEGEND_NAME_MAPPING.get(method, method)
            print(f"No valid points for {display_name} beamwidth {beamwidth} with {num_servers} servers")
    
    # Create custom legend with sections
    legend_elements = []
    from matplotlib.lines import Line2D
    
    if plotted_methods:
        method_display_order = ['STATE_SEND', 'SCATTER_GATHER', 'SINGLE_SERVER', "DISTRIBUTED_ANN"]
        for method in method_display_order:
            if method in plotted_methods:
                color = METHOD_COLORS.get(method, '#000000')
                marker = METHOD_MARKERS.get(method, 'o')
                display_name = LEGEND_NAME_MAPPING.get(method, method)
                legend_elements.append(Line2D([0], [0], color=color, linewidth=2, 
                                              marker=marker, markersize=8, 
                                              label=display_name))
    
    if plotted_methods and plotted_servers:
        legend_elements.append(Line2D([0], [0], color='none', label=''))
    
    if len(plotted_servers) > 1:
        for num_servers in sorted(plotted_servers):
            linestyle = SERVER_STYLES.get(num_servers, '-')
            legend_elements.append(Line2D([0], [0], color='black', linewidth=2, 
                                         linestyle=linestyle, 
                                         label=f'{num_servers} Server{"s" if num_servers > 1 else ""}'))
    
    # Create title
    if dataset_info['name'] and dataset_info['size']:
        title = f"{dataset_info['name']} ({dataset_info['size']}) - Latency vs Recall"
    else:
        title = 'Latency vs Recall Comparison'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Recall@10', fontsize=12)
    ax.set_ylabel('Avg Latency (μs)', fontsize=12)
    ax.set_xlim(min_recall, 1.0)
    ax.set_ylim(0, 10000)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_elements, fontsize=10, loc='best')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot Latency vs Recall curves comparing STATE_SEND, SCATTER_GATHER, and SINGLE_SERVER performance across different beamwidth values (combined in single plot)'
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
        default='latency_recall_combined.png',
        help='Output filename for the plot (default: latency_recall_combined.png)'
    )
    
    args = parser.parse_args()
    
    print(f"Collecting data from {len(args.logs_folders)} folder(s)")
    data, dataset_info = collect_data(args.logs_folders)
    
    if not data:
        print("No data collected. Please check your log folder structure.")
        return
    
    print(f"\nFound data for {len(data)} configuration(s)")
    print(f"Dataset: {dataset_info['name']} ({dataset_info['size']})")
    
    # Group by server count for summary
    by_server = defaultdict(list)
    for (num_servers, method, beamwidth) in sorted(data.keys()):
        by_server[num_servers].append((method, beamwidth))
    
    # Print configuration summary
    for num_servers in sorted(by_server.keys()):
        print(f"  {num_servers} server(s):")
        for method, beamwidth in sorted(by_server[num_servers]):
            display_name = LEGEND_NAME_MAPPING.get(method, method)
            print(f"    {display_name} with beamwidth={beamwidth}")

    # NEW: Print Latency at 0.95 Recall
    print("\n" + "="*45)
    print("       LATENCY AT 0.95 RECALL")
    print("="*45)
    
    method_order = ['STATE_SEND', 'SCATTER_GATHER', 'SINGLE_SERVER', 'DISTRIBUTED_ANN']
    
    for num_servers in sorted(by_server.keys()):
        print(f"\n[ {num_servers} Server(s) ]")
        
        # Sort methods by standard order so it's easier to read
        server_configs = sorted(by_server[num_servers], 
                                key=lambda x: (method_order.index(x[0]) if x[0] in method_order else 999, x[1]))
        
        for method, beamwidth in server_configs:
            display_name = LEGEND_NAME_MAPPING.get(method, method)
            key = (num_servers, method, beamwidth)
            lat_at_95 = get_latency_at_target_recall(data[key], target_recall=0.95)
            
            # Format nicely
            print(f"  {display_name:<16} | Beamwidth: {beamwidth:<3} | Latency: {lat_at_95}")
    print("="*45 + "\n")

    print(f"Generating plot...")
    fig = plot_lat_acc(data, dataset_info, args.min_recall)
    
    if fig:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {args.output}")
    else:
        print("Failed to generate plot.")

if __name__ == '__main__':
    main()
