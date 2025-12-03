#!/usr/bin/env python3
"""
Script to plot QPS vs Recall and Latency vs Recall curves comparing different beamwidths.
Creates two separate plots showing how beamwidth affects throughput and latency.
All server configurations and methods are combined in each plot.
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

# Define markers for different methods
METHOD_MARKERS = {
    'STATE_SEND': 'x',        # X marker
    'SCATTER_GATHER': '.',    # Dot marker
    'SINGLE_SERVER': 'o',     # Circle marker
}

# Define line styles for different server counts
SERVER_STYLES = {
    1: '-',     # Solid line for 1 server
    2: '--',    # Dashed line for 2 servers
    3: '-.',    # Dash-dot for 3 servers
    4: ':',     # Dotted for 4 servers
    5: '-',     # Solid line for 5 servers
    10: '-',    # Dotted line for 10 servers
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
    if method in ['STATE_SEND', 'SCATTER_GATHER']:
        dataset_match = re.search(r'distributed_([A-Za-z0-9]+)_(\d+[BKMG])', folder_name)
        if not dataset_match:
            return None
        dataset_name_with_size = dataset_match.group(1)
        dataset_size = dataset_match.group(2)
        
        # Remove size suffix from dataset name if present
        dataset_name = re.sub(r'_?\d+[BKMG]$', '', dataset_name_with_size)
        if not dataset_name:
            dataset_name = dataset_name_with_size
        
        # Extract number of servers
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
                recall = float(parts[8])
                
                data_points.append((qps, avg_latency, recall / 100.0))  # Convert recall to 0-1 range
            except (ValueError, IndexError):
                continue
    
    except FileNotFoundError:
        print(f"Warning: File not found: {log_file_path}")
    except Exception as e:
        print(f"Error parsing {log_file_path}: {e}")
    
    return data_points


def collect_data_separately(throughput_folders, latency_folders):
    """
    Collect throughput and latency data from separate folders.
    
    Args:
        throughput_folders: List of paths to folders containing throughput logs
        latency_folders: List of paths to folders containing latency logs
    
    Returns a tuple: (data_dict, dataset_info)
    where data_dict is {(num_servers, method, beamwidth): [(qps, avg_latency, recall), ...]}
    and dataset_info is {'name': str, 'size': str}
    
    The data from throughput logs provides QPS and recall, while latency logs provide
    latency and recall. They are merged based on matching (num_servers, method, beamwidth).
    """
    throughput_data = {}
    latency_data = {}
    dataset_info = {'name': None, 'size': None}
    
    # Collect throughput data
    print("=" * 60)
    print("COLLECTING THROUGHPUT DATA")
    print("=" * 60)
    for logs_root_folder in throughput_folders:
        logs_root = Path(logs_root_folder)
        
        if not logs_root.exists():
            print(f"Warning: Root folder '{logs_root_folder}' does not exist, skipping")
            continue
        
        print(f"Scanning folder: {logs_root_folder}")
        
        for folder in logs_root.iterdir():
            if not folder.is_dir():
                continue
            
            metadata = parse_folder_name(folder.name)
            if metadata is None:
                print(f"  Skipping folder (couldn't parse): {folder.name}")
                continue
            
            if dataset_info['name'] is None and metadata['dataset_name']:
                dataset_info['name'] = metadata['dataset_name']
                dataset_info['size'] = metadata['dataset_size']
            
            client_log = folder / 'client.log'
            if not client_log.exists():
                print(f"  Warning: client.log not found in {folder.name}")
                continue
            
            data_points = parse_client_log(client_log)
            if not data_points:
                print(f"  Warning: No data extracted from {client_log}")
                continue
            
            key = (metadata['num_servers'], metadata['method'], metadata['beamwidth'])
            throughput_data[key] = data_points
            print(f"  Loaded {len(data_points)} throughput data points from {folder.name} "
                  f"(method={metadata['method']}, beamwidth={metadata['beamwidth']}, servers={metadata['num_servers']})")
    
    # Collect latency data
    print("\n" + "=" * 60)
    print("COLLECTING LATENCY DATA")
    print("=" * 60)
    for logs_root_folder in latency_folders:
        logs_root = Path(logs_root_folder)
        
        if not logs_root.exists():
            print(f"Warning: Root folder '{logs_root_folder}' does not exist, skipping")
            continue
        
        print(f"Scanning folder: {logs_root_folder}")
        
        for folder in logs_root.iterdir():
            if not folder.is_dir():
                continue
            
            metadata = parse_folder_name(folder.name)
            if metadata is None:
                print(f"  Skipping folder (couldn't parse): {folder.name}")
                continue
            
            if dataset_info['name'] is None and metadata['dataset_name']:
                dataset_info['name'] = metadata['dataset_name']
                dataset_info['size'] = metadata['dataset_size']
            
            client_log = folder / 'client.log'
            if not client_log.exists():
                print(f"  Warning: client.log not found in {folder.name}")
                continue
            
            data_points = parse_client_log(client_log)
            if not data_points:
                print(f"  Warning: No data extracted from {client_log}")
                continue
            
            key = (metadata['num_servers'], metadata['method'], metadata['beamwidth'])
            latency_data[key] = data_points
            print(f"  Loaded {len(data_points)} latency data points from {folder.name} "
                  f"(method={metadata['method']}, beamwidth={metadata['beamwidth']}, servers={metadata['num_servers']})")
    
    # Merge the data - use the union of keys from both datasets
    all_keys = set(throughput_data.keys()) | set(latency_data.keys())
    merged_data = {}
    
    print("\n" + "=" * 60)
    print("MERGING DATA")
    print("=" * 60)
    for key in all_keys:
        if key in throughput_data and key in latency_data:
            # Both datasets have this configuration - use data from respective sources
            merged_data[key] = {
                'throughput': throughput_data[key],
                'latency': latency_data[key]
            }
            print(f"  Merged data for {key}: throughput={len(throughput_data[key])} points, latency={len(latency_data[key])} points")
        elif key in throughput_data:
            # Only throughput data available
            merged_data[key] = {
                'throughput': throughput_data[key],
                'latency': None
            }
            print(f"  Only throughput data for {key}: {len(throughput_data[key])} points")
        else:
            # Only latency data available
            merged_data[key] = {
                'throughput': None,
                'latency': latency_data[key]
            }
            print(f"  Only latency data for {key}: {len(latency_data[key])} points")
    
    return merged_data, dataset_info


def plot_combined(data, dataset_info, min_recall):
    """
    Plot both QPS vs Recall and Latency vs Recall side by side.
    
    data: {(num_servers, method, beamwidth): {'throughput': [...], 'latency': [...]}}
    dataset_info: {'name': str, 'size': str}
    
    Legend scheme:
    - Color = Beamwidth (blue=1, orange=2, green=4, red=8, purple=16, brown=32)
    - Line style = Server count (solid, dashed, dotted, etc.)
    - Marker = Method (x=STATE_SEND, .=SCATTER_GATHER, o=SINGLE_SERVER)
    """
    if not data:
        print("No data to plot!")
        return None
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Track what we've plotted for custom legend
    plotted_beamwidths = set()
    plotted_methods = set()
    plotted_servers = set()
    
    # Sort keys by beamwidth, then server count, then method
    method_order = ['STATE_SEND', 'SCATTER_GATHER', 'SINGLE_SERVER']
    sorted_keys = sorted(data.keys(), 
                        key=lambda x: (x[2], x[0], method_order.index(x[1]) if x[1] in method_order else 999))
    
    # Plot both QPS and Latency
    for (num_servers, method, beamwidth) in sorted_keys:
        data_entry = data[(num_servers, method, beamwidth)]
        
        # Get styling
        color = BEAMWIDTH_COLORS.get(beamwidth, '#000000')
        linestyle = SERVER_STYLES.get(num_servers, '-')
        marker = METHOD_MARKERS.get(method, 'o')
        
        # Plot 1: QPS vs Recall (from throughput data)
        if data_entry['throughput'] is not None:
            throughput_points = data_entry['throughput']
            qps_values = [point[0] for point in throughput_points if len(point) >= 3]
            recall_values = [point[2] for point in throughput_points if len(point) >= 3]
            
            if recall_values and qps_values:
                sorted_points_qps = sorted(zip(recall_values, qps_values))
                x_qps, y_qps = zip(*sorted_points_qps)
                ax1.plot(x_qps, y_qps, 
                        marker=marker, linestyle=linestyle, 
                        linewidth=2, markersize=6, color=color)
                plotted_beamwidths.add(beamwidth)
                plotted_methods.add(method)
                plotted_servers.add(num_servers)
        
        # Plot 2: Latency vs Recall (from latency data)
        if data_entry['latency'] is not None:
            latency_points = data_entry['latency']
            latency_values = [point[1] for point in latency_points if len(point) >= 3]
            recall_values = [point[2] for point in latency_points if len(point) >= 3]
            
            if latency_values and recall_values:
                sorted_points_lat = sorted(zip(latency_values, recall_values))
                x_lat, y_lat = zip(*sorted_points_lat)
                ax2.plot(x_lat, y_lat, 
                        marker=marker, linestyle=linestyle, 
                        linewidth=2, markersize=6, color=color)
                plotted_beamwidths.add(beamwidth)
                plotted_methods.add(method)
                plotted_servers.add(num_servers)
    
    # Create custom legend with sections
    legend_elements = []
    from matplotlib.lines import Line2D
    
    # Section 1: Beamwidth (colors)
    for bw in sorted(plotted_beamwidths):
        color = BEAMWIDTH_COLORS.get(bw)
        legend_elements.append(Line2D([0], [0], color=color, linewidth=4, 
                                     label=f'BW={bw}'))
    
    # Add a separator (invisible line)
    if plotted_servers or plotted_methods:
        legend_elements.append(Line2D([0], [0], color='none', label=''))
    
    # Section 2: Server count (line styles)
    if len(plotted_servers) > 1:
        for num_servers in sorted(plotted_servers):
            linestyle = SERVER_STYLES.get(num_servers, '-')
            legend_elements.append(Line2D([0], [0], color='black', linewidth=2, 
                                         linestyle=linestyle, 
                                         label=f'{num_servers} Server{"s" if num_servers > 1 else ""}'))
        
        # Add another separator
        if plotted_methods:
            legend_elements.append(Line2D([0], [0], color='none', label=''))
    
    # Section 3: Methods (markers)
    if len(plotted_methods) > 1:
        method_display_order = ['STATE_SEND', 'SCATTER_GATHER', 'SINGLE_SERVER']
        for method in method_display_order:
            if method in plotted_methods:
                marker = METHOD_MARKERS.get(method, 'o')
                legend_elements.append(Line2D([0], [0], color='black', linewidth=0, 
                                             marker=marker, markersize=8, 
                                             label=method))
    
    # Configure left plot (QPS vs Recall)
    ax1.set_title('Throughput vs Recall', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Recall@10', fontsize=16)
    ax1.set_ylabel('QPS', fontsize=16)
    ax1.set_xlim(min_recall, 1.01)
    ax1.grid(True, alpha=0.3)
    ax1.legend(handles=legend_elements, fontsize=16, loc='best')
    ax1.tick_params(axis='y', labelsize=18)
    ax1.tick_params(axis='x', labelsize=16)
    # Make x-ticks more sparse
    import numpy as np
    if min_recall <= 0.8:
        ax1.set_xticks([0.80, 0.90, 1.00])
    elif min_recall <= 0.85:
        ax1.set_xticks([0.85, 0.90, 0.95, 1.00])
    elif min_recall <= 0.90:
        ax1.set_xticks([0.90, 0.95, 1.00])
    else:
        ax1.set_xticks([0.95, 1.00])
    # Reduce number of y-ticks
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
    
    # Configure right plot (Latency vs Recall)
    ax2.set_title('Recall vs Latency', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Avg Latency (μs)', fontsize=16)
    ax2.set_ylabel('Recall@10', fontsize=16)
    ax2.set_ylim(min_recall, 1.01)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelsize=18)
    ax2.tick_params(axis='x', labelsize=16)
    # Make y-ticks more sparse
    if min_recall <= 0.8:
        ax2.set_yticks([0.80, 0.90, 1.00])
    elif min_recall <= 0.85:
        ax2.set_yticks([0.85, 0.90, 0.95, 1.00])
    elif min_recall <= 0.90:
        ax2.set_yticks([0.90, 0.95, 1.00])
    else:
        ax2.set_yticks([0.95, 1.00])
    # Reduce number of x-ticks
    ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
    
    # Add overall title with dataset info
    # if dataset_info['name'] and dataset_info['size']:
    #     fig.suptitle(f"{dataset_info['name']} ({dataset_info['size']}) - Beamwidth Comparison", 
    #                 fontsize=16, fontweight='bold', y=0.98)
    # else:
    #     fig.suptitle('Beamwidth Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot QPS vs Recall and Latency vs Recall curves comparing different beamwidths (combined in single PNG)'
    )
    parser.add_argument(
        'logs_folder',
        type=str,
        help='Path to the root folder containing "throughput" and "latency" subfolders'
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
        default='beamwidth_comparison.png',
        help='Output filename for the combined plot (default: beamwidth_comparison.png)'
    )
    
    args = parser.parse_args()
    
    # Construct paths to throughput and latency subfolders
    root_path = Path(args.logs_folder)
    throughput_folder = root_path / 'throughput'
    latency_folder = root_path / 'latency'
    
    # Check if folders exist
    if not throughput_folder.exists():
        print(f"Error: Throughput folder not found: {throughput_folder}")
        return
    if not latency_folder.exists():
        print(f"Error: Latency folder not found: {latency_folder}")
        return
    
    print(f"Collecting data from:")
    print(f"  Throughput folder: {throughput_folder}")
    print(f"  Latency folder: {latency_folder}")
    
    data, dataset_info = collect_data_separately([str(throughput_folder)], [str(latency_folder)])
    
    if not data:
        print("No data collected. Please check your log folder structure.")
        return
    
    print(f"\nFound data for {len(data)} configuration(s)")
    print(f"Dataset: {dataset_info['name']} ({dataset_info['size']})")
    
    # Group by server count for summary
    by_server = defaultdict(list)
    for (num_servers, method, beamwidth) in sorted(data.keys()):
        by_server[num_servers].append((method, beamwidth))
    
    for num_servers in sorted(by_server.keys()):
        print(f"  {num_servers} server(s):")
        for method, beamwidth in sorted(by_server[num_servers]):
            has_throughput = data[(num_servers, method, beamwidth)]['throughput'] is not None
            has_latency = data[(num_servers, method, beamwidth)]['latency'] is not None
            status = []
            if has_throughput:
                status.append("throughput")
            if has_latency:
                status.append("latency")
            print(f"    {method} with beamwidth={beamwidth} ({', '.join(status)})")
    
    # Generate combined plot
    print(f"\nGenerating combined throughput and latency plot...")
    fig = plot_combined(data, dataset_info, args.min_recall)
    
    if fig:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {args.output}")
    else:
        print("Failed to generate plot.")


if __name__ == '__main__':
    main()
