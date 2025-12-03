#!/usr/bin/env python3
"""
Script to plot scalability of STATE_SEND with and without overlap
as the number of servers increases.
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Define consistent colors and markers for overlap vs non-overlap
METHOD_STYLES = {
    'OVERLAP_true': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'label': 'STATE_SEND (Overlap)'},
    'OVERLAP_false': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-', 'label': 'STATE_SEND (No Overlap)'},
}


def parse_folder_name(folder_name):
    """
    Parse the log folder name to extract metadata.
    Expected format:
    - logs_STATE_SEND_distributed_bigann_100M_2_10_MS_NUM_SEARCH_THREADS_8_MAX_BATCH_SIZE_8_K_10_OVERLAP_true_LVEC_...
    - logs_STATE_SEND_distributed_bigann_100M_2_10_MS_NUM_SEARCH_THREADS_8_MAX_BATCH_SIZE_8_K_10_OVERLAP_false_LVEC_...
    
    Returns a dict with extracted fields or None if parsing fails.
    """
    # Only process folders that start with 'logs_STATE_SEND'
    if not folder_name.startswith('logs_STATE_SEND'):
        return None
    
    # Split by underscore
    parts = folder_name.split('_')
    
    try:
        # Find number of servers (should be after 100M and before 10)
        num_servers = None
        for i in range(len(parts)):
            if '100M' in parts[i] and i + 1 < len(parts):
                # Next part should be the number of servers
                if parts[i + 1].isdigit() and len(parts[i + 1]) <= 2:
                    num_servers = int(parts[i + 1])
                    break
        
        if num_servers is None:
            return None
        
        # Find OVERLAP value
        overlap = None
        for i in range(len(parts)):
            if parts[i] == 'OVERLAP' and i + 1 < len(parts):
                overlap_str = parts[i + 1].lower()
                if overlap_str in ['true', 'false']:
                    overlap = overlap_str == 'true'
                    break
        
        if overlap is None:
            return None
        
        return {
            'num_servers': num_servers,
            'overlap': overlap,
            'full_name': folder_name
        }
    
    except (IndexError, ValueError) as e:
        print(f"Error parsing {folder_name}: {e}")
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
            if len(parts) < 8:
                continue
            
            try:
                # Extract QPS (column 3), and Recall (column 8)
                qps = float(parts[2])
                avg_latency = float(parts[3])
                recall = float(parts[7])
                
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
    Get QPS at the closest recall value to target_recall.
    
    Returns QPS value or None if no data available.
    """
    if not data_points:
        return None
    
    # Find the data point with recall closest to target_recall
    closest_point = min(data_points, key=lambda x: abs(x[2] - target_recall))
    
    # Only return if the recall is reasonably close (within 0.05)
    if abs(closest_point[2] - target_recall) <= 0.05:
        return closest_point[0]
    
    return None


def collect_data(logs_root_folder, target_recall=0.95):
    """
    Collect QPS at target recall for overlap and non-overlap configurations.
    
    Returns a dict: {'OVERLAP_true': {num_servers: qps}, 'OVERLAP_false': {num_servers: qps}}
    """
    data = {
        'OVERLAP_true': {},
        'OVERLAP_false': {}
    }
    
    logs_root = Path(logs_root_folder)
    
    if not logs_root.exists():
        print(f"Error: Root folder '{logs_root_folder}' does not exist")
        return data
    
    # Iterate through all subdirectories
    for folder in logs_root.iterdir():
        if not folder.is_dir():
            continue
        
        # Parse folder name
        metadata = parse_folder_name(folder.name)
        if metadata is None:
            continue
        
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
        
        num_servers = metadata['num_servers']
        overlap = metadata['overlap']
        
        # Store data
        key = f"OVERLAP_{str(overlap).lower()}"
        data[key][num_servers] = qps
        
        print(f"Loaded {folder.name}: num_servers={num_servers}, overlap={overlap}, QPS@{target_recall}={qps:.2f}")
    
    return data


def plot_overlap_comparison(data, target_recall):
    """
    Plot scalability comparison between overlap and non-overlap STATE_SEND.
    
    data: {'OVERLAP_true': {num_servers: qps}, 'OVERLAP_false': {num_servers: qps}}
    """
    if not data['OVERLAP_true'] and not data['OVERLAP_false']:
        print("No data to plot!")
        return None
    
    # Get baseline QPS from 2 servers with no overlap
    baseline_qps = None
    if 2 in data['OVERLAP_false']:
        baseline_qps = data['OVERLAP_false'][2]
        print(f"\nBaseline QPS (2 servers, no overlap): {baseline_qps:.2f}")
    elif data['OVERLAP_false']:
        # Use minimum number of servers as baseline
        min_servers = min(data['OVERLAP_false'].keys())
        baseline_qps = data['OVERLAP_false'][min_servers]
        print(f"\nBaseline QPS ({min_servers} servers, no overlap): {baseline_qps:.2f}")
    elif data['OVERLAP_true']:
        # Fallback to overlap data if no non-overlap data
        min_servers = min(data['OVERLAP_true'].keys())
        baseline_qps = data['OVERLAP_true'][min_servers]
        print(f"\nBaseline QPS ({min_servers} servers, with overlap): {baseline_qps:.2f}")
    else:
        print("Cannot plot without baseline data!")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot optimal scalability line (y = x) starting from origin (1, 1)
    max_x = 1
    for key in ['OVERLAP_true', 'OVERLAP_false']:
        if data[key]:
            max_x = max(max_x, max(data[key].keys()))
    
    # Optimal line starting from 1 server
    ax.plot([1, max_x], [1, max_x], 'k--', linewidth=1.5, alpha=0.5, label='Optimal Scalability')
    
    # Plot each configuration
    for key in ['OVERLAP_false', 'OVERLAP_true']:
        if not data[key]:
            continue
        
        style = METHOD_STYLES[key]
        
        x_values = [1]  # Always start from 1 server
        y_values = [1]  # With speedup of 1
        
        for num_servers, qps in sorted(data[key].items()):
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
    
    ax.set_xlabel('Number of Servers', fontsize=13, fontweight='bold')
    ax.set_ylabel('Speedup (relative to baseline)', fontsize=13, fontweight='bold')
    ax.set_title(f'STATE_SEND: Overlap vs No Overlap (Recall={target_recall})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, loc='upper left')
    
    # Set x-axis to show integer values
    all_x = []
    for key in ['OVERLAP_true', 'OVERLAP_false']:
        if data[key]:
            all_x.extend(data[key].keys())
    if all_x:
        min_x = int(min(all_x))
        max_x = int(max(all_x)) + 1
        ax.set_xticks(range(min_x, max_x))
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot scalability of STATE_SEND with and without overlap'
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
        '--output',
        type=str,
        default='overlap_comparison.png',
        help='Output filename for the plot (default: overlap_comparison.png)'
    )
    
    args = parser.parse_args()
    
    print(f"Collecting data from: {args.logs_folder}")
    print(f"Target recall: {args.target_recall}")
    
    data = collect_data(args.logs_folder, args.target_recall)
    
    if not data['OVERLAP_true'] and not data['OVERLAP_false']:
        print("No data collected. Please check your log folder structure.")
        return
    
    print(f"\nFound data for:")
    for key, values in data.items():
        if values:
            print(f"  {key}: {sorted(values.keys())} servers")
    
    print(f"\nGenerating plot...")
    fig = plot_overlap_comparison(data, args.target_recall)
    
    if fig:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {args.output}")
    else:
        print("Failed to generate plot.")


if __name__ == '__main__':
    main()
