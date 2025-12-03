#!/usr/bin/env python3
"""
Script to plot QPS vs Recall curves from individual log files.
Input folder contains [method_name].log files with QPS and recall data.
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Use matplotlib's default color cycle


def parse_log_file(log_file_path):
    """
    Parse log file and extract QPS and Recall data.
    
    Expected format (similar to client.log):
    - Header line containing 'QPS' and 'Recall'
    - Data lines with whitespace-separated values
    
    Returns a list of tuples: (qps, recall)
    """
    data_points = []
    
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
        
        # Find the header line
        header_found = False
        data_start_idx = 0
        
        for i, line in enumerate(lines):
            if 'QPS' in line and 'Recall' in line:
                header_found = True
                data_start_idx = i + 2  # Skip header and separator line
                break
        
        if not header_found:
            print(f"Warning: Header not found in {log_file_path}")
            # Try to parse as simple two-column format
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        qps = float(parts[0])
                        recall = float(parts[1])
                        # Convert recall to 0-1 range if it's in percentage
                        if recall > 1.0:
                            recall = recall / 100.0
                        data_points.append((qps, recall))
                    except ValueError:
                        continue
            return data_points
        
        # Parse data lines
        for line in lines[data_start_idx:]:
            line = line.strip()
            if not line:
                continue
            
            # Split by whitespace and extract relevant columns
            parts = line.split()
            if len(parts) < 7:  # Need at least 7 columns based on the format
                continue
            
            try:
                # Column indices: L(0), I/O Width(1), QPS(2), AvgLat(3), P99 Lat(4), Mean Hops(5), Mean IOs(6), Recall@10(7)
                qps = float(parts[2])
                recall = float(parts[-1])  # Last column is Recall@10
                
                # Convert recall to 0-1 range if it's in percentage
                if recall > 1.0:
                    recall = recall / 100.0
                
                data_points.append((qps, recall))
            except (ValueError, IndexError) as e:
                continue
    
    except FileNotFoundError:
        print(f"Warning: File not found: {log_file_path}")
    except Exception as e:
        print(f"Error parsing {log_file_path}: {e}")
    
    return data_points


def extract_method_name(filename):
    """
    Extract method name from filename by removing .log extension.
    
    Args:
        filename: Name of the log file (e.g., "STATE_SEND.log")
    
    Returns:
        Method name (e.g., "STATE_SEND")
    """
    # Remove .log extension
    name = filename
    if name.endswith('.log'):
        name = name[:-4]
    
    return name


def collect_data(logs_folder):
    """
    Collect all data from log files in the specified folder.
    
    Args:
        logs_folder: Path to folder containing [method_name].log files
    
    Returns a dict: 
    {
        method_name: [(qps, recall), ...]
    }
    """
    data = {}
    
    logs_root = Path(logs_folder)
    
    if not logs_root.exists():
        print(f"Error: Folder '{logs_folder}' does not exist")
        return data
    
    print(f"Scanning folder: {logs_folder}")
    
    # Iterate through all .log files in the directory
    for log_file in logs_root.glob('*.log'):
        method_name = extract_method_name(log_file.name)
        
        # Parse the log file
        data_points = parse_log_file(log_file)
        
        if not data_points:
            print(f"  Warning: No data extracted from {log_file.name}")
            continue
        
        data[method_name] = data_points
        print(f"  Loaded {len(data_points)} data points from {log_file.name} (method={method_name})")
    
    return data


def plot_qps_recall(data, min_recall, output_file, title=None):
    """
    Plot QPS vs Recall curves for all methods.
    
    Args:
        data: {method_name: [(qps, recall), ...]}
        min_recall: Minimum recall value for x-axis
        output_file: Path to save the output plot
        title: Optional plot title
    
    Returns:
        matplotlib figure
    """
    if not data:
        print("No data to plot!")
        return None
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Plot each method (matplotlib will use default color cycle)
    for method in sorted(data.keys()):
        data_points = data[method]
        
        # Sort by recall
        sorted_points = sorted(data_points, key=lambda p: p[1])
        qps_values = [p[0] for p in sorted_points]
        recall_values = [p[1] for p in sorted_points]
        
        # Plot
        ax.plot(recall_values, qps_values,
               marker='o', linestyle='-',
               linewidth=2, markersize=6,
               label=method)
        
        print(f"  Plotted {method}: {len(data_points)} points, "
              f"Recall range: [{min(recall_values):.3f}, {max(recall_values):.3f}], "
              f"QPS range: [{min(qps_values):.1f}, {max(qps_values):.1f}]")
    
    # Set labels and title
    ax.set_xlabel('Recall@10', fontsize=14)
    ax.set_ylabel('QPS', fontsize=14)
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')
    else:
        ax.set_title('BIGANN 100M', fontsize=10, fontweight='bold')
    
    # Set x-axis limits
    ax.set_xlim(min_recall, 1.01)
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot QPS vs Recall curves from individual log files'
    )
    parser.add_argument(
        'logs_folder',
        type=str,
        help='Path to the folder containing [method_name].log files'
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
        default='qps_recall_comparison.png',
        help='Output filename for the plot (default: qps_recall_comparison.png)'
    )
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Optional title for the plot'
    )
    
    args = parser.parse_args()
    
    print(f"Collecting data from: {args.logs_folder}")
    data = collect_data(args.logs_folder)
    
    if not data:
        print("No data collected. Please check your log folder and file format.")
        return
    
    print(f"\nFound data for {len(data)} method(s):")
    for method in sorted(data.keys()):
        print(f"  - {method}: {len(data[method])} data points")
    
    print(f"\nGenerating plot...")
    fig = plot_qps_recall(data, args.min_recall, args.output, args.title)
    
    if fig:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {args.output}")
    else:
        print("Failed to generate plot.")


if __name__ == '__main__':
    main()
