#!/usr/bin/env python3
"""
Script to plot the relationship between L values and hop metrics:
- Total number of hops vs L
- Inter-partition hops vs L

This script processes multiple result_L_*.csv files and creates a comparison plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import argparse
import sys
import re

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')

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

def parse_partition_history(partition_str):
    """Parse partition history string into list of integers."""
    # Remove brackets and split by comma
    partition_str = partition_str.strip('[]')
    if partition_str:
        partitions = [int(p.strip()) for p in partition_str.split(',') if p.strip()]
        return partitions
    return []

def load_and_process_single_file(file_path):
    """Load and process data from a single CSV file."""
    # Custom parsing to handle partition_history field with commas
    data = []
    with open(file_path, 'r') as f:
        header = f.readline().strip().split(',')
        
        for line in f:
            # Find the first '[' to identify start of partition_history
            first_bracket = line.find('[')
            if first_bracket == -1:
                continue
            
            # Find the second '[' for partition_history_hop_idx
            second_bracket = line.find('[', first_bracket + 1)
            
            # Split the part before first bracket normally
            before_bracket = line[:first_bracket].rstrip(',')
            parts = before_bracket.split(',')
            
            if second_bracket != -1:
                # Extract both partition_history and partition_history_hop_idx
                partition_history = line[first_bracket:second_bracket].rstrip(',')
                partition_history_hop_idx = line[second_bracket:].strip()
                row = parts + [partition_history, partition_history_hop_idx]
            else:
                # Only partition_history exists (for backward compatibility)
                partition_history = line[first_bracket:].strip()
                row = parts + [partition_history]
            
            data.append(row)
    
    df = pd.DataFrame(data, columns=header)
    
    # Convert numeric columns to appropriate types
    df['query_id'] = pd.to_numeric(df['query_id'])
    df['send_timestamp_ns'] = pd.to_numeric(df['send_timestamp_ns'])
    df['receive_timestamp_ns'] = pd.to_numeric(df['receive_timestamp_ns'])
    df['completion_time_us'] = pd.to_numeric(df['completion_time_us'])
    df['n_hops'] = pd.to_numeric(df['n_hops'])
    df['n_ios'] = pd.to_numeric(df['n_ios'])
    df['n_cmps'] = pd.to_numeric(df['n_cmps'])
    
    # Extract L value from filename (e.g., result_L_10.csv -> 10)
    l_value = file_path.stem.split('_')[-1]
    df['L'] = int(l_value)
    
    # Parse partition history
    df['partition_list'] = df['partition_history'].apply(parse_partition_history)
    
    # Calculate inter-partition hops (length of partition history - 1, minimum 0)
    df['inter_partition_hops'] = df['partition_list'].apply(lambda x: max(0, len(x) - 1))
    
    return df

def plot_hops_vs_L(summary_stats, metadata=None):
    """Create a plot showing hops vs L values."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    
    # Extract data
    L_values = summary_stats['L'].values
    
    # Calculate percentage of inter-partition hops
    summary_stats['inter_partition_percentage'] = (
        summary_stats['inter_partition_hops_mean'] / summary_stats['n_hops_mean'] * 100
    )
    
    # Plot total hops
    ax.plot(L_values, summary_stats['n_hops_mean'], 
            marker='o', linewidth=2.5, markersize=8, 
            label='Total Hops (mean)', color='steelblue')
    
    # Plot inter-partition hops
    ax.plot(L_values, summary_stats['inter_partition_hops_mean'], 
            marker='s', linewidth=2.5, markersize=8, 
            label='Inter-Partition Hops (mean)', color='darkorange')
    
    # Formatting
    ax.set_xlabel('L Value', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Hops', fontsize=14, fontweight='bold')
    
    # Build title with metadata
    if metadata:
        title = f'Hop Count vs L Value - {metadata["dataset_name"]} {metadata["dataset_size"]}, {metadata["num_servers"]} Servers, Beamwidth={metadata["beamwidth"]}'
    else:
        title = 'Hop Count vs L Value'
    
    # ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show all L values
    ax.set_xticks(L_values)
    
    # Add value labels and percentage on data points
    for i, L in enumerate(L_values):
        # Total hops label
        ax.annotate(f'{summary_stats["n_hops_mean"].iloc[i]:.1f}', 
                   xy=(L, summary_stats['n_hops_mean'].iloc[i]),
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='steelblue', alpha=0.3))
        
        # Inter-partition hops label
        ax.annotate(f'{summary_stats["inter_partition_hops_mean"].iloc[i]:.1f}', 
                   xy=(L, summary_stats['inter_partition_hops_mean'].iloc[i]),
                   xytext=(0, -15), textcoords='offset points',
                   ha='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='darkorange', alpha=0.3))
        
        # Percentage label - positioned between the two lines
        mid_y = (summary_stats['n_hops_mean'].iloc[i] + summary_stats['inter_partition_hops_mean'].iloc[i]) / 2
        percentage = summary_stats['inter_partition_percentage'].iloc[i]
        ax.annotate(f'{percentage:.1f}%', 
                   xy=(L, mid_y),
                   xytext=(0, 0), textcoords='offset points',
                   ha='center', fontsize=15, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.6, edgecolor='darkgreen'))
    
    # Add overall average percentage as a text box
    avg_percentage = summary_stats['inter_partition_percentage'].mean()
    textstr = f'Overall Average:\n{avg_percentage:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', horizontalalignment='left', bbox=props, fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    # Set up argument parser
    home_directory = Path.home()
    parser = argparse.ArgumentParser(
        description='Plot hop metrics vs L values from multiple CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python plot_hops_vs_L.py -i /path/to/input -o /path/to/output.png
  python plot_hops_vs_L.py --input-folder ./data --output-file ./graphs/hops.png
  python plot_hops_vs_L.py  # Uses default paths
        """
    )
    
    parser.add_argument(
        '-i', '--input-folder',
        type=str,
        default=f'{home_directory}/workspace/rdma_anns/logs/',
        help='Path to the folder containing CSV files with prefix "result_L" (default: ~/workspace/rdma_anns/logs/)'
    )
    
    parser.add_argument(
        '-o', '--output-file',
        type=str,
        default=f'{home_directory}/workspace/rdma_anns/figures/plot_hops_vs_L.png',
        help='Path to the output PNG file (default: ~/workspace/rdma_anns/figures/plot_hops_vs_L.png)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_folder = Path(args.input_folder)
    output_file = Path(args.output_file)
    
    # Try to parse metadata from folder name
    metadata = parse_folder_name(input_folder.name)
    if metadata:
        print(f"\nExtracted metadata from folder name:")
        print(f"  Dataset: {metadata['dataset_name']} {metadata['dataset_size']}")
        print(f"  Beamwidth: {metadata['beamwidth']}")
        print(f"  Method: {metadata['method']}")
        print(f"  Num Servers: {metadata['num_servers']}")
    else:
        print(f"\nWarning: Could not parse metadata from folder name: {input_folder.name}")
        print("Title will not include dataset information.")
        metadata = None
    
    # Validate input folder exists
    if not input_folder.exists():
        print(f"Error: Input folder does not exist: {input_folder}")
        sys.exit(1)
    
    if not input_folder.is_dir():
        print(f"Error: Input path is not a directory: {input_folder}")
        sys.exit(1)
    
    # Create output folder if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files with prefix "result_L"
    csv_files = sorted(input_folder.glob('result_L*.csv'))
    
    if not csv_files:
        print(f"Error: No CSV files with prefix 'result_L' found in {input_folder}")
        sys.exit(1)
    
    print(f"\nFound {len(csv_files)} CSV file(s) to process:")
    for f in csv_files:
        print(f"  - {f.name}")
    
    # Process all files and collect summary statistics
    summary_data = []
    
    for file_path in csv_files:
        print(f"\nProcessing: {file_path.name}")
        
        # Load and process data
        df = load_and_process_single_file(file_path)
        l_value = df['L'].iloc[0]
        
        # Calculate summary statistics
        stats = {
            'L': l_value,
            'n_queries': len(df),
            'n_hops_mean': df['n_hops'].mean(),
            'n_hops_median': df['n_hops'].median(),
            'n_hops_std': df['n_hops'].std(),
            'n_hops_min': df['n_hops'].min(),
            'n_hops_max': df['n_hops'].max(),
            'n_hops_p95': df['n_hops'].quantile(0.95),
            'n_hops_p99': df['n_hops'].quantile(0.99),
            'inter_partition_hops_mean': df['inter_partition_hops'].mean(),
            'inter_partition_hops_median': df['inter_partition_hops'].median(),
            'inter_partition_hops_std': df['inter_partition_hops'].std(),
            'inter_partition_hops_min': df['inter_partition_hops'].min(),
            'inter_partition_hops_max': df['inter_partition_hops'].max(),
            'inter_partition_hops_p95': df['inter_partition_hops'].quantile(0.95),
            'inter_partition_hops_p99': df['inter_partition_hops'].quantile(0.99),
        }
        
        summary_data.append(stats)
        
        print(f"  L={l_value}: {len(df)} queries, "
              f"Total hops (mean/median): {stats['n_hops_mean']:.2f}/{stats['n_hops_median']:.2f}, "
              f"Inter-partition hops (mean/median): {stats['inter_partition_hops_mean']:.2f}/{stats['inter_partition_hops_median']:.2f}")
    
    # Create summary DataFrame
    summary_stats = pd.DataFrame(summary_data).sort_values('L')
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS ACROSS ALL L VALUES")
    print("="*80)
    print(summary_stats.to_string(index=False))
    
    # Calculate and display percentage of inter-partition hops
    summary_stats['inter_partition_percentage'] = (
        summary_stats['inter_partition_hops_mean'] / summary_stats['n_hops_mean'] * 100
    )
    
    print("\n" + "="*80)
    print("PERCENTAGE OF INTER-PARTITION HOPS")
    print("="*80)
    for idx, row in summary_stats.iterrows():
        print(f"L={row['L']:3.0f}: {row['inter_partition_percentage']:5.2f}% "
              f"({row['inter_partition_hops_mean']:.2f} / {row['n_hops_mean']:.2f})")
    
    # Calculate overall average percentage
    avg_percentage = summary_stats['inter_partition_percentage'].mean()
    print("-" * 80)
    print(f"OVERALL AVERAGE: {avg_percentage:.2f}%")
    print("="*80)
    
    # Create simple plot
    print("\nGenerating comparison plot...")
    fig = plot_hops_vs_L(summary_stats, metadata)
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {output_file}")
    
    print("\n" + "="*80)
    print("ALL PROCESSING COMPLETE!")
    print("="*80)
    print(f"\nOutput file: {output_file.absolute()}")

if __name__ == "__main__":
    main()
