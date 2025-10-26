#!/usr/bin/env python3
"""
Script to visualize query performance data including:
1. Completion time as queries progress
2. Distribution of number of partitions visited
3. Distribution of inter-partition hops
4. Distribution of partition visit frequency across all queries
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from collections import Counter
import argparse
import sys

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')

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
            # Find the last '[' to identify start of partition_history
            bracket_pos = line.rfind('[')
            if bracket_pos == -1:
                continue
            
            # Split the part before partition_history normally
            before_bracket = line[:bracket_pos].rstrip(',')
            parts = before_bracket.split(',')
            
            # Extract partition_history
            partition_history = line[bracket_pos:].strip()
            
            # Combine all parts
            row = parts + [partition_history]
            data.append(row)
    
    df = pd.DataFrame(data, columns=header)
    
    # Convert numeric columns to appropriate types
    df['query_id'] = pd.to_numeric(df['query_id'])
    df['send_timestamp_ns'] = pd.to_numeric(df['send_timestamp_ns'])
    df['receive_timestamp_ns'] = pd.to_numeric(df['receive_timestamp_ns'])
    df['completion_time_us'] = pd.to_numeric(df['completion_time_us'])
    df['n_hops'] = pd.to_numeric(df['n_hops'])
    
    # Extract L value from filename (e.g., result_L_10.csv -> 10)
    l_value = file_path.stem.split('_')[-1]
    df['L'] = int(l_value)
    
    # Parse partition history
    df['partition_list'] = df['partition_history'].apply(parse_partition_history)
    
    # Calculate number of unique partitions visited
    df['n_partitions_visited'] = df['partition_list'].apply(lambda x: len(set(x)))
    
    # Calculate inter-partition hops (length of partition history - 1)
    df['inter_partition_hops'] = df['partition_list'].apply(lambda x: len(x) - 1)
    
    return df

def plot_all_metrics(df, l_value):
    """Create a single figure with all 4 plots as subplots."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Completion time progression - spans top row
    ax1 = fig.add_subplot(gs[0, :])
    df_sorted = df.sort_values('query_id')
    ax1.plot(range(len(df_sorted)), df_sorted['completion_time_us'], 
            alpha=0.6, linewidth=0.5, color='steelblue')
    ax1.set_xlabel('Query Index', fontsize=11)
    ax1.set_ylabel('Completion Time (μs)', fontsize=11)
    ax1.set_title(f'Completion Time Progression (L={l_value})', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add rolling average
    window = 100
    if len(df_sorted) >= window:
        rolling_avg = df_sorted['completion_time_us'].rolling(window=window).mean()
        ax1.plot(range(len(df_sorted)), rolling_avg, 
                color='red', linewidth=2, label=f'{window}-query moving avg')
        ax1.legend()
    
    # 2. Partitions visited distribution - bottom left
    ax2 = fig.add_subplot(gs[1, 0])
    value_counts = df['n_partitions_visited'].value_counts().sort_index()
    ax2.bar(value_counts.index, value_counts.values, 
           color='steelblue', alpha=0.7, edgecolor='black', width=0.6)
    ax2.set_xlabel('Number of Unique Partitions', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'Unique Partitions Visited per Query', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(value_counts.index)
    
    # Add statistics
    mean_val = df['n_partitions_visited'].mean()
    median_val = df['n_partitions_visited'].median()
    stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}'
    ax2.text(0.98, 0.02, stats_text,
            transform=ax2.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Inter-partition hops distribution - bottom middle
    ax3 = fig.add_subplot(gs[1, 1])
    value_counts = df['inter_partition_hops'].value_counts().sort_index()
    
    # Bar plot for all ranges
    ax3.bar(value_counts.index, value_counts.values, 
           color='steelblue', alpha=0.7, edgecolor='black', width=0.6)
    ax3.set_xlabel('Number of Inter-Partition Hops', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title(f'Inter-Partition Hops Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Limit number of x-axis ticks to avoid clutter
    all_ticks = value_counts.index.tolist()
    if len(all_ticks) > 15:
        # Show every nth tick to keep it readable
        step = max(1, len(all_ticks) // 10)
        shown_ticks = all_ticks[::step]
        if all_ticks[-1] not in shown_ticks:
            shown_ticks.append(all_ticks[-1])  # Always show the last tick
        ax3.set_xticks(shown_ticks)
        
        # Add labels above each shown tick to indicate the hop count
        for tick in shown_ticks:
            if tick in value_counts.index:
                count = value_counts[tick]
                ax3.text(tick, count, str(count), ha='center', va='bottom', fontsize=8, fontweight='bold')
    else:
        ax3.set_xticks(all_ticks)
        
        # Add frequency labels on top of bars for small ranges
        for idx, count in zip(value_counts.index, value_counts.values):
            ax3.text(idx, count, str(count), ha='center', va='bottom', fontsize=8)
    
    # Add statistics - moved to top right for this graph
    mean_val = df['inter_partition_hops'].mean()
    median_val = df['inter_partition_hops'].median()
    stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}'
    ax3.text(0.98, 0.98, stats_text,
            transform=ax3.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Partition visit frequency - bottom right
    ax4 = fig.add_subplot(gs[1, 2])
    
    # Count all partition visits across all queries
    all_partitions = []
    for partition_list in df['partition_list']:
        all_partitions.extend(partition_list)
    
    partition_counts = Counter(all_partitions)
    
    # Sort by partition ID
    sorted_partitions = sorted(partition_counts.items())
    partition_ids = [pid for pid, _ in sorted_partitions]
    visit_counts = [count for _, count in sorted_partitions]
    
    # Create bar plot
    ax4.bar(partition_ids, visit_counts, 
           color='steelblue', alpha=0.7, edgecolor='black', width=0.6)
    ax4.set_xlabel('Partition ID', fontsize=11)
    ax4.set_ylabel('Number of Times Visited', fontsize=11)
    ax4.set_title(f'Partition Visit Frequency', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xticks(partition_ids)
    
    # Add value labels on top of bars
    for pid, count in zip(partition_ids, visit_counts):
        ax4.text(pid, count, str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add statistics
    total_visits = sum(visit_counts)
    stats_text = f'Total: {total_visits}\n'
    stats_text += f'Unique: {len(partition_ids)}\n'
    stats_text += f'Avg: {total_visits/len(partition_ids):.1f}'
    ax4.text(0.98, 0.02, stats_text,
            transform=ax4.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

def print_summary_statistics(df, l_value):
    """Print summary statistics for the data."""
    print("\n" + "="*80)
    print(f"SUMMARY STATISTICS - L = {l_value}")
    print("="*80)
    
    print(f"\nNumber of queries: {len(df)}")
    print(f"\nCompletion Time (μs):")
    print(f"  Mean: {df['completion_time_us'].mean():.2f}")
    print(f"  Median: {df['completion_time_us'].median():.2f}")
    print(f"  Std Dev: {df['completion_time_us'].std():.2f}")
    print(f"  Min: {df['completion_time_us'].min():.2f}")
    print(f"  Max: {df['completion_time_us'].max():.2f}")
    
    print(f"\nPartitions Visited:")
    print(f"  Mean: {df['n_partitions_visited'].mean():.2f}")
    print(f"  Median: {df['n_partitions_visited'].median():.2f}")
    print(f"  Min: {df['n_partitions_visited'].min()}")
    print(f"  Max: {df['n_partitions_visited'].max()}")
    
    print(f"\nInter-Partition Hops:")
    print(f"  Mean: {df['inter_partition_hops'].mean():.2f}")
    print(f"  Median: {df['inter_partition_hops'].median():.2f}")
    print(f"  Min: {df['inter_partition_hops'].min()}")
    print(f"  Max: {df['inter_partition_hops'].max()}")
    
    # Partition visit statistics
    all_partitions = []
    for partition_list in df['partition_list']:
        all_partitions.extend(partition_list)
    partition_counts = Counter(all_partitions)
    
    print(f"\nPartition Usage:")
    print(f"  Unique partitions accessed: {len(partition_counts)}")
    print(f"  Total partition visits: {sum(partition_counts.values())}")
    print(f"  Most visited partition: {max(partition_counts, key=partition_counts.get)} ({partition_counts[max(partition_counts, key=partition_counts.get)]} visits)")

def main():
    # Set up argument parser
    home_directory = Path.home()
    parser = argparse.ArgumentParser(
        description='Visualize query performance data from CSV files with prefix result_L',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python visualize_query_data.py -i /path/to/input -o /path/to/output
  python visualize_query_data.py --input-folder ./data --output-folder ./graphs
  python visualize_query_data.py  # Uses default paths
        """
    )
    
    parser.add_argument(
        '-i', '--input-folder',
        type=str,
        default=f'{home_directory}/workspace/rdma_anns/logs/',
        help='Path to the folder containing CSV files with prefix "result_L" (default: ~/workspace/rdma_anns/logs/)'
    )
    
    parser.add_argument(
        '-o', '--output-folder',
        type=str,
        default=f'{home_directory}/workspace/rdma_anns/figures/',
        help='Path to the folder where output graphs will be saved (default: ~/workspace/rdma_anns/figures/)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    
    # Validate input folder exists
    if not input_folder.exists():
        print(f"Error: Input folder does not exist: {input_folder}")
        sys.exit(1)
    
    if not input_folder.is_dir():
        print(f"Error: Input path is not a directory: {input_folder}")
        sys.exit(1)
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files with prefix "result_L"
    csv_files = sorted(input_folder.glob('result_L*.csv'))
    
    if not csv_files:
        print(f"Error: No CSV files with prefix 'result_L' found in {input_folder}")
        sys.exit(1)
    
    print(f"\nFound {len(csv_files)} CSV file(s) to process:")
    for f in csv_files:
        print(f"  - {f.name}")
    
    # Process each file separately
    for file_path in csv_files:
        print(f"\n{'='*80}")
        print(f"Processing: {file_path.name}")
        print(f"{'='*80}")
        
        # Load and process data
        print("Loading and processing data...")
        df = load_and_process_single_file(file_path)
        l_value = df['L'].iloc[0]
        print(f"Queries loaded: {len(df)}")
        
        # Print summary statistics
        print_summary_statistics(df, l_value)
        
        # Create combined plot
        print("\nGenerating combined plot with all metrics...")
        fig = plot_all_metrics(df, l_value)
        
        # Save to output folder
        output_file = output_folder / f'all_metrics_L_{l_value}.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\nCombined plot saved to: {output_file}")
    
    print("\n" + "="*80)
    print("ALL FILES PROCESSED SUCCESSFULLY!")
    print("="*80)
    print(f"\nOutput folder: {output_folder.absolute()}")
    print("\nGenerated files:")
    for output_file in sorted(output_folder.glob('all_metrics_L_*.png')):
        print(f"  - {output_file.name}")

if __name__ == "__main__":
    main()
