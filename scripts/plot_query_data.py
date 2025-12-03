#!/usr/bin/env python3
"""
Script to visualize query performance data including:
1. Number of hops as queries progress
2. Number of comparisons as queries progress
3. Number of I/Os as queries progress
4. Distribution of number of partitions visited
5. Distribution of inter-partition hops
6. Distribution of partition visit frequency across all queries
7. Completion time percentage (completion_time_us / e2e_latency)
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
    # Read header
    with open(file_path, 'r') as f:
        header_line = f.readline().strip()
    
    # Split header by comma
    header = header_line.split(',')
    
    # Use pandas to read CSV, but we need to handle the bracket fields specially
    # Read the entire file and manually parse it
    data = []
    with open(file_path, 'r') as f:
        _ = f.readline()  # Skip header
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Find both bracket sections
            # partition_history starts with first '['
            # partition_history_hop_idx starts with second '['
            first_bracket = line.find('[')
            if first_bracket == -1:
                continue
            
            # Everything before first bracket is regular CSV
            before_brackets = line[:first_bracket].rstrip(',')
            regular_parts = before_brackets.split(',')
            
            # Everything from first bracket onwards contains both bracket fields
            bracket_section = line[first_bracket:]
            
            # Find the closing bracket of the first array
            bracket_depth = 0
            first_array_end = -1
            for i, char in enumerate(bracket_section):
                if char == '[':
                    bracket_depth += 1
                elif char == ']':
                    bracket_depth -= 1
                    if bracket_depth == 0:
                        first_array_end = i + 1
                        break
            
            if first_array_end == -1:
                print(f"Warning: Could not parse line: {line[:50]}...")
                continue
            
            # Extract the two bracket fields
            partition_history = bracket_section[:first_array_end]
            remaining = bracket_section[first_array_end:].lstrip(',')
            
            # The remaining part should be the second bracket field
            partition_history_hop_idx = remaining.strip()
            
            # Combine all parts
            row = regular_parts + [partition_history, partition_history_hop_idx]
            
            # Verify we have the right number of columns
            if len(row) != len(header):
                print(f"Warning: Row has {len(row)} columns, expected {len(header)}")
                print(f"  Row: {row[:5]}... (showing first 5)")
                continue
            
            data.append(row)
    
    df = pd.DataFrame(data, columns=header)
    
    # Convert numeric columns to appropriate types
    df['query_id'] = pd.to_numeric(df['query_id'])
    df['send_timestamp_ns'] = pd.to_numeric(df['send_timestamp_ns'])
    df['receive_timestamp_ns'] = pd.to_numeric(df['receive_timestamp_ns'])
    df['completion_time_us'] = pd.to_numeric(df['completion_time_us'])
    df['io_us'] = pd.to_numeric(df['io_us'])
    df['n_hops'] = pd.to_numeric(df['n_hops'])
    df['n_ios'] = pd.to_numeric(df['n_ios'])
    df['n_cmps'] = pd.to_numeric(df['n_cmps'])
    
    # Calculate end-to-end latency in microseconds
    df['e2e_latency_us'] = (df['receive_timestamp_ns'] - df['send_timestamp_ns']) / 1000.0
    
    # Calculate completion time percentage
    df['completion_pct'] = (df['completion_time_us'] / df['e2e_latency_us']) * 100.0
    
    # Calculate io_us percentage
    df['io_pct'] = (df['io_us'] / df['e2e_latency_us']) * 100.0
    
    # Extract L value from filename (e.g., result_L_10.csv -> 10)
    l_value = file_path.stem.split('_')[-1]
    df['L'] = int(l_value)
    
    # Parse partition history
    df['partition_list'] = df['partition_history'].apply(parse_partition_history)
    
    # Calculate number of unique partitions visited
    df['n_partitions_visited'] = df['partition_list'].apply(lambda x: len(set(x)))
    
    # Calculate inter-partition hops (length of partition history - 1, minimum 0)
    df['inter_partition_hops'] = df['partition_list'].apply(lambda x: max(0, len(x) - 1))
    
    return df


def plot_time_comparison_across_L(all_dfs, output_folder):
    """Create a line plot comparing completion time and e2e latency across different L values."""
    # Aggregate data across all L values
    l_values = []
    completion_times = []
    e2e_latencies = []
    completion_pcts = []
    
    for df in all_dfs:
        l_value = df['L'].iloc[0]
        l_values.append(l_value)
        completion_times.append(df['completion_time_us'].mean())
        e2e_latencies.append(df['e2e_latency_us'].mean())
        completion_pcts.append(df['completion_pct'].mean())
    
    # Sort by L value
    sorted_indices = sorted(range(len(l_values)), key=lambda i: l_values[i])
    l_values = [l_values[i] for i in sorted_indices]
    completion_times = [completion_times[i] for i in sorted_indices]
    e2e_latencies = [e2e_latencies[i] for i in sorted_indices]
    completion_pcts = [completion_pcts[i] for i in sorted_indices]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot lines
    line1 = ax.plot(l_values, e2e_latencies, marker='o', linewidth=2.5, 
                    markersize=8, color='steelblue', label='E2E Latency', zorder=3)
    line2 = ax.plot(l_values, completion_times, marker='s', linewidth=2.5, 
                    markersize=8, color='darkorange', label='Completion Time', zorder=3)
    
    # Add percentage labels above the E2E latency line
    for i, (l, e2e_time, pct) in enumerate(zip(l_values, e2e_latencies, completion_pcts)):
        # Calculate offset to place label above the line
        y_range = max(e2e_latencies) - min(completion_times)
        offset = y_range * 0.03  # 3% of the range
        
        ax.text(l, e2e_time + offset, f'{pct:.1f}%', 
               ha='center', va='bottom', fontsize=10, fontweight='bold',
               color='darkorange', bbox=dict(boxstyle='round,pad=0.3', 
               facecolor='white', edgecolor='darkorange', alpha=0.9))
    
    ax.set_xlabel('L Value', fontsize=13, fontweight='bold')
    ax.set_ylabel('Time (μs)', fontsize=13, fontweight='bold')
    ax.set_title('Completion Time vs E2E Latency Across L Values', 
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Set x-axis ticks to show all L values
    ax.set_xticks(l_values)
    
    # Add some padding to y-axis for the percentage labels
    y_min = min(completion_times)
    y_max = max(e2e_latencies)
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.15 * y_range)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = output_folder / 'time_comparison_across_L.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nTime comparison plot saved to: {output_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("TIME COMPARISON SUMMARY")
    print("="*80)
    print(f"{'L Value':<10} {'E2E Latency (μs)':<20} {'Completion (μs)':<20} {'Completion %':<15}")
    print("-"*80)
    for l, e2e, comp, pct in zip(l_values, e2e_latencies, completion_times, completion_pcts):
        print(f"{l:<10} {e2e:<20.2f} {comp:<20.2f} {pct:<15.2f}%")
    print("="*80)
    
def plot_all_metrics(df, l_value):
    """Create a single figure with all plots as subplots."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Define consistent color palette for partitions
    all_partitions = []
    for partition_list in df['partition_list']:
        all_partitions.extend(partition_list)
    unique_partitions = sorted(set(all_partitions))
    
    # Create color map for partitions
    if unique_partitions:
        colors = plt.cm.tab10(range(len(unique_partitions)))
        partition_colors = {pid: colors[i] for i, pid in enumerate(unique_partitions)}
    else:
        partition_colors = {}
    
    # 1. Number of hops distribution - top left
    ax1 = fig.add_subplot(gs[0, 0])
    bins = 30
    n, bins_edges, patches = ax1.hist(df['n_hops'], bins=bins, color='steelblue', 
                                       alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Number of Hops', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title(f'Distribution of Hops (L={l_value})', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add statistics box
    mean_val = df['n_hops'].mean()
    median_val = df['n_hops'].median()
    p95_val = df['n_hops'].quantile(0.95)
    p99_val = df['n_hops'].quantile(0.99)
    std_val = df['n_hops'].std()
    stats_text = f'Mean: {mean_val:.1f}\nMedian: {median_val:.1f}\n'
    stats_text += f'95th %: {p95_val:.1f}\n99th %: {p99_val:.1f}\n'
    stats_text += f'Std Dev: {std_val:.1f}'
    ax1.text(0.98, 0.98, stats_text,
            transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Number of comparisons distribution - top middle
    ax2 = fig.add_subplot(gs[0, 1])
    n, bins_edges, patches = ax2.hist(df['n_cmps'], bins=bins, color='forestgreen', 
                                       alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Comparisons', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'Distribution of Comparisons (L={l_value})', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics box
    mean_val = df['n_cmps'].mean()
    median_val = df['n_cmps'].median()
    p95_val = df['n_cmps'].quantile(0.95)
    p99_val = df['n_cmps'].quantile(0.99)
    std_val = df['n_cmps'].std()
    stats_text = f'Mean: {mean_val:.1f}\nMedian: {median_val:.1f}\n'
    stats_text += f'95th %: {p95_val:.1f}\n99th %: {p99_val:.1f}\n'
    stats_text += f'Std Dev: {std_val:.1f}'
    ax2.text(0.98, 0.98, stats_text,
            transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Number of I/Os distribution - top right
    ax3 = fig.add_subplot(gs[0, 2])
    n, bins_edges, patches = ax3.hist(df['n_ios'], bins=bins, color='darkorange', 
                                       alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Number of I/Os', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title(f'Distribution of I/Os (L={l_value})', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add statistics box
    mean_val = df['n_ios'].mean()
    median_val = df['n_ios'].median()
    p95_val = df['n_ios'].quantile(0.95)
    p99_val = df['n_ios'].quantile(0.99)
    std_val = df['n_ios'].std()
    stats_text = f'Mean: {mean_val:.1f}\nMedian: {median_val:.1f}\n'
    stats_text += f'95th %: {p95_val:.1f}\n99th %: {p99_val:.1f}\n'
    stats_text += f'Std Dev: {std_val:.1f}'
    ax3.text(0.98, 0.98, stats_text,
            transform=ax3.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Partitions visited distribution - middle left
    ax4 = fig.add_subplot(gs[1, 0])
    value_counts = df['n_partitions_visited'].value_counts().sort_index()
    ax4.bar(value_counts.index, value_counts.values, 
           color='steelblue', alpha=0.7, edgecolor='black', width=0.6)
    ax4.set_xlabel('Number of Unique Partitions', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title(f'Unique Partitions Visited per Query', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xticks(value_counts.index)
    
    # Add statistics
    mean_val = df['n_partitions_visited'].mean()
    median_val = df['n_partitions_visited'].median()
    stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}'
    ax4.text(0.98, 0.02, stats_text,
            transform=ax4.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5. Inter-partition hops distribution - middle middle
    ax5 = fig.add_subplot(gs[1, 1])
    value_counts = df['inter_partition_hops'].value_counts().sort_index()
    
    ax5.bar(value_counts.index, value_counts.values, 
           color='steelblue', alpha=0.7, edgecolor='black', width=0.6)
    ax5.set_xlabel('Number of Inter-Partition Hops', fontsize=11)
    ax5.set_ylabel('Frequency', fontsize=11)
    ax5.set_title(f'Inter-Partition Hops Distribution', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Limit number of x-axis ticks to avoid clutter
    all_ticks = value_counts.index.tolist()
    if len(all_ticks) > 15:
        step = max(1, len(all_ticks) // 10)
        shown_ticks = all_ticks[::step]
        if all_ticks[-1] not in shown_ticks:
            shown_ticks.append(all_ticks[-1])
        ax5.set_xticks(shown_ticks)
        
        for tick in shown_ticks:
            if tick in value_counts.index:
                count = value_counts[tick]
                ax5.text(tick, count, str(count), ha='center', va='bottom', fontsize=8, fontweight='bold')
    else:
        ax5.set_xticks(all_ticks)
        
        for idx, count in zip(value_counts.index, value_counts.values):
            ax5.text(idx, count, str(count), ha='center', va='bottom', fontsize=8)
    
    # Add statistics
    mean_val = df['inter_partition_hops'].mean()
    median_val = df['inter_partition_hops'].median()
    stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}'
    ax5.text(0.98, 0.98, stats_text,
            transform=ax5.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 6. Partition visit frequency - middle right
    ax6 = fig.add_subplot(gs[1, 2])
    
    all_partitions_freq = []
    for partition_list in df['partition_list']:
        all_partitions_freq.extend(partition_list)
    
    partition_counts = Counter(all_partitions_freq)
    
    if partition_counts:
        sorted_partitions = sorted(partition_counts.items())
        partition_ids = [pid for pid, _ in sorted_partitions]
        visit_counts = [count for _, count in sorted_partitions]
        
        bar_colors = [partition_colors[pid] for pid in partition_ids]
        ax6.bar(partition_ids, visit_counts, 
               color=bar_colors, alpha=0.7, edgecolor='black', width=0.6)
        ax6.set_xticks(partition_ids)
        
        for pid, count in zip(partition_ids, visit_counts):
            ax6.text(pid, count, str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        total_visits = sum(visit_counts)
        stats_text = f'Total: {total_visits}\n'
        stats_text += f'Unique: {len(partition_ids)}\n'
        stats_text += f'Avg: {total_visits/len(partition_ids):.1f}'
        ax6.text(0.98, 0.02, stats_text,
                transform=ax6.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax6.text(0.5, 0.5, 'No partition visits',
                transform=ax6.transAxes, fontsize=12,
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    ax6.set_xlabel('Partition ID', fontsize=11)
    ax6.set_ylabel('Number of Times Visited', fontsize=11)
    ax6.set_title(f'Partition Visit Frequency', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Completion time percentage - bottom left
    ax7 = fig.add_subplot(gs[2, 0])
    n, bins_edges, patches = ax7.hist(df['completion_pct'], bins=bins, color='purple', 
                                       alpha=0.7, edgecolor='black')
    ax7.set_xlabel('Completion Time %', fontsize=11)
    ax7.set_ylabel('Frequency', fontsize=11)
    ax7.set_title(f'Completion Time as % of E2E Latency (L={l_value})', fontsize=13, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add statistics box
    mean_val = df['completion_pct'].mean()
    median_val = df['completion_pct'].median()
    p95_val = df['completion_pct'].quantile(0.95)
    p99_val = df['completion_pct'].quantile(0.99)
    std_val = df['completion_pct'].std()
    stats_text = f'Mean: {mean_val:.1f}%\nMedian: {median_val:.1f}%\n'
    stats_text += f'95th %: {p95_val:.1f}%\n99th %: {p99_val:.1f}%\n'
    stats_text += f'Std Dev: {std_val:.1f}%'
    ax7.text(0.98, 0.98, stats_text,
            transform=ax7.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 8. I/O time percentage - bottom middle
    ax8 = fig.add_subplot(gs[2, 1])
    n, bins_edges, patches = ax8.hist(df['io_pct'], bins=bins, color='teal', 
                                       alpha=0.7, edgecolor='black')
    ax8.set_xlabel('I/O Time %', fontsize=11)
    ax8.set_ylabel('Frequency', fontsize=11)
    ax8.set_title(f'I/O Time as % of E2E Latency (L={l_value})', fontsize=13, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Add statistics box
    mean_val = df['io_pct'].mean()
    median_val = df['io_pct'].median()
    p95_val = df['io_pct'].quantile(0.95)
    p99_val = df['io_pct'].quantile(0.99)
    std_val = df['io_pct'].std()
    stats_text = f'Mean: {mean_val:.1f}%\nMedian: {median_val:.1f}%\n'
    stats_text += f'95th %: {p95_val:.1f}%\n99th %: {p99_val:.1f}%\n'
    stats_text += f'Std Dev: {std_val:.1f}%'
    ax8.text(0.98, 0.98, stats_text,
            transform=ax8.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

def print_summary_statistics(df, l_value):
    """Print summary statistics for the data."""
    print("\n" + "="*80)
    print(f"SUMMARY STATISTICS - L = {l_value}")
    print("="*80)
    
    print(f"\nNumber of queries: {len(df)}")
    
    print(f"\nEnd-to-End Latency (μs):")
    print(f"  Mean: {df['e2e_latency_us'].mean():.2f}")
    print(f"  Median: {df['e2e_latency_us'].median():.2f}")
    print(f"  95th percentile: {df['e2e_latency_us'].quantile(0.95):.2f}")
    print(f"  99th percentile: {df['e2e_latency_us'].quantile(0.99):.2f}")
    print(f"  Std Dev: {df['e2e_latency_us'].std():.2f}")
    
    print(f"\nCompletion Time Percentage:")
    print(f"  Mean: {df['completion_pct'].mean():.2f}%")
    print(f"  Median: {df['completion_pct'].median():.2f}%")
    print(f"  95th percentile: {df['completion_pct'].quantile(0.95):.2f}%")
    print(f"  99th percentile: {df['completion_pct'].quantile(0.99):.2f}%")
    print(f"  Std Dev: {df['completion_pct'].std():.2f}%")
    
    print(f"\nI/O Time Percentage:")
    print(f"  Mean: {df['io_pct'].mean():.2f}%")
    print(f"  Median: {df['io_pct'].median():.2f}%")
    print(f"  95th percentile: {df['io_pct'].quantile(0.95):.2f}%")
    print(f"  99th percentile: {df['io_pct'].quantile(0.99):.2f}%")
    print(f"  Std Dev: {df['io_pct'].std():.2f}%")
    
    print(f"\nNumber of Hops:")
    print(f"  Mean: {df['n_hops'].mean():.2f}")
    print(f"  Median: {df['n_hops'].median():.2f}")
    print(f"  95th percentile: {df['n_hops'].quantile(0.95):.2f}")
    print(f"  99th percentile: {df['n_hops'].quantile(0.99):.2f}")
    print(f"  Std Dev: {df['n_hops'].std():.2f}")
    print(f"  Min: {df['n_hops'].min():.2f}")
    print(f"  Max: {df['n_hops'].max():.2f}")
    
    print(f"\nNumber of I/Os:")
    print(f"  Mean: {df['n_ios'].mean():.2f}")
    print(f"  Median: {df['n_ios'].median():.2f}")
    print(f"  95th percentile: {df['n_ios'].quantile(0.95):.2f}")
    print(f"  99th percentile: {df['n_ios'].quantile(0.99):.2f}")
    print(f"  Std Dev: {df['n_ios'].std():.2f}")
    print(f"  Min: {df['n_ios'].min():.2f}")
    print(f"  Max: {df['n_ios'].max():.2f}")
    
    print(f"\nNumber of Comparisons:")
    print(f"  Mean: {df['n_cmps'].mean():.2f}")
    print(f"  Median: {df['n_cmps'].median():.2f}")
    print(f"  95th percentile: {df['n_cmps'].quantile(0.95):.2f}")
    print(f"  99th percentile: {df['n_cmps'].quantile(0.99):.2f}")
    print(f"  Std Dev: {df['n_cmps'].std():.2f}")
    print(f"  Min: {df['n_cmps'].min():.2f}")
    print(f"  Max: {df['n_cmps'].max():.2f}")
    
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
    if partition_counts:
        most_visited_pid = max(partition_counts, key=partition_counts.get)
        print(f"  Most visited partition: {most_visited_pid} ({partition_counts[most_visited_pid]} visits)")
    else:
        print(f"  Most visited partition: N/A (no partition visits)")

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
    
    # Store all dataframes for cross-L analysis
    all_dfs = []
    
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
        
        # Store dataframe for later
        all_dfs.append(df)
        
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
    
    # Create comparison plot across all L values
    if len(all_dfs) > 1:
        print("\n" + "="*80)
        print("CREATING CROSS-L COMPARISON PLOT")
        print("="*80)
        plot_time_comparison_across_L(all_dfs, output_folder)
    
    print("\n" + "="*80)
    print("ALL FILES PROCESSED SUCCESSFULLY!")
    print("="*80)
    print(f"\nOutput folder: {output_folder.absolute()}")
    print("\nGenerated files:")
    for output_file in sorted(output_folder.glob('*.png')):
        print(f"  - {output_file.name}")

if __name__ == "__main__":
    main()
