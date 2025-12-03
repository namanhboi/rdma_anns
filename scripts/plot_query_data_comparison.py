#!/usr/bin/env python3
"""
Script to compare query performance metrics across different beam widths.
Generates aligned plots for:
1. Number of hops distribution across beam widths
2. Inter-partition hops distribution across beam widths
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from collections import Counter
import argparse
import sys
import re

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')

def parse_partition_history(partition_str):
    """Parse partition history string into list of integers."""
    partition_str = partition_str.strip('[]')
    if partition_str:
        partitions = [int(p.strip()) for p in partition_str.split(',') if p.strip()]
        return partitions
    return []

def extract_beam_width(folder_name):
    """Extract beam width from folder name."""
    # Look for pattern BEAMWIDTH_<number>
    match = re.search(r'BEAMWIDTH[_-](\d+)', folder_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def extract_dataset_info(folder_name):
    """Extract dataset name and size from folder name."""
    # Pattern: dataset_name_size (e.g., bigann_1B, sift_100M)
    # Look for common patterns
    match = re.search(r'(bigann|sift|deep|gist|glove)_(\d+[MBK])', folder_name, re.IGNORECASE)
    if match:
        dataset_name = match.group(1)
        dataset_size = match.group(2)
        return dataset_name, dataset_size
    return None, None

def load_and_process_single_file(file_path):
    """Load and process data from a single CSV file."""
    data = []
    with open(file_path, 'r') as f:
        header = f.readline().strip().split(',')
        
        for line in f:
            bracket_pos = line.rfind('[')
            if bracket_pos == -1:
                continue
            
            before_bracket = line[:bracket_pos].rstrip(',')
            parts = before_bracket.split(',')
            partition_history = line[bracket_pos:].strip()
            row = parts + [partition_history]
            data.append(row)
    
    df = pd.DataFrame(data, columns=header)
    
    # Convert numeric columns
    df['query_id'] = pd.to_numeric(df['query_id'])
    df['send_timestamp_ns'] = pd.to_numeric(df['send_timestamp_ns'])
    df['receive_timestamp_ns'] = pd.to_numeric(df['receive_timestamp_ns'])
    df['completion_time_us'] = pd.to_numeric(df['completion_time_us'])
    df['n_hops'] = pd.to_numeric(df['n_hops'])
    df['n_ios'] = pd.to_numeric(df['n_ios'])
    df['n_cmps'] = pd.to_numeric(df['n_cmps'])
    
    # Extract L value
    l_value = file_path.stem.split('_')[-1]
    df['L'] = int(l_value)
    
    # Parse partition history
    df['partition_list'] = df['partition_history'].apply(parse_partition_history)
    df['n_partitions_visited'] = df['partition_list'].apply(lambda x: len(set(x)))
    df['inter_partition_hops'] = df['partition_list'].apply(lambda x: max(0, len(x) - 1))
    
    return df

def find_result_files(folder_path):
    """Find all result_L*.csv files in a folder."""
    return sorted(folder_path.glob('result_L*.csv'))

def load_beam_width_data(comparison_folder, target_l=None):
    """
    Load data from all beam width experiments.
    Returns dict: {beam_width: {L_value: dataframe}}
    Also returns dataset_name and dataset_size.
    If target_l is specified, only load data for that L value.
    """
    comparison_path = Path(comparison_folder)
    
    if not comparison_path.exists():
        print(f"Error: Comparison folder does not exist: {comparison_path}")
        sys.exit(1)
    
    beam_data = {}
    dataset_name = None
    dataset_size = None
    
    # Iterate through subdirectories
    for subfolder in sorted(comparison_path.iterdir()):
        if not subfolder.is_dir():
            continue
        
        # Extract beam width from folder name
        beam_width = extract_beam_width(subfolder.name)
        if beam_width is None:
            print(f"Warning: Could not extract beam width from folder: {subfolder.name}")
            continue
        
        # Extract dataset info (only need to do this once)
        if dataset_name is None:
            dataset_name, dataset_size = extract_dataset_info(subfolder.name)
        
        # Find CSV files in this folder
        csv_files = find_result_files(subfolder)
        
        if not csv_files:
            print(f"Warning: No result_L*.csv files found in {subfolder.name}")
            continue
        
        print(f"Found beam width {beam_width} with {len(csv_files)} CSV file(s)")
        
        # Load data for each L value
        beam_data[beam_width] = {}
        for csv_file in csv_files:
            df = load_and_process_single_file(csv_file)
            l_value = df['L'].iloc[0]
            
            # Filter by target L if specified
            if target_l is not None and l_value != target_l:
                print(f"  - Skipping L={l_value} (only processing L={target_l})")
                continue
            
            beam_data[beam_width][l_value] = df
            print(f"  - Loaded L={l_value}: {len(df)} queries")
        
        # Remove beam width if no matching L values found
        if not beam_data[beam_width]:
            del beam_data[beam_width]
            print(f"  - No matching L values found for beam width {beam_width}")
    
    return beam_data, dataset_name, dataset_size

def plot_metric_comparison(beam_data, metric_name, metric_label, output_file, 
                          dataset_name=None, dataset_size=None, bins=30):
    """
    Create comparison plots for a specific metric across beam widths.
    Each beam width gets its own subplot, with shared y-axis limits.
    """
    # Get sorted beam widths
    beam_widths = sorted(beam_data.keys())
    n_beams = len(beam_widths)
    
    if n_beams == 0:
        print("Error: No beam width data to plot")
        return
    
    # Determine global x-axis range across all data
    all_values = []
    for beam_width in beam_widths:
        for l_value, df in beam_data[beam_width].items():
            all_values.extend(df[metric_name].values)
    
    if not all_values:
        print(f"Error: No data for metric {metric_name}")
        return
    
    global_min = min(all_values)
    global_max = max(all_values)
    
    # Create shared bins for consistent x-axis
    bin_edges = np.linspace(global_min, global_max, bins + 1)
    
    # First pass: compute all histograms to find global y-axis max
    all_hist_counts = []
    for beam_width in beam_widths:
        for l_value, df in beam_data[beam_width].items():
            counts, _ = np.histogram(df[metric_name], bins=bin_edges)
            all_hist_counts.extend(counts)
    
    global_y_max = max(all_hist_counts) if all_hist_counts else 1
    
    # Create figure with subplots (one per beam width)
    # Note: sharey=True to share y-axis, but NOT sharex
    fig, axes = plt.subplots(n_beams, 1, figsize=(12, 4 * n_beams), sharey=True)
    
    # Handle case of single beam width
    if n_beams == 1:
        axes = [axes]
    
    # Define colors for different L values
    colors = plt.cm.Set2(range(8))
    
    # Plot each beam width
    for idx, (beam_width, ax) in enumerate(zip(beam_widths, axes)):
        l_values = sorted(beam_data[beam_width].keys())
        
        # Plot histogram for each L value with different colors
        for l_idx, l_value in enumerate(l_values):
            df = beam_data[beam_width][l_value]
            color = colors[l_idx % len(colors)]
            
            ax.hist(df[metric_name], bins=bin_edges, 
                   alpha=0.6, label=f'L={l_value}', 
                   color=color, edgecolor='black', linewidth=0.5)
        
        # Set labels and title
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Beam Width = {beam_width}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis limits explicitly for all subplots
        ax.set_ylim(0, global_y_max * 1.05)  # Add 5% padding at the top
        
        # Only show legend if there are multiple L values
        if len(l_values) > 1:
            ax.legend(loc='upper right', fontsize=9)
        
        # Add statistics box
        all_values_beam = []
        for l_value, df in beam_data[beam_width].items():
            all_values_beam.extend(df[metric_name].values)
        
        mean_val = np.mean(all_values_beam)
        median_val = np.median(all_values_beam)
        p95_val = np.percentile(all_values_beam, 95)
        p99_val = np.percentile(all_values_beam, 99)
        std_val = np.std(all_values_beam)
        
        stats_text = f'Mean: {mean_val:.1f}\nMedian: {median_val:.1f}\n'
        stats_text += f'95th %: {p95_val:.1f}\n99th %: {p99_val:.1f}\n'
        stats_text += f'Std: {std_val:.1f}'
        
        ax.text(0.98, 0.98, stats_text,
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Set x-label on bottom plot only
    axes[-1].set_xlabel(metric_label, fontsize=12)
    
    # Create title with dataset name, size, and L value
    l_values_in_data = set()
    for beam_width in beam_widths:
        l_values_in_data.update(beam_data[beam_width].keys())
    
    title_parts = []
    if dataset_name:
        title_parts.append(dataset_name)
    if dataset_size:
        title_parts.append(dataset_size)
    if len(l_values_in_data) == 1:
        title_parts.append(f'L={list(l_values_in_data)[0]}')
    
    if title_parts:
        title = ' '.join(title_parts)
    else:
        title = metric_label
    
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison plot: {output_file}")

def plot_summary_statistics(beam_data, output_file, dataset_name=None, dataset_size=None):
    """
    Create a summary plot showing mean and percentiles for each metric across beam widths.
    """
    beam_widths = sorted(beam_data.keys())
    
    # Collect statistics for each beam width
    stats = {
        'beam_width': [],
        'n_hops_mean': [], 'n_hops_median': [], 'n_hops_p95': [],
        'inter_hops_mean': [], 'inter_hops_median': [], 'inter_hops_p95': [],
        'n_ios_mean': [], 'n_ios_median': [], 'n_ios_p95': [],
        'n_cmps_mean': [], 'n_cmps_median': [], 'n_cmps_p95': []
    }
    
    for beam_width in beam_widths:
        stats['beam_width'].append(beam_width)
        
        # Aggregate all data across L values for this beam width
        all_hops = []
        all_inter_hops = []
        all_ios = []
        all_cmps = []
        
        for l_value, df in beam_data[beam_width].items():
            all_hops.extend(df['n_hops'].values)
            all_inter_hops.extend(df['inter_partition_hops'].values)
            all_ios.extend(df['n_ios'].values)
            all_cmps.extend(df['n_cmps'].values)
        
        # Calculate statistics
        stats['n_hops_mean'].append(np.mean(all_hops))
        stats['n_hops_median'].append(np.median(all_hops))
        stats['n_hops_p95'].append(np.percentile(all_hops, 95))
        
        stats['inter_hops_mean'].append(np.mean(all_inter_hops))
        stats['inter_hops_median'].append(np.median(all_inter_hops))
        stats['inter_hops_p95'].append(np.percentile(all_inter_hops, 95))
        
        stats['n_ios_mean'].append(np.mean(all_ios))
        stats['n_ios_median'].append(np.median(all_ios))
        stats['n_ios_p95'].append(np.percentile(all_ios, 95))
        
        stats['n_cmps_mean'].append(np.mean(all_cmps))
        stats['n_cmps_median'].append(np.median(all_cmps))
        stats['n_cmps_p95'].append(np.percentile(all_cmps, 95))
    
    # Create figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = [
        ('n_hops', 'Number of Hops', axes[0, 0]),
        ('inter_hops', 'Inter-Partition Hops', axes[0, 1]),
        ('n_ios', 'Number of I/Os', axes[1, 0]),
        ('n_cmps', 'Number of Comparisons', axes[1, 1])
    ]
    
    # Get L value(s) for title
    all_l_values = set()
    for beam_width in beam_widths:
        all_l_values.update(beam_data[beam_width].keys())
    
    for metric_prefix, metric_label, ax in metrics:
        x = stats['beam_width']
        mean_vals = stats[f'{metric_prefix}_mean']
        median_vals = stats[f'{metric_prefix}_median']
        p95_vals = stats[f'{metric_prefix}_p95']
        
        # Plot lines
        ax.plot(x, mean_vals, 'o-', label='Mean', linewidth=2, markersize=8)
        ax.plot(x, median_vals, 's-', label='Median', linewidth=2, markersize=8)
        ax.plot(x, p95_vals, '^-', label='95th percentile', linewidth=2, markersize=8)
        
        ax.set_xlabel('Beam Width', fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(f'{metric_label} vs Beam Width', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)
    
    # Create title with dataset name, size, and L value
    title_parts = []
    if dataset_name:
        title_parts.append(dataset_name)
    if dataset_size:
        title_parts.append(dataset_size)
    if len(all_l_values) == 1:
        title_parts.append(f'L={list(all_l_values)[0]}')
    
    if title_parts:
        title = ' '.join(title_parts)
    else:
        title = 'Summary Statistics'
    
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.998)
    
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved summary statistics plot: {output_file}")

def print_summary_table(beam_data):
    """Print a summary table of statistics across beam widths."""
    print("\n" + "="*100)
    print("SUMMARY STATISTICS ACROSS BEAM WIDTHS")
    print("="*100)
    
    beam_widths = sorted(beam_data.keys())
    
    # Print header
    header = f"{'Beam Width':<12} {'Metric':<20} {'Mean':<10} {'Median':<10} {'95th %':<10} {'99th %':<10}"
    print(header)
    print("-" * 100)
    
    for beam_width in beam_widths:
        # Aggregate data
        all_hops = []
        all_inter_hops = []
        all_ios = []
        all_cmps = []
        
        for l_value, df in beam_data[beam_width].items():
            all_hops.extend(df['n_hops'].values)
            all_inter_hops.extend(df['inter_partition_hops'].values)
            all_ios.extend(df['n_ios'].values)
            all_cmps.extend(df['n_cmps'].values)
        
        # Print stats for each metric
        metrics_data = [
            ('Hops', all_hops),
            ('Inter-Partition', all_inter_hops),
            ('I/Os', all_ios),
            ('Comparisons', all_cmps)
        ]
        
        for idx, (metric_name, values) in enumerate(metrics_data):
            mean_val = np.mean(values)
            median_val = np.median(values)
            p95_val = np.percentile(values, 95)
            p99_val = np.percentile(values, 99)
            
            bw_str = str(beam_width) if idx == 0 else ""
            print(f"{bw_str:<12} {metric_name:<20} {mean_val:<10.2f} {median_val:<10.2f} {p95_val:<10.2f} {p99_val:<10.2f}")
        
        print("-" * 100)

def main():
    parser = argparse.ArgumentParser(
        description='Compare query performance metrics across different beam widths',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python compare_beam_widths.py -i /path/to/comparison/folder -o /path/to/output
  python compare_beam_widths.py -i ./bigann_1B_hops_beam_width_comparison -L 10
  python compare_beam_widths.py -i ./bigann_1B_hops_beam_width_comparison -L 10 -o ./figures
        """
    )
    
    home_directory = Path.home()
    
    parser.add_argument(
        '-i', '--input-folder',
        type=str,
        required=True,
        help='Path to folder containing subfolders for different beam widths'
    )
    
    parser.add_argument(
        '-o', '--output-folder',
        type=str,
        default=f'{home_directory}/workspace/rdma_anns/figures/beam_width_comparison/',
        help='Path to folder where output graphs will be saved'
    )
    
    parser.add_argument(
        '-L', '--l-value',
        type=int,
        default=None,
        help='Only process data for this specific L value (default: process all L values)'
    )
    
    args = parser.parse_args()
    
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    target_l = args.l_value
    
    # Create output folder with L-specific subfolder if filtering
    if target_l is not None:
        output_folder = output_folder / f'L_{target_l}'
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading data from: {input_folder}")
    if target_l is not None:
        print(f"Filtering for L = {target_l}")
    print("="*100)
    
    # Load all beam width data
    beam_data, dataset_name, dataset_size = load_beam_width_data(input_folder, target_l=target_l)
    
    if not beam_data:
        print("Error: No valid beam width data found")
        if target_l is not None:
            print(f"No data found for L={target_l}. Check if CSV files with this L value exist.")
        sys.exit(1)
    
    print("\n" + "="*100)
    print(f"Successfully loaded data for {len(beam_data)} beam width(s): {sorted(beam_data.keys())}")
    
    # Show which L values are present
    all_l_values = set()
    for beam_width in beam_data:
        all_l_values.update(beam_data[beam_width].keys())
    print(f"L value(s) in data: {sorted(all_l_values)}")
    
    if dataset_name and dataset_size:
        print(f"Dataset: {dataset_name} {dataset_size}")
    
    print("="*100)
    
    # Print summary table
    print_summary_table(beam_data)
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    
    # 1. Number of hops comparison
    plot_metric_comparison(
        beam_data,
        'n_hops',
        'Number of Hops',
        output_folder / 'hops_comparison.png',
        dataset_name=dataset_name,
        dataset_size=dataset_size,
        bins=30
    )
    
    # 2. Inter-partition hops comparison
    plot_metric_comparison(
        beam_data,
        'inter_partition_hops',
        'Inter-Partition Hops',
        output_folder / 'inter_partition_hops_comparison.png',
        dataset_name=dataset_name,
        dataset_size=dataset_size,
        bins=30
    )
    
    # 3. Number of I/Os comparison
    plot_metric_comparison(
        beam_data,
        'n_ios',
        'Number of I/Os',
        output_folder / 'ios_comparison.png',
        dataset_name=dataset_name,
        dataset_size=dataset_size,
        bins=30
    )
    
    # 4. Number of comparisons comparison
    plot_metric_comparison(
        beam_data,
        'n_cmps',
        'Number of Comparisons',
        output_folder / 'comparisons_comparison.png',
        dataset_name=dataset_name,
        dataset_size=dataset_size,
        bins=30
    )
    
    # 5. Summary statistics plot
    plot_summary_statistics(
        beam_data,
        output_folder / 'summary_statistics.png',
        dataset_name=dataset_name,
        dataset_size=dataset_size
    )
    
    print("\n" + "="*100)
    print("ALL COMPARISONS COMPLETED SUCCESSFULLY!")
    print("="*100)
    print(f"\nOutput folder: {output_folder.absolute()}")
    print("\nGenerated files:")
    for output_file in sorted(output_folder.glob('*.png')):
        print(f"  - {output_file.name}")

if __name__ == "__main__":
    main()
