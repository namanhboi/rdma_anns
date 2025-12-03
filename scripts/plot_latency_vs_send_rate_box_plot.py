import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import re

# Define legend name mapping
LEGEND_NAME_MAPPING = {
    'STATE_SEND': 'BatANN',
    'SCATTER_GATHER': 'ScatterGather',
    'SINGLE_SERVER': 'SingleServer'
}

def parse_partition_history(history_str):
    """Parse partition history string into list of integers."""
    if pd.isna(history_str) or history_str == '':
        return []
    # Remove brackets and split by comma
    cleaned = history_str.strip('[]').strip()
    if not cleaned:
        return []
    return [int(x.strip()) for x in cleaned.split(',') if x.strip()]

def calculate_outliers(data):
    """Calculate outlier statistics for a dataset using 1.5*IQR rule."""
    if len(data) == 0:
        return 0, 0, 0.0
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    n_outliers = len(outliers)
    total = len(data)
    pct_outliers = (n_outliers / total) * 100.0 if total > 0 else 0.0
    
    return n_outliers, total, pct_outliers

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
    
    return df

def load_data_from_folder(base_folder):
    """Load all data from the folder structure."""
    base_path = Path(base_folder)
    all_data = []
    
    # Iterate through send_rate folders
    for send_rate_folder in sorted(base_path.iterdir()):
        if not send_rate_folder.is_dir():
            continue
        
        # Extract send rate from folder name (e.g., send_rate_10000 -> 10000)
        match = re.search(r'send_rate_(\d+)', send_rate_folder.name)
        if not match:
            continue
        
        send_rate = int(match.group(1))
        
        # Look for method folders (SCATTER_GATHER and STATE_SEND)
        for method_folder in send_rate_folder.iterdir():
            if not method_folder.is_dir():
                continue
            
            method = method_folder.name
            if method not in ['SCATTER_GATHER', 'STATE_SEND']:
                continue
            
            # Find CSV files in this folder
            csv_files = list(method_folder.glob('*.csv'))
            
            for csv_file in csv_files:
                print(f"Loading {csv_file}...")
                df = load_and_process_single_file(csv_file)
                df['send_rate'] = send_rate
                df['method'] = method
                all_data.append(df)
    
    if not all_data:
        raise ValueError("No data found in the specified folder structure")
    
    return pd.concat(all_data, ignore_index=True)

def plot_latency_comparison(df, output_file='latency_comparison.png'):
    """Create box-and-whisker plot comparing latencies."""
    # Get unique send rates and sort them
    send_rates = sorted(df['send_rate'].unique())
    methods = ['SCATTER_GATHER', 'STATE_SEND']
    
    # Prepare data for plotting - check which methods exist for each rate
    data_by_rate_method = {}
    for rate in send_rates:
        data_by_rate_method[rate] = {}
        for method in methods:
            mask = (df['send_rate'] == rate) & (df['method'] == method)
            latencies = df[mask]['e2e_latency_us'].values
            if len(latencies) > 0:
                data_by_rate_method[rate][method] = latencies
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up positions for box plots
    n_rates = len(send_rates)
    width = 0.5  # Increased from 0.35 to 0.5
    colors = {'SCATTER_GATHER': '#ff7f0e', 'STATE_SEND': '#1f77b4'}  # Swapped: orange for SCATTER_GATHER, blue for STATE_SEND
    
    # Prepare data for box plots
    all_data = []
    all_positions = []
    all_colors = []
    
    for i, rate in enumerate(send_rates):
        base_pos = i * 2.5  # Space between groups
        
        # Get available methods for this rate
        available_methods = [m for m in methods if m in data_by_rate_method[rate]]
        
        if len(available_methods) == 0:
            continue
        elif len(available_methods) == 1:
            # Center the single box plot
            method = available_methods[0]
            latencies = data_by_rate_method[rate][method]
            all_data.append(latencies)
            all_positions.append(base_pos + width/2)  # Center it
            all_colors.append(colors[method])
        else:
            # Both methods available - plot side by side
            for j, method in enumerate(methods):
                if method in data_by_rate_method[rate]:
                    latencies = data_by_rate_method[rate][method]
                    all_data.append(latencies)
                    pos = base_pos + j * width
                    all_positions.append(pos)
                    all_colors.append(colors[method])
    
    # Create box plots without outliers
    bp = ax.boxplot(all_data, positions=all_positions, widths=width*0.9,  # Increased from 0.8 to 0.9
                     patch_artist=True, showfliers=False,
                     boxprops=dict(linewidth=1.5),
                     medianprops=dict(linewidth=2, color='red'),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], all_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Set x-axis labels at the center of each group
    center_positions = [i * 2.5 + width/2 for i in range(n_rates)]
    ax.set_xticks(center_positions)
    ax.set_xticklabels([f'{rate}' for rate in send_rates])
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Labels and title
    ax.set_xlabel('Send Rate (queries/sec)', fontsize=14, fontweight='bold')
    ax.set_ylabel('End-to-End Latency (μs)', fontsize=14, fontweight='bold')
    # ax.set_title('Latency Comparison: SCATTER_GATHER vs STATE_SEND', 
                 # fontsize=16, fontweight='bold', pad=20)
    
    # Add legend with mapped names
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['SCATTER_GATHER'], alpha=0.7, 
              label=LEGEND_NAME_MAPPING.get('SCATTER_GATHER', 'SCATTER_GATHER')),
        Patch(facecolor=colors['STATE_SEND'], alpha=0.7, 
              label=LEGEND_NAME_MAPPING.get('STATE_SEND', 'STATE_SEND'))
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=18)
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    # Display summary statistics with outlier information
    print("\n=== Summary Statistics ===")
    for rate in send_rates:
        print(f"\nSend Rate: {rate}")
        for method in methods:
            if method in data_by_rate_method[rate]:
                latencies = data_by_rate_method[rate][method]
                n_outliers, total, pct_outliers = calculate_outliers(latencies)
                
                # Use display name in output
                display_name = LEGEND_NAME_MAPPING.get(method, method)
                print(f"  {display_name}:")
                print(f"    Count: {len(latencies)}")
                print(f"    Mean: {np.mean(latencies):.2f} μs")
                print(f"    Median: {np.median(latencies):.2f} μs")
                print(f"    Std: {np.std(latencies):.2f} μs")
                print(f"    Min: {np.min(latencies):.2f} μs")
                print(f"    Max: {np.max(latencies):.2f} μs")
                print(f"    P25: {np.percentile(latencies, 25):.2f} μs")
                print(f"    P75: {np.percentile(latencies, 75):.2f} μs")
                print(f"    Outliers: {n_outliers} / {total} ({pct_outliers:.2f}%)")
            else:
                display_name = LEGEND_NAME_MAPPING.get(method, method)
                print(f"  {display_name}: No data")
    
    # Overall outlier summary
    print("\n=== Overall Outlier Summary ===")
    for method in methods:
        all_latencies = []
        for rate in send_rates:
            if method in data_by_rate_method[rate]:
                all_latencies.extend(data_by_rate_method[rate][method])
        
        if len(all_latencies) > 0:
            all_latencies = np.array(all_latencies)
            n_outliers, total, pct_outliers = calculate_outliers(all_latencies)
            display_name = LEGEND_NAME_MAPPING.get(method, method)
            print(f"{display_name}: {n_outliers} / {total} ({pct_outliers:.2f}%) outliers across all send rates")

def main():
    parser = argparse.ArgumentParser(description='Plot latency comparison from experimental results')
    parser.add_argument('folder', type=str, help='Base folder containing send_rate_* subdirectories')
    parser.add_argument('--output', type=str, default='latency_comparison.png',
                       help='Output file name for the plot (default: latency_comparison.png)')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    df = load_data_from_folder(args.folder)
    print(f"Loaded {len(df)} records")
    
    # Create plot
    plot_latency_comparison(df, args.output)

if __name__ == '__main__':
    main()
