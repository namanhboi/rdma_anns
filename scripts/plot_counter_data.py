#!/usr/bin/env python3
"""
Distributed Vector Search System Visualization
Plots time series data from multiple servers showing queue states and thread metrics
Now with timestamp support and warmup period filtering
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

# Configuration
WARMUP_SECONDS = 30  # Filter out first 30 seconds

# Key metrics to visualize
KEY_METRICS = [
    'num_states_global_queue',
    'num_foreign_states_global_queue',
    'num_new_states_global_queue'
]

def load_server_data(file_path):
    """Load and prepare server data from CSV file"""
    df = pd.read_csv(file_path)
    
    # Convert timestamp from nanoseconds to seconds (relative to start)
    df['timestamp_ns'] = pd.to_numeric(df['timestamp_ns'])
    start_time_ns = df['timestamp_ns'].iloc[0]
    df['time_seconds'] = (df['timestamp_ns'] - start_time_ns) / 1e9
    
    # Filter out first 30 seconds (warmup period)
    df = df[df['time_seconds'] >= WARMUP_SECONDS].copy()
    
    # Reset time to start from 0 after warmup
    if len(df) > 0:
        df['time_seconds'] = df['time_seconds'] - df['time_seconds'].iloc[0]
    
    return df

def plot_individual_server(df, server_name, output_path):
    """Create individual time series plots for a single server"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 13))
    fig.suptitle(f'Vector Search System Metrics - {server_name} (after {WARMUP_SECONDS}s warmup)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Global Queue Metrics
    ax1 = axes[0]
    ax1.plot(df['time_seconds'], df['num_states_global_queue'], 
             label='Total States', linewidth=2, color='#2E86AB')
    ax1.plot(df['time_seconds'], df['num_foreign_states_global_queue'], 
             label='Foreign States', linewidth=2, color='#A23B72')
    ax1.plot(df['time_seconds'], df['num_new_states_global_queue'], 
             label='New States', linewidth=2, color='#F18F01')
    ax1.set_ylabel('Number of States', fontsize=11, fontweight='bold')
    ax1.set_title('Global Queue State Distribution', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Thread State Distribution (Stacked Area)
    ax2 = axes[1]
    thread_cols = [col for col in df.columns if col.startswith('thread') and col.endswith('num_states')]
    thread_cols_sorted = sorted(thread_cols, key=lambda x: int(x.split('thread')[1].split('_')[0]))
    
    # Prepare data for stacked area chart
    thread_data = []
    thread_labels = []
    for col in thread_cols_sorted:
        thread_num = col.split('_')[0].replace('thread', '')
        thread_data.append(df[col].values)
        thread_labels.append(f'Thread {thread_num}')
    
    # Create stacked area chart with distinct colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(thread_data)))
    ax2.stackplot(df['time_seconds'], *thread_data, labels=thread_labels, 
                  colors=colors, alpha=0.8)
    ax2.set_ylabel('Number of States', fontsize=11, fontweight='bold')
    ax2.set_title('Thread-Level State Distribution (Stacked)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', ncol=4, framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Thread Activity Heatmap
    ax3 = axes[2]
    thread_cols = [col for col in df.columns if col.startswith('thread') and col.endswith('num_states')]
    thread_cols_sorted = sorted(thread_cols, key=lambda x: int(x.split('thread')[1].split('_')[0]))
    
    # Prepare heatmap data
    heatmap_data = []
    thread_labels = []
    for col in thread_cols_sorted:
        thread_num = col.split('_')[0].replace('thread', '')
        heatmap_data.append(df[col].values)
        thread_labels.append(f'T{thread_num}')
    
    heatmap_array = np.array(heatmap_data)
    
    # Downsample if too many time steps for visualization
    step = max(1, len(df) // 500)
    heatmap_downsampled = heatmap_array[:, ::step]
    time_values = df['time_seconds'].values
    
    im = ax3.imshow(heatmap_downsampled, aspect='auto', cmap='YlOrRd', 
                    interpolation='nearest', 
                    extent=[time_values[0], time_values[-1], len(thread_labels)-0.5, -0.5])
    ax3.set_yticks(range(len(thread_labels)))
    ax3.set_yticklabels(thread_labels)
    ax3.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Thread', fontsize=11, fontweight='bold')
    ax3.set_title('Thread Activity Heatmap', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, orientation='vertical', pad=0.01)
    cbar.set_label('Number of States', fontsize=10)
    
    # Plot 4: Peer Communication Load
    ax4 = axes[3]
    peer_cols = [col for col in df.columns if col.startswith('peer')]
    peer_cols_sorted = sorted(peer_cols, key=lambda x: int(x.split('_')[1]))
    colors_peer = plt.cm.Set2(np.linspace(0, 1, len(peer_cols_sorted)))
    
    for i, col in enumerate(peer_cols_sorted):
        peer_num = col.split('_')[1]
        # Last peer is the client
        if i == len(peer_cols_sorted) - 1:
            label = 'Client'
        else:
            label = f'Peer {peer_num}'
        ax4.plot(df['time_seconds'], df[col], 
                label=label, linewidth=2, color=colors_peer[i])
    ax4.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Elements to Send', fontsize=11, fontweight='bold')
    ax4.set_title('Peer-to-Peer Communication Queue', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved individual plot: {output_path}")
    plt.close()

def plot_combined_servers(data_dict, output_path):
    """Create combined comparison plots for all servers"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(f'Distributed Vector Search System - Multi-Server Comparison (after {WARMUP_SECONDS}s warmup)', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    # Plot 1: Global Queue States Comparison
    ax1 = axes[0]
    for i, (server_name, df) in enumerate(data_dict.items()):
        ax1.plot(df['time_seconds'], df['num_states_global_queue'], 
                label=server_name, linewidth=2.5, color=colors[i % len(colors)], alpha=0.8)
    ax1.set_ylabel('Number of States', fontsize=12, fontweight='bold')
    ax1.set_title('Global Queue - Total States (All Servers)', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9, fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Foreign States Comparison
    ax2 = axes[1]
    for i, (server_name, df) in enumerate(data_dict.items()):
        ax2.plot(df['time_seconds'], df['num_foreign_states_global_queue'], 
                label=server_name, linewidth=2.5, color=colors[i % len(colors)], alpha=0.8)
    ax2.set_ylabel('Foreign States', fontsize=12, fontweight='bold')
    ax2.set_title('Foreign States Distribution (All Servers)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9, fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: New States Comparison
    ax3 = axes[2]
    for i, (server_name, df) in enumerate(data_dict.items()):
        ax3.plot(df['time_seconds'], df['num_new_states_global_queue'], 
                label=server_name, linewidth=2.5, color=colors[i % len(colors)], alpha=0.8)
    ax3.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('New States', fontsize=12, fontweight='bold')
    ax3.set_title('New States Generation (All Servers)', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', framealpha=0.9, fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined plot: {output_path}")
    plt.close()


def plot_comprehensive_dashboard(data_dict, output_path):
    """Create a comprehensive dashboard showing all metrics for all servers"""
    n_servers = len(data_dict)
    
    # Create figure with grid layout: 
    # 3 rows for combined plots + 1 spacer + 5 rows for individual metrics x n_servers columns
    fig = plt.figure(figsize=(8 * n_servers, 28))
    gs = fig.add_gridspec(9, n_servers, hspace=0.35, wspace=0.25, 
                          height_ratios=[1, 1, 1, 0.1, 1, 1, 1, 1, 1],
                          top=0.96, bottom=0.02)
    
    fig.suptitle(f'Comprehensive Distributed Vector Search System Dashboard (after {WARMUP_SECONDS}s warmup)', 
                 fontsize=18, fontweight='bold', y=0.985)
    
    server_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    # ========== TOP SECTION: COMBINED COMPARISON PLOTS ==========
    # These span all columns
    
    # Row 0: Global Queue States Comparison
    ax_combined1 = plt.subplot(gs[0, :])
    for i, (server_name, df) in enumerate(data_dict.items()):
        ax_combined1.plot(df['time_seconds'], df['num_states_global_queue'], 
                label=server_name, linewidth=2.5, color=server_colors[i % len(server_colors)], alpha=0.8)
    ax_combined1.set_ylabel('Number of States', fontsize=12, fontweight='bold')
    ax_combined1.set_title('COMBINED VIEW: Global Queue - Total States (All Servers)', 
                           fontsize=14, fontweight='bold', pad=10)
    ax_combined1.legend(loc='best', framealpha=0.9, fontsize=11)
    ax_combined1.grid(True, alpha=0.3)
    
    # Row 1: Foreign States Comparison
    ax_combined2 = plt.subplot(gs[1, :])
    for i, (server_name, df) in enumerate(data_dict.items()):
        ax_combined2.plot(df['time_seconds'], df['num_foreign_states_global_queue'], 
                label=server_name, linewidth=2.5, color=server_colors[i % len(server_colors)], alpha=0.8)
    ax_combined2.set_ylabel('Foreign States', fontsize=12, fontweight='bold')
    ax_combined2.set_title('COMBINED VIEW: Foreign States Distribution (All Servers)', 
                           fontsize=14, fontweight='bold', pad=10)
    ax_combined2.legend(loc='best', framealpha=0.9, fontsize=11)
    ax_combined2.grid(True, alpha=0.3)
    
    # Row 2: New States Comparison
    ax_combined3 = plt.subplot(gs[2, :])
    for i, (server_name, df) in enumerate(data_dict.items()):
        ax_combined3.plot(df['time_seconds'], df['num_new_states_global_queue'], 
                label=server_name, linewidth=2.5, color=server_colors[i % len(server_colors)], alpha=0.8)
    ax_combined3.set_ylabel('New States', fontsize=12, fontweight='bold')
    ax_combined3.set_title('COMBINED VIEW: New States Generation (All Servers)', 
                           fontsize=14, fontweight='bold', pad=10)
    ax_combined3.legend(loc='best', framealpha=0.9, fontsize=11)
    ax_combined3.grid(True, alpha=0.3)
    
    # Row 3 is a spacer (empty)
    
    # ========== BOTTOM SECTION: INDIVIDUAL SERVER DETAILS ==========
    # Calculate y-axis limits for each row to ensure consistency
    global_queue_max = max(df['num_states_global_queue'].max() for df in data_dict.values())
    foreign_queue_max = max(df['num_foreign_states_global_queue'].max() for df in data_dict.values())
    new_queue_max = max(df['num_new_states_global_queue'].max() for df in data_dict.values())
    
    # For thread stacked areas, we need the sum
    thread_stack_max = 0
    for df in data_dict.values():
        thread_cols = [col for col in df.columns if col.startswith('thread') and col.endswith('num_states')]
        thread_sum = df[thread_cols].sum(axis=1).max()
        thread_stack_max = max(thread_stack_max, thread_sum)
    
    # For peer communication
    peer_max = 0
    for df in data_dict.values():
        peer_cols = [col for col in df.columns if col.startswith('peer')]
        if peer_cols:
            peer_max = max(peer_max, df[peer_cols].max().max())
    
    # For own vs foreign
    own_foreign_max = 0
    for df in data_dict.values():
        own_cols = [col for col in df.columns if col.startswith('thread') and col.endswith('num_own_states')]
        foreign_cols = [col for col in df.columns if col.startswith('thread') and col.endswith('num_foreign_states')]
        total = df[own_cols].sum(axis=1) + df[foreign_cols].sum(axis=1)
        own_foreign_max = max(own_foreign_max, total.max())
    
    for col_idx, (server_name, df) in enumerate(data_dict.items()):
        server_color = server_colors[col_idx % len(server_colors)]
        
        # Row 4: Global Queue Metrics
        ax1 = fig.add_subplot(gs[4, col_idx])
        ax1.plot(df['time_seconds'], df['num_states_global_queue'], 
                 label='Total States', linewidth=2, color='#2E86AB')
        ax1.plot(df['time_seconds'], df['num_foreign_states_global_queue'], 
                 label='Foreign States', linewidth=2, color='#A23B72')
        ax1.plot(df['time_seconds'], df['num_new_states_global_queue'], 
                 label='New States', linewidth=2, color='#F18F01')
        ax1.set_ylabel('Number of States', fontsize=10, fontweight='bold')
        ax1.set_title(f'{server_name}\nGlobal Queue States', fontsize=11, fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor=server_color, alpha=0.3))
        ax1.legend(loc='best', fontsize=8, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, max(global_queue_max, foreign_queue_max, new_queue_max) * 1.1])
        
        # Row 5: Thread State Distribution (Stacked Area)
        ax2 = fig.add_subplot(gs[5, col_idx])
        thread_cols = [col for col in df.columns if col.startswith('thread') and col.endswith('num_states')]
        thread_cols_sorted = sorted(thread_cols, key=lambda x: int(x.split('thread')[1].split('_')[0]))
        
        thread_data = []
        thread_labels = []
        for col in thread_cols_sorted:
            thread_num = col.split('_')[0].replace('thread', '')
            thread_data.append(df[col].values)
            thread_labels.append(f'T{thread_num}')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(thread_data)))
        ax2.stackplot(df['time_seconds'], *thread_data, labels=thread_labels, 
                      colors=colors, alpha=0.8)
        ax2.set_ylabel('Number of States', fontsize=10, fontweight='bold')
        ax2.set_title('Thread Distribution (Stacked)', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper left', ncol=4, fontsize=7, framealpha=0.9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, thread_stack_max * 1.1])
        
        # Row 6: Thread Activity Heatmap
        ax3 = fig.add_subplot(gs[6, col_idx])
        heatmap_data = []
        thread_labels_hm = []
        for col in thread_cols_sorted:
            thread_num = col.split('_')[0].replace('thread', '')
            heatmap_data.append(df[col].values)
            thread_labels_hm.append(f'T{thread_num}')
        
        heatmap_array = np.array(heatmap_data)
        step = max(1, len(df) // 500)
        heatmap_downsampled = heatmap_array[:, ::step]
        time_values = df['time_seconds'].values
        
        # Find global max for heatmap to keep color scale consistent
        heatmap_global_max = 0
        for df_temp in data_dict.values():
            thread_cols_temp = [col for col in df_temp.columns if col.startswith('thread') and col.endswith('num_states')]
            if thread_cols_temp:
                heatmap_global_max = max(heatmap_global_max, df_temp[thread_cols_temp].max().max())
        
        im = ax3.imshow(heatmap_downsampled, aspect='auto', cmap='YlOrRd', 
                        interpolation='nearest', vmin=0, vmax=heatmap_global_max,
                        extent=[time_values[0], time_values[-1], len(thread_labels_hm)-0.5, -0.5])
        ax3.set_yticks(range(len(thread_labels_hm)))
        ax3.set_yticklabels(thread_labels_hm, fontsize=8)
        ax3.set_ylabel('Thread', fontsize=10, fontweight='bold')
        ax3.set_title('Thread Activity Heatmap', fontsize=11, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax3, orientation='vertical', pad=0.01)
        cbar.set_label('States', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        
        # Row 7: Peer Communication Load
        ax4 = fig.add_subplot(gs[7, col_idx])
        peer_cols = [col for col in df.columns if col.startswith('peer')]
        peer_cols_sorted = sorted(peer_cols, key=lambda x: int(x.split('_')[1]))
        colors_peer = plt.cm.Set2(np.linspace(0, 1, len(peer_cols_sorted)))
        
        for i, col in enumerate(peer_cols_sorted):
            peer_num = col.split('_')[1]
            if i == len(peer_cols_sorted) - 1:
                label = 'Client'
            else:
                label = f'Peer {peer_num}'
            ax4.plot(df['time_seconds'], df[col], 
                    label=label, linewidth=2, color=colors_peer[i])
        ax4.set_ylabel('Elements to Send', fontsize=10, fontweight='bold')
        ax4.set_title('P2P Communication Queue', fontsize=11, fontweight='bold')
        ax4.legend(loc='best', fontsize=8, framealpha=0.9)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, peer_max * 1.1 if peer_max > 0 else 1])
        
        # Row 8: Per-thread breakdown (Own vs Foreign states)
        ax5 = fig.add_subplot(gs[8, col_idx])
        own_cols = [col for col in df.columns if col.startswith('thread') and col.endswith('num_own_states')]
        foreign_cols = [col for col in df.columns if col.startswith('thread') and col.endswith('num_foreign_states')]
        
        own_cols_sorted = sorted(own_cols, key=lambda x: int(x.split('thread')[1].split('_')[0]))
        foreign_cols_sorted = sorted(foreign_cols, key=lambda x: int(x.split('thread')[1].split('_')[0]))
        
        # Sum across all threads
        total_own = df[own_cols_sorted].sum(axis=1)
        total_foreign = df[foreign_cols_sorted].sum(axis=1)
        
        ax5.fill_between(df['time_seconds'], 0, total_own, 
                         label='Own States', color='#6A994E', alpha=0.7)
        ax5.fill_between(df['time_seconds'], total_own, total_own + total_foreign, 
                         label='Foreign States', color='#E63946', alpha=0.7)
        ax5.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
        ax5.set_ylabel('Number of States', fontsize=10, fontweight='bold')
        ax5.set_title('Own vs Foreign States', fontsize=11, fontweight='bold')
        ax5.legend(loc='best', fontsize=8, framealpha=0.9)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, own_foreign_max * 1.1])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comprehensive dashboard: {output_path}")
    plt.close()

def main():
    """Main execution function"""
    # Parse command line arguments
    home_directory = Path.home()
    parser = argparse.ArgumentParser(
        description='Visualize distributed vector search system metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                                    # Use default input/output directories
  %(prog)s -i /path/to/data -o /path/to/output               # Specify both input and output
  %(prog)s --input-dir ./data --output-dir ./results         # Use relative paths
        """
    )
    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        default=f'{home_directory}/workspace/rdma_anns/logs/',
        help='Directory containing input CSV files with counter_ prefix (default: /mnt/user-data/uploads)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=f'{home_directory}/workspace/rdma_anns/figures/',
        help='Directory to save the output plots (default: /mnt/user-data/outputs)'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all counter CSV files in input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"\n✗ Error: Input directory '{input_dir}' does not exist!")
        return
    
    counter_files = sorted(input_dir.glob('counter_*.csv'))
    
    if not counter_files:
        print(f"\n✗ Error: No files matching 'counter_*.csv' found in '{input_dir}'!")
        return
    
    print("\n" + "="*70)
    print("Distributed Vector Search System - Data Visualization")
    print(f"Filtering warmup period: first {WARMUP_SECONDS} seconds")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(counter_files)} counter file(s)")
    print("="*70 + "\n")
    
    # Load data from all servers
    data_dict = {}
    for file_path in counter_files:
        filename = file_path.name
        server_name = f"Server {filename.replace('counter_', '').replace('.csv', '')}"
        print(f"Loading {filename}... ", end='')
        df = load_server_data(file_path)
        data_dict[server_name] = df
        print(f"✓ ({len(df)} time steps after warmup)")
    
    if not data_dict:
        print("\n✗ Error: No data loaded!")
        return
    
    print(f"\nLoaded data from {len(data_dict)} server(s)")
    print("-" * 70)
    
    # Create individual plots for each server
    print("\nGenerating individual server plots...")
    for server_name, df in data_dict.items():
        output_file = output_dir / f"{server_name.lower().replace(' ', '_')}_plot.png"
        plot_individual_server(df, server_name, str(output_file))
    
    # Create combined comparison plot
    print("\nGenerating combined comparison plot...")
    combined_output = output_dir / "all_servers_combined_plot.png"
    plot_combined_servers(data_dict, str(combined_output))
    
    # Create comprehensive dashboard
    print("\nGenerating comprehensive dashboard...")
    dashboard_output = output_dir / "comprehensive_dashboard.png"
    plot_comprehensive_dashboard(data_dict, str(dashboard_output))
    
    print("\n" + "="*70)
    print("Visualization Complete!")
    print("="*70)
    print(f"\nGenerated {len(data_dict) + 2} plot(s) in {output_dir}:")
    for server_name in data_dict.keys():
        print(f"  • {server_name.lower().replace(' ', '_')}_plot.png")
    print(f"  • all_servers_combined_plot.png")
    print(f"  • comprehensive_dashboard.png (ALL DATA IN ONE VIEW)")
    print()

if __name__ == "__main__":
    main()
