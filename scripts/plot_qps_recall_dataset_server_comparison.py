#!/usr/bin/env python3
"""
Script to plot QPS vs Recall curves comparing different datasets and methods.
Creates a grid with dataset sizes (100M, 1B) as columns, dataset names as rows.
All server configurations are plotted in the same subplot with different line/marker styles.
Adds speedup annotations for STATE_SEND vs SCATTER_GATHER and DISTRIBUTED_ANN.
Aggregates runs by taking the average of QPS and Recall for the same L value.
"""

import os
import re
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import hashlib

# Define consistent colors for different methods
METHOD_COLORS = {
    'STATE_SEND': '#1f77b4',      # Blue
    'SCATTER_GATHER': '#ff7f0e',  # Orange
    'SINGLE_SERVER': '#2ca02c',   # Green
    'DISTRIBUTED_ANN': 'red'      # Red
}

def get_color_for_method(method):
    """Fallback to a generated color if the method isnt in the predefined list."""
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    # Generate a consistent hex color based on the method name
    hash_obj = hashlib.md5(method.encode())
    return '#' + hash_obj.hexdigest()[:6]

# Legend name mapping
LEGEND_NAME_MAPPING = {
    'STATE_SEND': 'BatANN',
    'SCATTER_GATHER': 'ScatterGather',
    'SINGLE_SERVER': 'SingleServer',
    'DISTRIBUTED_ANN': 'DistributedANN'
}

# Define line styles for different server configurations
SERVER_LINE_STYLES = {
    1: (0, (1, 1)),         # Densely dotted
    2: ':',                 # Dotted
    3: (0, (3, 1, 1, 1)),   # Densely dashdotted
    4: '-.',                # Dash-dot
    5: '--',                # Dashed
    6: (0, (5, 5)),         # Dashed (longer)
    7: (0, (3, 1, 1, 1, 1, 1)),  # Densely dashdotdotted
    8: (0, (5, 1)),         # Densely dashed
    9: (0, (3, 5, 1, 5)),   # Dashdotted (longer)
    10: '-',                # Solid
}

# Define marker styles for different server configurations
SERVER_MARKERS = {
    1: 'o',      # Circle
    2: 's',      # Square
    3: '^',      # Triangle up
    4: 'D',      # Diamond
    5: 'o',      # Hollow circle (will use fillstyle='none')
    6: 'p',      # Pentagon
    7: '*',      # Star
    8: 'v',      # Triangle down
    9: 'P',      # Plus (filled)
    10: '^',     # Hollow triangle (will use fillstyle='none')
}

# Define which markers should be hollow
HOLLOW_MARKERS = {5, 10}  # Server configs that should have hollow markers


def get_display_name(dataset_name):
    """
    Convert dataset name to display format.
    
    Args:
        dataset_name: Internal dataset name (e.g., 'bigann', 'deep1b')
    
    Returns:
        Display name (e.g., 'BIGANN', 'DEEP')
    """
    name_mapping = {
        'bigann': 'BIGANN',
        'deep1b': 'DEEP',
        'MSSPACEV1B': 'MSSPACEV1B',
        'msspacev1b': 'MSSPACEV1B',
        'text2image1B' : 'text2image',
    }
    
    return name_mapping.get(dataset_name, dataset_name.upper())


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
    method = None
    if folder_name.startswith('logs_'):
        if '_distributed_' in folder_name:
            method = folder_name[5:folder_name.find('_distributed_')]
        elif folder_name.startswith('logs_SINGLE_SERVER'):
            method = 'SINGLE_SERVER'
        else:
            match = re.match(r'logs_([A-Z_]+)_', folder_name)
            if match:
                method = match.group(1)
                
    if not method:
        return None
    
    beamwidth_match = re.search(r'BEAMWIDTH_(\d+)', folder_name)
    if not beamwidth_match:
        return None
    beamwidth = int(beamwidth_match.group(1))
    
    if method != 'SINGLE_SERVER':
        dataset_match = re.search(r'distributed_([A-Za-z0-9]+)_(\d+[BKMG])', folder_name)
        if not dataset_match:
            return None
        dataset_name_with_size = dataset_match.group(1)
        dataset_size = dataset_match.group(2)
        
        dataset_name = re.sub(r'_?\d+[BKMG]$', '', dataset_name_with_size)
        if not dataset_name:
            dataset_name = dataset_name_with_size
        
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
        
        dataset_name = re.sub(r'_?\d+[BKMG]$', '', dataset_name_with_size)
        if not dataset_name:
            dataset_name = dataset_name_with_size
        
        num_threads_match = re.search(r'NUM_SEARCH_THREADS_(\d+)', folder_name)
        if not num_threads_match:
            return None
        num_threads = int(num_threads_match.group(1))
        
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
    Parse client.log file and extract L, QPS and Recall data.
    """
    data_points = []
    
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
        
        header_found = False
        data_start_idx = 0
        
        for i, line in enumerate(lines):
            if 'L   I/O Width' in line and 'QPS' in line and 'Recall' in line:
                header_found = True
                data_start_idx = i + 2
                break
        
        if not header_found:
            print(f"Warning: Header not found in {log_file_path}")
            return data_points
        
        for line in lines[data_start_idx:]:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 9:
                continue
            
            try:
                l_val = int(parts[0])
                qps = float(parts[2])
                avg_latency = float(parts[3])
                recall = float(parts[-1])
                
                data_points.append((l_val, qps, avg_latency, recall / 100.0))
            except (ValueError, IndexError):
                continue
    
    except FileNotFoundError:
        print(f"Warning: File not found: {log_file_path}")
    except Exception as e:
        print(f"Error parsing {log_file_path}: {e}")
    
    return data_points


def collect_data(logs_folder):
    """
    Collect all data from log folder and aggregate results (average) for the same L.
    """
    raw_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    raw_single_server_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    logs_root = Path(logs_folder)
    
    if not logs_root.exists():
        print(f"Error: Root folder '{logs_folder}' does not exist")
        return {}
    
    print(f"Scanning folder: {logs_folder}")
    
    for folder in logs_root.iterdir():
        if not folder.is_dir():
            continue
        
        metadata = parse_folder_name(folder.name)
        if metadata is None:
            continue
        
        client_log = folder / 'client.log'
        if not client_log.exists():
            continue
        
        data_points = parse_client_log(client_log)
        if not data_points:
            continue
        
        dataset_name = metadata['dataset_name']
        dataset_size = metadata['dataset_size']
        num_servers = metadata['num_servers']
        method = metadata['method']
        beamwidth = metadata['beamwidth']
        
        dataset_key = f"{dataset_name}_{dataset_size}"
        
        if method == 'SINGLE_SERVER':
            for l_val, qps, lat, rec in data_points:
                raw_single_server_data[dataset_key][num_servers][(method, beamwidth)][l_val].append((qps, lat, rec))
            print(f"  Loaded {len(data_points)} data points from {folder.name} "
                  f"(dataset={dataset_key}, method={method}, bw={beamwidth}, equiv_servers={num_servers})")
        else:
            for l_val, qps, lat, rec in data_points:
                raw_data[dataset_key][num_servers][(method, beamwidth)][l_val].append((qps, lat, rec))
            print(f"  Loaded {len(data_points)} data points from {folder.name} "
                  f"(dataset={dataset_key}, method={method}, bw={beamwidth}, servers={num_servers})")
    
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    single_server_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for dataset_key, servers_dict in raw_data.items():
        for num_servers, methods_dict in servers_dict.items():
            for mb_key, l_dict in methods_dict.items():
                for l_val, points in l_dict.items():
                    avg_qps = np.mean([p[0] for p in points])
                    avg_lat = np.mean([p[1] for p in points])
                    avg_rec = np.mean([p[2] for p in points])
                    data[dataset_key][num_servers][mb_key].append((avg_qps, avg_lat, avg_rec))
                    
    for dataset_key, servers_dict in raw_single_server_data.items():
        for num_servers, methods_dict in servers_dict.items():
            for mb_key, l_dict in methods_dict.items():
                for l_val, points in l_dict.items():
                    avg_qps = np.mean([p[0] for p in points])
                    avg_lat = np.mean([p[1] for p in points])
                    avg_rec = np.mean([p[2] for p in points])
                    single_server_data[dataset_key][num_servers][mb_key].append((avg_qps, avg_lat, avg_rec))
    
    for dataset_key in list(single_server_data.keys()):
        for num_servers in list(single_server_data[dataset_key].keys()):
            if dataset_key in data and num_servers in data[dataset_key]:
                for key, value in single_server_data[dataset_key][num_servers].items():
                    data[dataset_key][num_servers][key] = value
    
    return data


def plot_comparison_grid(data, global_min_recall, dataset_sizes):
    if not data:
        print("No data to plot!")
        return None
    
    # --- SETUP SPEEDUP SUMMARY COLLECTIONS ---
    # Structure: method -> target_recall -> num_servers -> list of speedups
    global_speedup_summary = {
        'SCATTER_GATHER': defaultdict(lambda: defaultdict(list)),
        'DISTRIBUTED_ANN': defaultdict(lambda: defaultdict(list))
    }
    
    # Structure: dataset -> method -> target_recall -> num_servers -> list of speedups
    dataset_speedup_summary = defaultdict(lambda: {
        'SCATTER_GATHER': defaultdict(lambda: defaultdict(list)),
        'DISTRIBUTED_ANN': defaultdict(lambda: defaultdict(list))
    })
    
    global_methods = set()
    global_servers = set()
    for dataset_key in data:
        for num_servers in data[dataset_key]:
            global_servers.add(num_servers)
            for (method, beamwidth) in data[dataset_key][num_servers]:
                global_methods.add(method)
    
    reorganized_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for dataset_key in data.keys():
        match = re.match(r'(.+)_(\d+[BKMG])$', dataset_key)
        if match:
            base_name = match.group(1)
            dataset_size = match.group(2)
        else:
            base_name = dataset_key
            dataset_size = dataset_sizes.get(dataset_key, 'unknown')
        
        reorganized_data[base_name][dataset_size] = data[dataset_key]
    
    dataset_base_names = sorted(reorganized_data.keys())
    all_dataset_sizes = set()
    
    for base_name in dataset_base_names:
        all_dataset_sizes.update(reorganized_data[base_name].keys())
    
    def size_sort_key(size_str):
        if 'M' in size_str:
            return (0, int(size_str.replace('M', '')))
        elif 'B' in size_str:
            return (1, int(size_str.replace('B', '')))
        return (2, 0)
    
    dataset_size_list = sorted(all_dataset_sizes, key=size_sort_key)
    
    if not dataset_base_names or not dataset_size_list:
        print("No valid data configuration found!")
        return None
    
    num_rows = len(dataset_size_list)
    num_cols = len(dataset_base_names)
    
    print(f"\nCreating grid: {num_rows} rows (dataset sizes) × {num_cols} cols (dataset types)")
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows))
    
    if num_rows == 1 and num_cols == 1:
        axes = [[axes]]
    elif num_rows == 1:
        axes = [axes]
    elif num_cols == 1:
        axes = [[ax] for ax in axes]
    
    dataset_qps_limits = {}
    for base_name in dataset_base_names:
        current_min_recall = 0.2 if base_name in ['text2image1B', 'text2image'] else global_min_recall
        
        for dataset_size in dataset_size_list:
            if dataset_size not in reorganized_data[base_name]:
                continue
            
            qps_values = []
            for num_servers in reorganized_data[base_name][dataset_size].keys():
                for (method, beamwidth), data_points in reorganized_data[base_name][dataset_size][num_servers].items():
                    filtered_points = [(qps, recall) for qps, latency, recall in data_points if recall >= current_min_recall]
                    y_values = [qps for qps, recall in filtered_points]
                    qps_values.extend(y_values)
            
            if qps_values:
                min_qps = min(qps_values)
                max_qps = max(qps_values)
                qps_range = max_qps - min_qps
                y_min = min_qps - 0.05 * qps_range
                y_max = max_qps + 0.05 * qps_range
                dataset_qps_limits[(base_name, dataset_size)] = (y_min, y_max)
            else:
                dataset_qps_limits[(base_name, dataset_size)] = (None, None)
    
    # Plot each cell
    for row_idx, dataset_size in enumerate(dataset_size_list):
        for col_idx, base_name in enumerate(dataset_base_names):
            ax = axes[row_idx][col_idx]
            
            y_min, y_max = dataset_qps_limits.get((base_name, dataset_size), (None, None))
            current_min_recall = 0.2 if base_name in ['text2image1B', 'text2image'] else global_min_recall
            
            if dataset_size not in reorganized_data[base_name]:
                ax.set_visible(False)
                continue
            
            dataset_data = reorganized_data[base_name][dataset_size]
            server_configs = sorted(dataset_data.keys())
            
            state_send_data = {}
            scatter_gather_data = {}
            distributed_ann_data = {}
            
            for num_servers in server_configs:
                methods_data = dataset_data[num_servers]
                
                method_order = ['STATE_SEND', 'SCATTER_GATHER', 'DISTRIBUTED_ANN', 'SINGLE_SERVER']
                sorted_keys = sorted(methods_data.keys(),
                                   key=lambda x: (method_order.index(x[0]) if x[0] in method_order else 999, x[1]))
                
                for (method, beamwidth) in sorted_keys:
                    data_points = methods_data[(method, beamwidth)]
                    
                    x_values = [point[2] for point in data_points if len(point) >= 3]
                    y_values = [point[0] for point in data_points if len(point) >= 3]
                    
                    if x_values:
                        sorted_points = sorted(zip(x_values, y_values))
                        x_values_sorted, y_values_sorted = zip(*sorted_points)
                        
                        if method == 'STATE_SEND':
                            state_send_data[(num_servers, beamwidth)] = list(zip(x_values_sorted, y_values_sorted))
                        elif method == 'SCATTER_GATHER':
                            scatter_gather_data[(num_servers, beamwidth)] = list(zip(x_values_sorted, y_values_sorted))
                        elif method == 'DISTRIBUTED_ANN':
                            distributed_ann_data[(num_servers, beamwidth)] = list(zip(x_values_sorted, y_values_sorted))
                        
                        color = get_color_for_method(method)
                        linestyle = SERVER_LINE_STYLES.get(num_servers, '-')
                        marker = SERVER_MARKERS.get(num_servers, 'o')
                        fillstyle = 'none' if num_servers in HOLLOW_MARKERS else 'full'
                        
                        ax.plot(x_values_sorted, y_values_sorted,
                               marker=marker, linestyle=linestyle,
                               linewidth=2, markersize=8, markeredgewidth=2,
                               fillstyle=fillstyle,
                               color=color,
                               label=f"{method} ({num_servers} srv)")
            
            # --- DYNAMIC TARGET RECALL LOGIC ---
            if base_name in ['text2image1B', 'text2image']:
                target_recalls = [0.7]
            else:
                target_recalls = [0.95]
            
            def interpolate_qps(sorted_points, target_recall):
                if not sorted_points: return None
                if len(sorted_points) == 1: return sorted_points[0][1]
                
                for i in range(len(sorted_points) - 1):
                    r1, q1 = sorted_points[i]
                    r2, q2 = sorted_points[i + 1]
                    if r1 <= target_recall <= r2:
                        if r2 != r1:
                            weight = (target_recall - r1) / (r2 - r1)
                            return q1 + weight * (q2 - q1)
                        return q1
                
                if target_recall < sorted_points[0][0]: return sorted_points[0][1]
                if target_recall > sorted_points[-1][0]: return sorted_points[-1][1]
                return None

            for (ss_num_servers, ss_beamwidth), state_send_points in state_send_data.items():
                
                ss_sorted = sorted(state_send_points, key=lambda p: p[0])
                
                sg_match_points, sg_beamwidth = None, None
                for (sg_num_servers, sg_bw), points in scatter_gather_data.items():
                    if sg_num_servers == ss_num_servers:
                        sg_match_points, sg_beamwidth = points, sg_bw
                        break
                
                da_match_points, da_beamwidth = None, None
                for (da_num_servers, da_bw), points in distributed_ann_data.items():
                    if da_num_servers == ss_num_servers:
                        da_match_points, da_beamwidth = points, da_bw
                        break
                
                sg_sorted = sorted(sg_match_points, key=lambda p: p[0]) if sg_match_points else []
                da_sorted = sorted(da_match_points, key=lambda p: p[0]) if da_match_points else []

                for target_recall in target_recalls:
                    ss_qps = interpolate_qps(ss_sorted, target_recall)
                    if ss_qps is None:
                        continue
                        
                    if sg_match_points:
                        sg_qps = interpolate_qps(sg_sorted, target_recall)
                        if sg_qps is not None and sg_qps > 0:
                            speedup_sg = ss_qps / sg_qps
                            # Store in dicts mapping by target_recall
                            global_speedup_summary['SCATTER_GATHER'][target_recall][ss_num_servers].append(speedup_sg)
                            dataset_speedup_summary[base_name]['SCATTER_GATHER'][target_recall][ss_num_servers].append(speedup_sg)
                            
                            print(f"  {base_name} ({dataset_size}) - {ss_num_servers} server(s) - Recall@{target_recall}: "
                                  f"{speedup_sg:.3f}x speedup vs SCATTER_GATHER (SS bw={ss_beamwidth}: {ss_qps:.2f} QPS, SG bw={sg_beamwidth}: {sg_qps:.2f} QPS)")
                    
                    if da_match_points:
                        da_qps = interpolate_qps(da_sorted, target_recall)
                        if da_qps is not None and da_qps > 0:
                            speedup_da = ss_qps / da_qps
                            # Store in dicts mapping by target_recall
                            global_speedup_summary['DISTRIBUTED_ANN'][target_recall][ss_num_servers].append(speedup_da)
                            dataset_speedup_summary[base_name]['DISTRIBUTED_ANN'][target_recall][ss_num_servers].append(speedup_da)
                            
                            print(f"  {base_name} ({dataset_size}) - {ss_num_servers} server(s) - Recall@{target_recall}: "
                                  f"{speedup_da:.3f}x speedup vs DISTRIBUTED_ANN (SS bw={ss_beamwidth}: {ss_qps:.2f} QPS, DA bw={da_beamwidth}: {da_qps:.2f} QPS)")
            
            if row_idx == 0:
                display_name = get_display_name(base_name)
                ax.set_title(display_name, fontsize=14, fontweight='bold')
            
            if col_idx == 0:
                ax.set_ylabel('QPS', fontsize=13, fontweight='bold')
            else:
                ax.set_ylabel('QPS', fontsize=12)
            
            ax.set_xlim(current_min_recall, 1.01)
            
            import numpy as np
            if current_min_recall <= 0.2:
                x_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
            elif current_min_recall <= 0.8:
                x_ticks = [0.80, 0.85, 0.90, 0.95, 1.00]
            elif current_min_recall <= 0.85:
                x_ticks = [0.85, 0.90, 0.95, 1.00]
            elif current_min_recall <= 0.90:
                x_ticks = [0.90, 0.95, 1.00]
            else:
                x_ticks = [0.95, 1.00]
            ax.set_xticks(x_ticks)
            
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
            
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=11)
            
            if col_idx == 0:
                from matplotlib.lines import Line2D
                legend_elements = []
                method_display_order = ['STATE_SEND', 'SCATTER_GATHER', 'SINGLE_SERVER', 'DISTRIBUTED_ANN']
                
                for m in global_methods:
                    if m not in method_display_order:
                        method_display_order.append(m)
                
                methods_shown = set()
                for method in method_display_order:
                    if method in global_methods:
                        color = get_color_for_method(method)
                        display_name = LEGEND_NAME_MAPPING.get(method, method)
                        legend_elements.append(
                            Line2D([0], [0], color=color, linewidth=2, 
                                  linestyle='-', marker='',
                                  label=display_name)
                        )
                        methods_shown.add(method)
                
                if methods_shown:
                    legend_elements.append(Line2D([0], [0], color='none', linewidth=0, label=''))
                
                for num_servers in sorted(global_servers):
                    linestyle = SERVER_LINE_STYLES.get(num_servers, '-')
                    marker = SERVER_MARKERS.get(num_servers, 'o')
                    fillstyle = 'none' if num_servers in HOLLOW_MARKERS else 'full'
                    legend_elements.append(
                        Line2D([0], [0], color='black', linewidth=2, 
                              linestyle=linestyle, marker=marker, markersize=8,
                              markeredgewidth=2, fillstyle=fillstyle,
                              label=f"{num_servers} server{'s' if num_servers > 1 else ''}")
                    )
                
                ax.legend(handles=legend_elements, fontsize=14, loc='best')
            
            if row_idx == num_rows - 1:
                ax.set_xlabel('Recall', fontsize=12)

    # =========================================================================
    # PRINT SUMMARY BLOCK
    # =========================================================================
    print("\n" + "="*70)
    print("SPEEDUP RANGE SUMMARY (STATE_SEND vs Baselines)")
    print("="*70)
    
    # Extract all distinct target recalls globally
    all_global_recalls = set()
    for method in ['SCATTER_GATHER', 'DISTRIBUTED_ANN']:
        all_global_recalls.update(global_speedup_summary[method].keys())

    # 1. Global summary (Aggregated across datasets, separated by recall)
    print("\nOVERALL RANGES (Grouped by Target Recall and Server Configuration):")
    for tr in sorted(all_global_recalls):
        print(f"\n  TARGET RECALL @ {tr}:")
        
        # Get all servers that have data for this recall
        servers_for_tr = set()
        for method in ['SCATTER_GATHER', 'DISTRIBUTED_ANN']:
            servers_for_tr.update(global_speedup_summary[method][tr].keys())
            
        for num_servers in sorted(servers_for_tr):
            print(f"    {num_servers} Server Configuration:")
            
            sg_s = global_speedup_summary['SCATTER_GATHER'][tr].get(num_servers, [])
            if sg_s:
                print(f"      vs SCATTER_GATHER:  {min(sg_s):.2f}x - {max(sg_s):.2f}x")
                
            da_s = global_speedup_summary['DISTRIBUTED_ANN'][tr].get(num_servers, [])
            if da_s:
                print(f"      vs DISTRIBUTED_ANN: {min(da_s):.2f}x - {max(da_s):.2f}x")
            
    # 2. Per-Dataset Summary (Separated by recall)
    print("\n" + "-"*70)
    print("RANGES PER DATASET:")
    for dataset_name in sorted(dataset_speedup_summary.keys()):
        print(f"\n  Dataset: {dataset_name}")
        ds_summary = dataset_speedup_summary[dataset_name]
        
        all_ds_recalls = set()
        for method in ['SCATTER_GATHER', 'DISTRIBUTED_ANN']:
            all_ds_recalls.update(ds_summary[method].keys())
            
        for tr in sorted(all_ds_recalls):
            print(f"    Target Recall @ {tr}:")
            
            ds_servers = set()
            for method in ['SCATTER_GATHER', 'DISTRIBUTED_ANN']:
                ds_servers.update(ds_summary[method][tr].keys())
                
            for num_servers in sorted(ds_servers):
                print(f"      {num_servers} Server Configuration:")
                
                sg_s = ds_summary['SCATTER_GATHER'][tr].get(num_servers, [])
                if sg_s:
                    print(f"        vs SCATTER_GATHER:  {min(sg_s):.2f}x - {max(sg_s):.2f}x")
                    
                da_s = ds_summary['DISTRIBUTED_ANN'][tr].get(num_servers, [])
                if da_s:
                    print(f"        vs DISTRIBUTED_ANN: {min(da_s):.2f}x - {max(da_s):.2f}x")
    print("="*70 + "\n")

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot QPS vs Recall comparison grid with all server configs in same plot'
    )
    parser.add_argument(
        'logs_folder',
        type=str,
        help='Path to the root folder containing log subfolders'
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
        default='dataset_comparison_combined.png',
        help='Output filename for the plot (default: dataset_comparison_combined.png)'
    )
    
    args = parser.parse_args()
    
    print(f"Collecting data from: {args.logs_folder}")
    data = collect_data(args.logs_folder)
    
    if not data:
        print("No data collected. Please check your log folder structure.")
        return
    
    dataset_sizes = {}
    for dataset_key in data.keys():
        match = re.match(r'(.+)_(\d+[BKMG])$', dataset_key)
        if match:
            dataset_sizes[dataset_key] = match.group(2)
        else:
            for num_servers in data[dataset_key].keys():
                for (method, beamwidth), _ in data[dataset_key][num_servers].items():
                    logs_root = Path(args.logs_folder)
                    for folder in logs_root.iterdir():
                        if not folder.is_dir():
                            continue
                        metadata = parse_folder_name(folder.name)
                        if metadata:
                            check_key = f"{metadata['dataset_name']}_{metadata['dataset_size']}"
                            if check_key == dataset_key:
                                dataset_sizes[dataset_key] = metadata['dataset_size']
                                break
                    if dataset_key in dataset_sizes:
                        break
                if dataset_key in dataset_sizes:
                    break
    
    print(f"\nGenerating plot...")
    fig = plot_comparison_grid(data, args.min_recall, dataset_sizes)
    
    if fig:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {args.output}")
    else:
        print("Failed to generate plot.")


if __name__ == '__main__':
    main()
