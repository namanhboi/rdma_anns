#!/usr/bin/env python3
"""
Script to analyze and visualize performance logs for multiple timing metrics.

Each log line example:
  [80527369382422] [0] [QUERIES]:BEGIN_DESERIALIZE
  [80527369663312] [0] [QUERIES]:NUM_MSG 1
  [80527369769443] [0] [QUERIES]:END_DESERIALIZE
  [91057025693416] [30] [STATES]:BEGIN_PQ_POPULATE
  [91057025704306] [30] [STATES]:END_PQ_POPULATE

Tracks timing metrics:
  - HANDLER (BEGIN_HANDLER to END_HANDLER)
  - DESERIALIZE (BEGIN_DESERIALIZE to END_DESERIALIZE)
  - DESERIALIZE_STATE (BEGIN_DESERIALIZE_STATE to END_DESERIALIZE_STATE)
  - DESERIALIZE_VISITED_STATE (BEGIN_DESERIALIZE_VISITED_STATE to END_DESERIALIZE_VISITED_STATE)
  - PQ_POPULATE (BEGIN_PQ_POPULATE to END_PQ_POPULATE)
  - CREATE_STATE (BEGIN_CREATE_STATE to END_CREATE_STATE)
  - ENQUEUE_STATE (BEGIN_ENQUEUE_STATE to END_ENQUEUE_STATE)
  - QUERY_MAP_INSERT (BEGIN_QUERY_MAP_INSERT to END_QUERY_MAP_INSERT)
  - QUERY_MAP_ERASE (BEGIN_QUERY_MAP_ERASE to END_QUERY_MAP_ERASE)

It computes:
  - Duration statistics for each metric
  - NUM_MSG statistics
  - Summary plots and CSV output
"""

import argparse
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv

# Corrected regex: timestamp first, query_id second
# [CHANGED] — generalize to match *any* event with numeric value, not just NUM_MSG
RE_WITH_VALUE = re.compile(r'\[(\d+)\]\s+\[(\d+)\]\s+\[(\w+)\]:(\w+)\s+(\d+)')
RE_NO_VALUE   = re.compile(r'\[(\d+)\]\s+\[(\d+)\]\s+\[(\w+)\]:(\w+)')

def parse_log_file(filepath):
    """Parse a single log file and return timing results per (query_id, msg_type)."""
    results = defaultdict(lambda: defaultdict(dict))
    line_count = 0
    matched = 0

    with open(filepath, 'r') as f:
        for line in f:
            line_count += 1
            if match := RE_WITH_VALUE.search(line):
                timestamp, query_id, msg_type, action, value = match.groups()
                key = (query_id, msg_type)
                results[key][action] = int(value)
                results[key].setdefault('timestamps', []).append(int(timestamp))
                matched += 1
            elif match := RE_NO_VALUE.search(line):
                timestamp, query_id, msg_type, action = match.groups()
                key = (query_id, msg_type)
                results[key][action] = int(timestamp)
                results[key].setdefault('timestamps', []).append(int(timestamp))
                matched += 1

    print(f"Parsed {filepath}: {matched}/{line_count} lines matched.")
    return results

def aggregate_results(log_files):
    """Aggregate results across multiple log files."""
    all_results = []
    seen_types = set()
    
    # Define all the timing pairs we want to track
    timing_pairs = [
        ('BEGIN_HANDLER', 'END_HANDLER', 'handler'),
        ('BEGIN_DESERIALIZE', 'END_DESERIALIZE', 'deserialize'),
        ('BEGIN_DESERIALIZE_STATE', 'END_DESERIALIZE_STATE', 'deserialize_state'),
        ('BEGIN_DESERIALIZE_VISITED_STATE', 'END_DESERIALIZE_VISITED_STATE', 'deserialize_visited_state'),
        ('BEGIN_DESERIALIZE_QUERY_EMB', 'END_DESERIALIZE_QUERY_EMB', 'deserialize_query_emb_state'),
        ('BEGIN_DESERIALIZE_FULL_RETSET', 'END_DESERIALIZE_FULL_RETSET', 'deserialize_full_retset'),
        ('BEGIN_DESERIALIZE_RETSET', 'END_DESERIALIZE_RETSET', 'deserialize_retset'),        
        ('BEGIN_PQ_POPULATE', 'END_PQ_POPULATE', 'pq_populate'),
        ('BEGIN_CREATE_STATE', 'END_CREATE_STATE', 'create_state'),
        ('BEGIN_ENQUEUE_STATE', 'END_ENQUEUE_STATE', 'enqueue_state'),
        ('BEGIN_QUERY_MAP_INSERT', 'END_QUERY_MAP_INSERT', 'query_map_insert'),
        ('BEGIN_QUERY_MAP_ERASE', 'END_QUERY_MAP_ERASE', 'query_map_erase'),
        ('BEGIN_COPY_QUERY_EMB', 'END_COPY_QUERY_EMB', 'copy_query_emb'),
        ('BEGIN_DESERIALIZE_QUERY', 'END_DESERIALIZE_QUERY', 'deserialize_query'),
        ('BEGIN_ALLOCATE_QUERY', 'END_ALLOCATE_QUERY', 'allocate_query'),
        ('BEGIN_ALLOCATE_STATE', 'END_ALLOCATE_STATE', 'allocate_state')
    ]

    for log_file_idx, log_file in enumerate(log_files):
        log_data = parse_log_file(log_file)

        for (query_id, msg_type), data in log_data.items():
            num_msg = data.get('NUM_MSG', None)
            # [ADDED] Support VISITED_STATE numeric event
            visited_state = data.get('VISITED_STATE', None)

            for begin_key, end_key, metric_name in timing_pairs:
                if begin_key in data and end_key in data:
                    duration = data[end_key] - data[begin_key]
                    all_results.append({
                        'log_file_id': log_file_idx,
                        'query_id': int(query_id),
                        'type': msg_type,
                        'metric': metric_name,
                        'duration': duration,
                        'num_msg': num_msg,
                        'visited_state': visited_state,  # [ADDED]
                        'timestamp': data[begin_key]
                    })
                    seen_types.add(msg_type)

    print(f"\nFound {len(all_results)} total results across {len(seen_types)} message types: {', '.join(sorted(seen_types))}")
    return all_results

def compute_statistics(all_results):
    """Compute duration and NUM_MSG statistics per message type and metric."""
    stats = defaultdict(lambda: {'durations': [], 'num_msgs': []})
    for r in all_results:
        key = (r['type'], r['metric'])
        stats[key]['durations'].append(r['duration'])
        if r['num_msg'] is not None:
            stats[key]['num_msgs'].append(r['num_msg'])
    return stats


def print_statistics(stats):
    """Print readable summary statistics."""
    print("\n=== Timing and NUM_MSG Statistics ===")
    if not stats:
        print("No data found.")
        return
    for (msg_type, metric), data in sorted(stats.items()):
        durations = data['durations']
        num_msgs = data['num_msgs']

        print(f"\nMessage Type: {msg_type} | Metric: {metric}")
        if durations:
            # Convert nanoseconds to microseconds
            durations_us = [d / 1e3 for d in durations]
            print(f"  Count: {len(durations_us)}")
            print(f"  Mean duration:  {statistics.mean(durations_us):.2f} µs")
            print(f"  Median:         {statistics.median(durations_us):.2f} µs")
            print(f"  Min/Max:        {min(durations_us):.2f} / {max(durations_us):.2f} µs")
            if len(durations_us) > 1:
                print(f"  Stddev:         {statistics.stdev(durations_us):.2f} µs")

        if num_msgs:
            print("  NUM_MSG values:")
            print(f"    Mean:   {statistics.mean(num_msgs):.2f}")
            print(f"    Median: {statistics.median(num_msgs):.2f}")
            print(f"    Min/Max:{min(num_msgs)} / {max(num_msgs)}")
        else:
            print("  NUM_MSG: No data")


def export_csv(all_results, output_dir):
    """Save results to CSV."""
    if not all_results:
        return
    os.makedirs(output_dir, exist_ok=True)
    csv_path = Path(output_dir) / "timing_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['log_file_id', 'query_id', 'type', 'metric', 'duration', 'num_msg', 'timestamp'])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved CSV to {csv_path}")


def plot_deserialization_times(all_results, output_dir):
    """Plot timing durations and numeric events (NUM_MSG and VISITED_STATE)."""
    if not all_results:
        print("No data to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    grouped_by_log = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in all_results:
        grouped_by_log[r["log_file_id"]][r["type"]][r["metric"]].append(r)

    for log_file_id, type_dict in grouped_by_log.items():
        for msg_type, metrics_dict in type_dict.items():
            num_metrics = len(metrics_dict)
            if num_metrics == 0:
                continue

            # [CHANGED] Check for any numeric events (NUM_MSG or VISITED_STATE)
            has_numeric = any(
                any((r["num_msg"] is not None or r.get("visited_state") is not None) for r in records)
                for records in metrics_dict.values()
            )

            # Calculate global y-axis limits for durations
            all_durations_us = [r["duration"] / 1e3 for recs in metrics_dict.values() for r in recs]
            y_min, y_max = (min(all_durations_us), max(all_durations_us)) if all_durations_us else (0, 1)
            y_range = y_max - y_min
            y_min_plot = y_min - 0.05 * y_range
            y_max_plot = y_max + 0.05 * y_range

            # Calculate global x-axis limits (timestamps)
            all_timestamps = [r["timestamp"] / 1e9 for recs in metrics_dict.values() for r in recs]
            x_min = min(all_timestamps) if all_timestamps else 0
            x_max = max(all_timestamps) if all_timestamps else 1
            x_range = x_max - x_min
            x_min_plot = x_min - 0.02 * x_range
            x_max_plot = x_max + 0.02 * x_range

            if has_numeric:
                fig = plt.figure(figsize=(6 * num_metrics, 12))
                gs = fig.add_gridspec(3, num_metrics, hspace=0.3, wspace=0.3)
                duration_axes = [fig.add_subplot(gs[0, i]) for i in range(num_metrics)]
                num_msg_ax = fig.add_subplot(gs[1, 0])
                visited_ax = fig.add_subplot(gs[2, 0])
            else:
                fig, duration_axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
                if num_metrics == 1:
                    duration_axes = [duration_axes]

            # --- Duration plots ---
            for col_idx, (metric, records) in enumerate(sorted(metrics_dict.items())):
                records_sorted = sorted(records, key=lambda x: x["timestamp"])
                x = [r["timestamp"] / 1e9 for r in records_sorted]
                y = [r["duration"] / 1e3 for r in records_sorted]

                ax = duration_axes[col_idx]
                ax.scatter(x, y, alpha=0.6, s=10, color='C0')
                ax.set_xlabel("Timestamp (s)")
                ax.set_ylabel("Duration (µs)")
                ax.set_title(metric)
                ax.set_xlim(x_min_plot, x_max_plot)  # Set shared x-axis limits
                ax.set_ylim(y_min_plot, y_max_plot)  # Set shared y-axis limits
                ax.grid(True, alpha=0.3)

            # --- NUM_MSG plot ---
            if has_numeric:
                all_records = [r for recs in metrics_dict.values() for r in recs]
                num_records = [r for r in all_records if r["num_msg"] is not None]
                if num_records:
                    num_records = sorted(num_records, key=lambda r: r["timestamp"])
                    x = [r["timestamp"] / 1e9 for r in num_records]
                    y = [r["num_msg"] for r in num_records]
                    num_msg_ax.scatter(x, y, s=10, color='C1', alpha=0.6)
                num_msg_ax.set_title("NUM_MSG")
                num_msg_ax.set_xlabel("Timestamp (s)")
                num_msg_ax.set_ylabel("Count")
                num_msg_ax.set_xlim(x_min_plot, x_max_plot)  # Align x-axis with duration plots
                num_msg_ax.grid(True, alpha=0.3)

                # --- VISITED_STATE plot [ADDED] ---
                visited_records = [r for r in all_records if r.get("visited_state") is not None]
                if visited_records:
                    visited_records = sorted(visited_records, key=lambda r: r["timestamp"])
                    x = [r["timestamp"] / 1e9 for r in visited_records]
                    y = [r["visited_state"] for r in visited_records]
                    visited_ax.scatter(x, y, s=10, color='C2', alpha=0.6)
                visited_ax.set_title("VISITED_STATE")
                visited_ax.set_xlabel("Timestamp (s)")
                visited_ax.set_ylabel("Count")
                visited_ax.set_xlim(x_min_plot, x_max_plot)  # Align x-axis with duration plots
                visited_ax.grid(True, alpha=0.3)

            fig.suptitle(f"Timing Analysis - Log {log_file_id} - {msg_type}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            fig_path = Path(output_dir) / f"timing_analysis_log{log_file_id}_{msg_type}.png"
            plt.savefig(fig_path, dpi=100)
            plt.close()
            print(f"Saved plot for log {log_file_id} - {msg_type} to {fig_path}")

def plot_duration_distributions(all_results, output_dir):
    """Plot duration distribution histograms for each metric and message type."""
    if not all_results:
        print("No data to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Group results by log_file_id, message type, and metric
    grouped_by_log = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in all_results:
        grouped_by_log[r["log_file_id"]][r["type"]][r["metric"]].append(r)

    # Create distribution plots for each log file and message type
    for log_file_id, type_dict in grouped_by_log.items():
        for msg_type, metrics_dict in type_dict.items():
            num_metrics = len(metrics_dict)
            
            if num_metrics == 0:
                continue
            
            # First pass: find the global x and y-axis ranges for all histograms
            max_frequency = 0
            all_durations = []
            for records in metrics_dict.values():
                durations_us = [r["duration"] / 1e3 for r in records]
                all_durations.extend(durations_us)
            
            # Calculate global x-axis limits using 99.9th percentile to exclude extreme outliers
            x_min = min(all_durations) if all_durations else 0
            x_max = np.percentile(all_durations, 99.9) if all_durations else 1
            x_range = x_max - x_min
            x_min_plot = x_min - 0.05 * x_range
            x_max_plot = x_max + 0.05 * x_range
            
            # Create consistent bin edges for all histograms (more bins = narrower bars)
            bin_edges = np.linspace(x_min, x_max, 301)  # 300 bins for very narrow, highly accurate bars
            
            # Calculate max frequency with consistent bins
            for records in metrics_dict.values():
                durations_us = [r["duration"] / 1e3 for r in records]
                counts, _ = np.histogram(durations_us, bins=bin_edges)
                max_frequency = max(max_frequency, counts.max())
            
            # Create figure with histograms stacked vertically
            fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))
            if num_metrics == 1:
                axes = [axes]  # Ensure iterable even with 1 metric
            
            # Plot histogram for each metric
            # Sort metrics with handler always first
            def sort_key(item):
                metric, records = item
                return (0 if metric == 'handler' else 1, metric)
            
            for row_idx, (metric, records) in enumerate(sorted(metrics_dict.items(), key=sort_key)):
                ax = axes[row_idx]
                
                # Convert durations to microseconds
                durations_us = [r["duration"] / 1e3 for r in records]
                
                # Create histogram with consistent bin edges
                ax.hist(durations_us, bins=bin_edges, alpha=0.7, color='C0', edgecolor='black')
                ax.set_xlabel("Duration (µs)")
                ax.set_ylabel("Frequency")
                ax.set_title(f"{metric.replace('_', ' ').title()}")
                ax.set_xlim(x_min_plot, x_max_plot)  # Set shared x-axis limits
                ax.set_ylim(0, max_frequency * 1.1)  # Set shared y-axis limits with 10% padding
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add statistics text box with percentiles
                mean_val = statistics.mean(durations_us)
                median_val = statistics.median(durations_us)
                p95 = np.percentile(durations_us, 95)
                p99 = np.percentile(durations_us, 99)
                stats_text = f"Mean: {mean_val:.2f} µs\nMedian: {median_val:.2f} µs\n95th: {p95:.2f} µs\n99th: {p99:.2f} µs\nCount: {len(durations_us)}"
                ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=9)
            
            # Add overall title
            fig.suptitle(f"Duration Distributions - Log {log_file_id} - {msg_type}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Create safe filename
            safe_msg_type = msg_type.replace('/', '_').replace('\\', '_')
            fig_path = Path(output_dir) / f"duration_distribution_log{log_file_id}_{safe_msg_type}.png"
            plt.savefig(fig_path, dpi=100)
            plt.close()
            print(f"Saved distribution plot for log {log_file_id} - {msg_type} to {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze timing logs for multiple metrics")
    parser.add_argument('log_dir', help='Directory containing .log files')
    parser.add_argument('output_dir', nargs='?', default='timing_output',
                        help='Directory to save results (default: timing_output)')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: {log_dir} not found")
        return

    log_files = [str(f) for f in log_dir.glob("*.txt")]
    if not log_files:
        print(f"No .log files found in {log_dir}")
        return
    print(log_files)
    all_results = aggregate_results(log_files)
    stats = compute_statistics(all_results)
    print_statistics(stats)
    plot_deserialization_times(all_results, args.output_dir)
    plot_duration_distributions(all_results, args.output_dir)


if __name__ == "__main__":
    main()
