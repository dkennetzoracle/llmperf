#!/usr/bin/env python3
"""
Script to summarize benchmark results from JSON summary files into a table format.
"""

import json
import os
import sys
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict, Any


def load_summary_files(directory: str) -> List[Dict[str, Any]]:
    """Load all summary JSON files from the specified directory."""
    summary_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    # Find all summary JSON files
    for file_path in directory_path.glob("*summary.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                data['filename'] = file_path.name
                summary_files.append(data)
                print(f"Loaded: {file_path.name}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not summary_files:
        print(f"No summary.json files found in {directory}")
        return []
    
    return summary_files


def extract_key_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the most important metrics from a summary file."""
    metrics = {
        # Basic info
        'filename': data.get('filename', 'unknown'),
        'model': data.get('model', 'unknown'),
        'mean_input_tokens': data.get('mean_input_tokens', 0),
        'mean_output_tokens': data.get('mean_output_tokens', 0),
        'num_concurrent_requests': data.get('num_concurrent_requests', 0),
        
        # Performance metrics (rounded to 4 decimal places)
        'ttft_mean_s': round(data.get('results_ttft_s_mean', 0), 4),
        'ttft_p50_s': round(data.get('results_ttft_s_quantiles_p50', 0), 4),
        'ttft_p95_s': round(data.get('results_ttft_s_quantiles_p95', 0), 4),
        
        'inter_token_latency_mean_ms': round(data.get('results_inter_token_latency_s_mean', 0) * 1000, 2),
        'inter_token_latency_p50_ms': round(data.get('results_inter_token_latency_s_quantiles_p50', 0) * 1000, 2),
        'inter_token_latency_p95_ms': round(data.get('results_inter_token_latency_s_quantiles_p95', 0) * 1000, 2),
        
        'e2e_latency_mean_s': round(data.get('results_end_to_end_latency_s_mean', 0), 4),
        'e2e_latency_p50_s': round(data.get('results_end_to_end_latency_s_quantiles_p50', 0), 4),
        'e2e_latency_p95_s': round(data.get('results_end_to_end_latency_s_quantiles_p95', 0), 4),
        
        # Throughput
        'output_throughput_mean_tok_per_s': round(data.get('results_mean_output_throughput_token_per_s', 0), 2),
        'request_throughput_mean_tok_per_s': round(data.get('results_request_output_throughput_token_per_s_mean', 0), 2),
        'requests_per_min': round(data.get('results_num_completed_requests_per_min', 0), 2),
        
        # Actual token counts
        'actual_input_tokens_mean': round(data.get('results_number_input_tokens_mean', 0), 1),
        'actual_output_tokens_mean': round(data.get('results_number_output_tokens_mean', 0), 1),
        
        # Request stats
        'num_completed_requests': data.get('results_num_completed_requests', 0),
        'error_rate': data.get('results_error_rate', 0),
        'num_errors': data.get('results_number_errors', 0),
    }
    
    return metrics


def create_summary_table(summary_data: List[Dict[str, Any]], output_format: str = 'table') -> str:
    """Create a formatted table from the summary data."""
    if not summary_data:
        return "No data to summarize."
    
    # Extract metrics for all files
    all_metrics = [extract_key_metrics(data) for data in summary_data]
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Sort by model, then input tokens, then output tokens, then concurrency
    df = df.sort_values(['model', 'mean_input_tokens', 'mean_output_tokens', 'num_concurrent_requests'])
    
    if output_format.lower() == 'csv':
        return df.to_csv(index=False)
    elif output_format.lower() == 'markdown':
        return df.to_markdown(index=False)
    else:
        # Default table format
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        return str(df.to_string(index=False))


def create_performance_summary(summary_data: List[Dict[str, Any]]) -> str:
    """Create a focused performance summary table with key metrics."""
    if not summary_data:
        return "No data to summarize."
    
    performance_metrics = []
    
    for data in summary_data:
        metrics = {
            'model': data.get('model', 'unknown').split('/')[-1],  # Get just model name
            'input_tokens': data.get('mean_input_tokens', 0),
            'output_tokens': data.get('mean_output_tokens', 0),
            'concurrency': data.get('num_concurrent_requests', 0),
            'ttft_p50_ms': round(data.get('results_ttft_s_quantiles_p50', 0) * 1000, 1),
            'inter_token_p50_ms': round(data.get('results_inter_token_latency_s_quantiles_p50', 0) * 1000, 1),
            'throughput_tok/s': round(data.get('results_mean_output_throughput_token_per_s', 0), 1),
            'requests/min': round(data.get('results_num_completed_requests_per_min', 0), 1),
            'completed_reqs': data.get('results_num_completed_requests', 0),
            'errors': data.get('results_number_errors', 0),
        }
        performance_metrics.append(metrics)
    
    df = pd.DataFrame(performance_metrics)
    df = df.sort_values(['model', 'input_tokens', 'output_tokens', 'concurrency'])
    
    return df.to_string(index=False)


def main():
    parser = argparse.ArgumentParser(description='Summarize benchmark results from JSON files')
    parser.add_argument('directory', help='Directory containing summary JSON files')
    parser.add_argument('--format', choices=['table', 'csv', 'markdown'], default='table',
                       help='Output format (default: table)')
    parser.add_argument('--output', '-o', help='Output file (default: print to stdout)')
    parser.add_argument('--performance-only', action='store_true',
                       help='Show only key performance metrics in a compact format')
    
    args = parser.parse_args()
    
    try:
        # Load all summary files
        print(f"Loading summary files from: {args.directory}")
        summary_data = load_summary_files(args.directory)
        
        if not summary_data:
            print("No summary files found.")
            return 1
        
        print(f"Found {len(summary_data)} summary files\n")
        
        # Generate the appropriate summary
        if args.performance_only:
            result = create_performance_summary(summary_data)
        else:
            result = create_summary_table(summary_data, args.format)
        
        # Output result
        if args.output:
            with open(args.output, 'w') as f:
                f.write(result)
            print(f"Results written to: {args.output}")
        else:
            print(result)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())