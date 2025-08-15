#!/usr/bin/env python3
"""
Comprehensive comparison script for TP2 vs TP8 LLM performance results.
Generates a visualization matrix comparing latency, throughput, cost, and price-performance metrics.
"""

import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from typing import Dict, List, Tuple

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class LLMPerfComparator:
    def __init__(self, tp2_dir: str, tp8_dir: str):
        self.tp2_dir = Path(tp2_dir)
        self.tp8_dir = Path(tp8_dir)
        self.data = {}
        
    def parse_filename(self, filename: str) -> Dict:
        """Parse configuration from filename: openai-gpt-oss-120_{input}_{output}_{concurrent}_summary.json"""
        pattern = r'openai-gpt-oss-120_(\d+)_(\d+)_(\d+)_summary\.json'
        match = re.match(pattern, filename)
        if match:
            return {
                'input_tokens': int(match.group(1)),
                'output_tokens': int(match.group(2)),
                'concurrent_requests': int(match.group(3))
            }
        return None
    
    def load_data(self):
        """Load all summary JSON files from both TP directories"""
        print("Loading data from TP2 and TP8 directories...")
        
        # Load TP2 data
        tp2_files = glob.glob(str(self.tp2_dir / "*_summary.json"))
        for file_path in tp2_files:
            filename = os.path.basename(file_path)
            config = self.parse_filename(filename)
            if config:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    key = (config['input_tokens'], config['output_tokens'], config['concurrent_requests'])
                    if key not in self.data:
                        self.data[key] = {}
                    self.data[key]['tp2'] = data
        
        # Load TP8 data
        tp8_files = glob.glob(str(self.tp8_dir / "*_summary.json"))
        for file_path in tp8_files:
            filename = os.path.basename(file_path)
            config = self.parse_filename(filename)
            if config:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    key = (config['input_tokens'], config['output_tokens'], config['concurrent_requests'])
                    if key not in self.data:
                        self.data[key] = {}
                    self.data[key]['tp8'] = data
        
        # Filter to only include configurations that exist in both TP2 and TP8
        complete_configs = {k: v for k, v in self.data.items() if 'tp2' in v and 'tp8' in v}
        self.data = complete_configs
        
        print(f"Loaded {len(self.data)} complete configuration comparisons")
        for config in sorted(self.data.keys()):
            print(f"  Input: {config[0]}, Output: {config[1]}, Concurrent: {config[2]}")
    
    def extract_metrics(self) -> pd.DataFrame:
        """Extract key metrics for comparison"""
        rows = []
        
        for config, data in self.data.items():
            input_tokens, output_tokens, concurrent = config
            
            for tp_type in ['tp2', 'tp8']:
                if tp_type not in data:
                    continue
                    
                d = data[tp_type]
                
                # Calculate total tokens and cost metrics
                total_tokens = d.get('results_number_input_tokens_mean', 0) + d.get('results_number_output_tokens_mean', 0)
                
                row = {
                    'config': f"{input_tokens}_{output_tokens}_{concurrent}",
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'concurrent_requests': concurrent,
                    'tp_type': tp_type,
                    
                    # Latency metrics (in seconds)
                    'ttft_mean': d.get('results_ttft_s_mean', 0),
                    'ttft_p95': d.get('results_ttft_s_quantiles_p95', 0),
                    'inter_token_latency_mean': d.get('results_inter_token_latency_s_mean', 0),
                    'end_to_end_latency_mean': d.get('results_end_to_end_latency_s_mean', 0),
                    'end_to_end_latency_p95': d.get('results_end_to_end_latency_s_quantiles_p95', 0),
                    
                    # Throughput metrics
                    'request_throughput_mean': d.get('results_request_output_throughput_token_per_s_mean', 0),
                    'output_throughput_mean': d.get('results_mean_output_throughput_token_per_s', 0),
                    'requests_per_min': d.get('results_num_completed_requests_per_min', 0),
                    
                    # Performance and reliability
                    'error_rate': d.get('results_error_rate', 0),
                    'completed_requests': d.get('results_num_completed_requests', 0),
                    
                    # Token statistics
                    'actual_input_tokens': d.get('results_number_input_tokens_mean', 0),
                    'actual_output_tokens': d.get('results_number_output_tokens_mean', 0),
                    'total_tokens': total_tokens,
                }
                
                # Calculate efficiency metrics
                if row['end_to_end_latency_mean'] > 0:
                    row['tokens_per_second_efficiency'] = total_tokens / row['end_to_end_latency_mean']
                else:
                    row['tokens_per_second_efficiency'] = 0
                
                # Calculate cost-based price-performance metrics
                # Cost per hour: $10/GPU/hour, TP2 = 2 GPUs ($20/hour), TP8 = 8 GPUs ($80/hour)
                gpu_count = 2 if tp_type == 'tp2' else 8
                cost_per_hour = gpu_count * 10.0  # $10 per GPU per hour
                
                # Calculate cost per million tokens based on throughput capacity
                # If throughput is X tokens/sec, then in 1 hour you generate X * 3600 tokens
                # Cost per million tokens = (cost_per_hour / tokens_per_hour) * 1,000,000
                if row['output_throughput_mean'] > 0:
                    tokens_per_hour = row['output_throughput_mean'] * 3600.0
                    row['cost_per_million_tokens'] = (cost_per_hour / tokens_per_hour) * 1_000_000
                    row['tokens_per_dollar'] = tokens_per_hour / cost_per_hour
                else:
                    row['cost_per_million_tokens'] = 0
                    row['tokens_per_dollar'] = 0
                
                # Price-performance score: tokens per second per dollar of hourly cost
                # This represents how many tokens/sec you get for each dollar spent per hour
                if cost_per_hour > 0:
                    row['price_performance_score'] = row['output_throughput_mean'] / cost_per_hour
                else:
                    row['price_performance_score'] = 0
                
                # Calculate cost per request based on actual request latency (alternative metric)
                latency_hours = row['end_to_end_latency_mean'] / 3600.0
                cost_per_request = cost_per_hour * latency_hours
                
                # Store cost information
                row['gpu_count'] = gpu_count
                row['cost_per_hour'] = cost_per_hour
                row['cost_per_request'] = cost_per_request
                row['tokens_per_hour'] = row['output_throughput_mean'] * 3600.0 if row['output_throughput_mean'] > 0 else 0
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_comparison_plots(self, df: pd.DataFrame) -> plt.Figure:
        """Create comprehensive comparison visualization matrix"""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('LLM Performance Comparison: TP2 vs TP8\nComprehensive Analysis Across Multiple Metrics', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Define the grid layout (4 rows x 3 columns)
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Helper function to create comparison plots
        def create_comparison_subplot(ax, metric, title, ylabel, configs_to_plot=None):
            if configs_to_plot is None:
                plot_df = df
            else:
                plot_df = df[df['config'].isin(configs_to_plot)]
            
            pivot_data = plot_df.pivot_table(
                index='config', 
                columns='tp_type', 
                values=metric, 
                aggfunc='mean'
            )
            
            if pivot_data.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                return
            
            x = np.arange(len(pivot_data.index))
            width = 0.35
            
            tp2_values = pivot_data['tp2'].fillna(0).values
            tp8_values = pivot_data['tp8'].fillna(0).values
            
            bars1 = ax.bar(x - width/2, tp2_values, width, label='TP2', alpha=0.8, color='#1f77b4')
            bars2 = ax.bar(x + width/2, tp8_values, width, label='TP8', alpha=0.8, color='#ff7f0e')
            
            # Add value labels on bars
            for bar in bars1:
                if bar.get_height() > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
            
            for bar in bars2:
                if bar.get_height() > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Configuration (Input_Output_Concurrent)')
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(pivot_data.index, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 1. Latency Comparisons
        ax1 = fig.add_subplot(gs[0, 0])
        create_comparison_subplot(ax1, 'ttft_mean', 'Time to First Token (TTFT)', 'Seconds')
        
        ax2 = fig.add_subplot(gs[0, 1])
        create_comparison_subplot(ax2, 'end_to_end_latency_mean', 'End-to-End Latency (Mean)', 'Seconds')
        
        ax3 = fig.add_subplot(gs[0, 2])
        create_comparison_subplot(ax3, 'inter_token_latency_mean', 'Inter-token Latency (Mean)', 'Seconds')
        
        # 2. Throughput Comparisons
        ax4 = fig.add_subplot(gs[1, 0])
        create_comparison_subplot(ax4, 'output_throughput_mean', 'Output Throughput', 'Tokens/Second')
        
        ax5 = fig.add_subplot(gs[1, 1])
        create_comparison_subplot(ax5, 'request_throughput_mean', 'Request Throughput', 'Tokens/Second')
        
        ax6 = fig.add_subplot(gs[1, 2])
        create_comparison_subplot(ax6, 'requests_per_min', 'Requests per Minute', 'Requests/Min')
        
        # 3. Cost and Performance Efficiency
        ax7 = fig.add_subplot(gs[2, 0])
        create_comparison_subplot(ax7, 'cost_per_million_tokens', 'Cost per Million Tokens', 'USD')
        
        ax8 = fig.add_subplot(gs[2, 1])
        create_comparison_subplot(ax8, 'price_performance_score', 'Price-Performance Score\n(Tokens/sec per $/hour)', 'Tokens/sec/$')
        
        ax9 = fig.add_subplot(gs[2, 2])
        create_comparison_subplot(ax9, 'tokens_per_dollar', 'Tokens per Dollar', 'Tokens/$')
        
        # 4. Create summary comparison charts
        ax10 = fig.add_subplot(gs[3, :2])
        
        # Summary performance by configuration size
        summary_df = df.groupby(['tp_type', 'input_tokens', 'output_tokens']).agg({
            'end_to_end_latency_mean': 'mean',
            'output_throughput_mean': 'mean',
            'price_performance_score': 'mean'
        }).reset_index()
        
        summary_df['config_size'] = summary_df['input_tokens'].astype(str) + '_' + summary_df['output_tokens'].astype(str)
        
        # Create a grouped bar chart for summary metrics
        config_sizes = summary_df['config_size'].unique()
        x_pos = np.arange(len(config_sizes))
        
        tp2_perf = []
        tp8_perf = []
        
        for config_size in config_sizes:
            tp2_data = summary_df[(summary_df['config_size'] == config_size) & (summary_df['tp_type'] == 'tp2')]
            tp8_data = summary_df[(summary_df['config_size'] == config_size) & (summary_df['tp_type'] == 'tp8')]
            
            tp2_perf.append(tp2_data['price_performance_score'].mean() if not tp2_data.empty else 0)
            tp8_perf.append(tp8_data['price_performance_score'].mean() if not tp8_data.empty else 0)
        
        width = 0.35
        bars1 = ax10.bar(x_pos - width/2, tp2_perf, width, label='TP2', alpha=0.8, color='#1f77b4')
        bars2 = ax10.bar(x_pos + width/2, tp8_perf, width, label='TP8', alpha=0.8, color='#ff7f0e')
        
        ax10.set_xlabel('Configuration Size (Input_Output)')
        ax10.set_ylabel('Average Price-Performance Score')
        ax10.set_title('Overall Price-Performance Comparison by Configuration Size', fontweight='bold')
        ax10.set_xticks(x_pos)
        ax10.set_xticklabels(config_sizes, rotation=45, ha='right')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        # Add improvement percentages
        for i, (tp2_val, tp8_val) in enumerate(zip(tp2_perf, tp8_perf)):
            if tp2_val > 0:
                improvement = ((tp8_val - tp2_val) / tp2_val) * 100
                ax10.text(i, max(tp2_val, tp8_val) + max(tp2_perf + tp8_perf) * 0.02,
                         f'{improvement:+.1f}%', ha='center', va='bottom', fontweight='bold',
                         color='green' if improvement > 0 else 'red')
        
        # 5. Create latency vs throughput scatter plot
        ax11 = fig.add_subplot(gs[3, 2])
        
        tp2_data = df[df['tp_type'] == 'tp2']
        tp8_data = df[df['tp_type'] == 'tp8']
        
        if not tp2_data.empty:
            ax11.scatter(tp2_data['end_to_end_latency_mean'], tp2_data['output_throughput_mean'], 
                        alpha=0.7, s=60, label='TP2', color='#1f77b4')
        
        if not tp8_data.empty:
            ax11.scatter(tp8_data['end_to_end_latency_mean'], tp8_data['output_throughput_mean'], 
                        alpha=0.7, s=60, label='TP8', color='#ff7f0e')
        
        ax11.set_xlabel('End-to-End Latency (seconds)')
        ax11.set_ylabel('Output Throughput (tokens/sec)')
        ax11.set_title('Latency vs Throughput Trade-off', fontweight='bold')
        ax11.legend()
        ax11.grid(True, alpha=0.3)
        
        return fig
    
    def generate_summary_report(self, df: pd.DataFrame):
        """Generate a text summary of the comparison"""
        print("\n" + "="*80)
        print("LLM PERFORMANCE COMPARISON SUMMARY: TP2 vs TP8")
        print("="*80)
        
        # Overall statistics
        tp2_data = df[df['tp_type'] == 'tp2']
        tp8_data = df[df['tp_type'] == 'tp8']
        
        print(f"\nConfigurations analyzed: {len(df['config'].unique())}")
        print(f"Total test results: {len(df)} (TP2: {len(tp2_data)}, TP8: {len(tp8_data)})")
        
        print("\nCOST STRUCTURE:")
        print("-" * 60)
        print("TP2: 2 GPUs × $10/hour = $20/hour")
        print("TP8: 8 GPUs × $10/hour = $80/hour")
        print("TP8 costs 4x more per hour but delivers significant performance gains")
        
        # Key metrics comparison
        print("\nKEY PERFORMANCE METRICS (Average across all configurations):")
        print("-" * 60)
        
        metrics = [
            ('Time to First Token (TTFT)', 'ttft_mean', 'seconds'),
            ('End-to-End Latency', 'end_to_end_latency_mean', 'seconds'),
            ('Output Throughput', 'output_throughput_mean', 'tokens/sec'),
            ('Cost per Million Tokens', 'cost_per_million_tokens', 'USD'),
            ('Price-Performance Score', 'price_performance_score', 'tokens/sec/$'),
            ('Tokens per Dollar', 'tokens_per_dollar', 'tokens/$'),
            ('Error Rate', 'error_rate', '%')
        ]
        
        for metric_name, metric_col, unit in metrics:
            tp2_avg = tp2_data[metric_col].mean()
            tp8_avg = tp8_data[metric_col].mean()
            
            if tp2_avg > 0:
                improvement = ((tp8_avg - tp2_avg) / tp2_avg) * 100
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"
            
            print(f"{metric_name:25} | TP2: {tp2_avg:.3f} {unit:10} | TP8: {tp8_avg:.3f} {unit:10} | Change: {improvement_str}")
        
        # Best configurations
        print("\nBEST PERFORMING CONFIGURATIONS:")
        print("-" * 60)
        
        # Best latency
        best_latency_tp2 = tp2_data.loc[tp2_data['end_to_end_latency_mean'].idxmin()]
        best_latency_tp8 = tp8_data.loc[tp8_data['end_to_end_latency_mean'].idxmin()]
        
        print(f"Lowest Latency - TP2: {best_latency_tp2['config']} ({best_latency_tp2['end_to_end_latency_mean']:.3f}s)")
        print(f"Lowest Latency - TP8: {best_latency_tp8['config']} ({best_latency_tp8['end_to_end_latency_mean']:.3f}s)")
        
        # Best throughput
        best_throughput_tp2 = tp2_data.loc[tp2_data['output_throughput_mean'].idxmax()]
        best_throughput_tp8 = tp8_data.loc[tp8_data['output_throughput_mean'].idxmax()]
        
        print(f"Highest Throughput - TP2: {best_throughput_tp2['config']} ({best_throughput_tp2['output_throughput_mean']:.1f} tokens/sec)")
        print(f"Highest Throughput - TP8: {best_throughput_tp8['config']} ({best_throughput_tp8['output_throughput_mean']:.1f} tokens/sec)")
        
        # Best price-performance
        best_pp_tp2 = tp2_data.loc[tp2_data['price_performance_score'].idxmax()]
        best_pp_tp8 = tp8_data.loc[tp8_data['price_performance_score'].idxmax()]
        
        print(f"Best Price-Performance - TP2: {best_pp_tp2['config']} (score: {best_pp_tp2['price_performance_score']:.1f})")
        print(f"Best Price-Performance - TP8: {best_pp_tp8['config']} (score: {best_pp_tp8['price_performance_score']:.1f})")
    
    def run_comparison(self, output_file: str = 'tp2_vs_tp8_comparison.png'):
        """Run the complete comparison analysis"""
        print("Starting LLM Performance Comparison: TP2 vs TP8")
        print("="*50)
        
        # Load data
        self.load_data()
        
        if not self.data:
            print("ERROR: No comparable data found between TP2 and TP8 directories")
            return
        
        # Extract metrics
        df = self.extract_metrics()
        print(f"\nExtracted metrics for {len(df)} test configurations")
        
        # Generate summary report
        self.generate_summary_report(df)
        
        # Create visualization
        print(f"\nGenerating comprehensive visualization...")
        fig = self.create_comparison_plots(df)
        
        # Save the plot
        output_path = Path(output_file)
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Visualization saved to: {output_path.absolute()}")
        
        plt.close()
        
        return df

def main():
    """Main execution function"""
    comparator = LLMPerfComparator(
        tp2_dir='gpt-oss-h100-tp2',
        tp8_dir='gpt-oss-h100-tp8'
    )
    
    df = comparator.run_comparison('tp2_vs_tp8_comprehensive_comparison.png')
    
    if df is not None:
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("The comprehensive comparison visualization has been generated.")
        print("Check the PNG file for detailed visual analysis of:")
        print("• Latency metrics (TTFT, end-to-end, inter-token)")
        print("• Throughput performance (output, request, requests/min)")
        print("• Cost analysis (cost per million tokens, tokens per dollar)")
        print("• Price-performance metrics (throughput per dollar spent)")
        print("• Configuration-specific comparisons")
        print("• Trade-off analysis (latency vs throughput vs cost)")

if __name__ == "__main__":
    main()
