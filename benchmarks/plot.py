import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Configuration
INPUT_FILENAME = "benchmark_results.csv"

def generate_plots():
    if not os.path.exists(INPUT_FILENAME):
        print(f"Error: {INPUT_FILENAME} not found. Run benchmark_v2.py first.")
        sys.exit(1)

    try:
        df = pd.read_csv(INPUT_FILENAME)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Use a clean, scientific style
    plt.style.use('default') 
    
    # ==========================================
    # PLOT 1: Memory Usage (memplot.pdf)
    # ==========================================
    plt.figure(figsize=(6, 4)) # Standard single-column figure size
    
    plt.plot(df['nodes'], df['avg_mem'], marker='o', linestyle='-', 
             color='#2c3e50', linewidth=2, label='Avg Memory')

    # plt.title('RAM Usage vs. Network Size', fontsize=12, fontweight='bold')
    plt.xlabel('Nodes / Qubits', fontsize=10)
    plt.ylabel('Memory Usage (MB)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Auto-scale Y axis
    min_mem = df['avg_mem'].min()
    max_mem = df['avg_mem'].max()
    padding = (max_mem - min_mem) * 0.5 if max_mem != min_mem else 5
    plt.ylim(min_mem - padding, max_mem + padding)

    plt.tight_layout()
    plt.savefig('memplot.pdf', format='pdf', bbox_inches='tight')
    print("Generated memplot.pdf")
    plt.close()

    # ==========================================
    # PLOT 2: Generation Time (gentime.pdf)
    # ==========================================
    plt.figure(figsize=(6, 4))
    
    plt.errorbar(df['nodes'], df['avg_time'], yerr=df['std_dev'], 
                 fmt='-o', color='#e74c3c', ecolor='gray', 
                 elinewidth=1.5, capsize=4, linewidth=2)

    # plt.title('Circuit Synthesis Latency', fontsize=12, fontweight='bold')
    plt.xlabel('Nodes / Qubits', fontsize=10)
    plt.ylabel('Time (Seconds)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig('gentime.pdf', format='pdf', bbox_inches='tight')
    print("Generated gentime.pdf")
    plt.close()

if __name__ == "__main__":
    generate_plots()