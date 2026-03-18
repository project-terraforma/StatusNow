import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def create_presentation_chart():
    # Data
    datasets = ['NYC + SF', 'NYC + SF\n+ Season 2', '5 Cities', '12 Cities']
    samples = [12, 18.6, 53, 123]  # Thousands
    accuracies = [80.5, 81.7, 88.5, 89.18]

    # Style configuration
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(12, 7), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#ffffff')

    # Colors
    color_acc = '#4051CC' # Requested color (Blue/Purple)
    color_bars = '#dbeafe' # Light Blue

    # 1. Plot the bar chart for Samples
    bars = ax1.bar(datasets, samples, color=color_bars, width=0.6, label='Dataset Size (k)')
    
    # Customize left axis
    ax1.set_ylabel('Training Samples (Thousands)', color='#475569', fontweight='bold', fontsize=12, labelpad=15)
    ax1.tick_params(axis='y', labelcolor='#475569')
    ax1.set_ylim(0, 150)
    
    # Add values inside bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{height}k',
                ha='center', va='center', fontweight='bold', color='#1e3a8a', fontsize=11)

    # 2. Plot the line chart for Accuracy on secondary axis
    ax2 = ax1.twinx()
    line_acc, = ax2.plot(datasets, accuracies, color=color_acc, marker='o', 
                         linewidth=4, markersize=12, label='Balanced Accuracy (%)')
    
    # Customize right axis
    ax2.set_ylabel('Balanced Accuracy (%)', color=color_acc, fontweight='bold', fontsize=12, labelpad=15)
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.set_ylim(75, 95)
    
    # Add accuracy values above points
    for i, acc in enumerate(accuracies):
        ax2.annotate(f'{acc}%', 
                    (i, acc), 
                    textcoords="offset points", 
                    xytext=(0, 15), 
                    ha='center',
                    fontweight='bold',
                    fontsize=14,
                    color=color_acc,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_acc, alpha=0.9))

    # Clean up aesthetics
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.xticks(fontsize=12, fontweight='bold', color='#1e293b')
    
    # Title
    plt.title('Impact of Dataset Size on Model Accuracy',
              fontsize=18, fontweight='bold', color='#1e293b', pad=30)

    # Save
    out_path = 'presentation_dataset_impact.png'
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', transparent=False)
    print(f"✅ Successfully generated beautiful presentation asset: {out_path}")

if __name__ == "__main__":
    create_presentation_chart()
