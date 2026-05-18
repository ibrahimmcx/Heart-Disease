import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import load_data, analyze_and_clean_outliers

# Configure high-quality visualization style in pure Matplotlib
plt.rcParams.update({
    'font.size': 11,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Custom clinical palette
PALETTE = {0: "#2ec4b6", 1: "#e71d36"} # Teal for healthy, red-orange for cardiovascular disease

def generate_eda_reports(df, output_dir="reports/figures"):
    """Generates and saves clinical visual analysis plots using pure Matplotlib."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[EDA] Initiating clinical visual exploration. Storing charts in: {output_dir}")
    
    # 1. Class Balance Visualization
    plt.figure(figsize=(6, 5))
    counts = df['target'].value_counts()
    classes = [0, 1]
    class_labels = ['Healthy', 'Heart Disease']
    
    # Bar plot
    bars = plt.bar(classes, [counts.get(c, 0) for c in classes], color=[PALETTE[c] for c in classes], width=0.6)
    
    plt.title("Cardiovascular Disease Class Distribution", pad=15, weight='bold')
    plt.xlabel("Diagnostic Status")
    plt.ylabel("Patient Count")
    plt.xticks(classes, class_labels)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height - 35, f'{int(height)}',
                 ha='center', va='bottom', color='white', fontweight='bold', fontsize=12)
        
    plt.tight_layout()
    class_balance_path = os.path.join(output_dir, "class_balance.png")
    plt.savefig(class_balance_path)
    plt.close()
    print(f"[EDA] Successfully generated {class_balance_path}")

    # 2. Correlation Matrix Heatmap
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    corr_matrix = corr.values
    cols = corr.columns.tolist()
    
    # Create mask for upper triangle (replace values with NaN)
    n_features = len(cols)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            corr_matrix[i, j] = np.nan
            
    # Draw heatmap using imshow
    im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    plt.colorbar(im, shrink=0.7)
    
    # Configure axes
    plt.xticks(np.arange(n_features), [c.upper() for c in cols], rotation=45, ha='right', fontsize=9)
    plt.yticks(np.arange(n_features), [c.upper() for c in cols], fontsize=9)
    
    # Annotate correlation values inside cells
    for i in range(n_features):
        for j in range(i + 1):  # Only lower triangle
            val = corr_matrix[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}",
                         ha="center", va="center", color="black" if abs(val) < 0.4 else "white",
                         fontsize=8, weight="bold" if abs(val) > 0.3 else "normal")
                
    plt.title("Clinical Feature Correlation Matrix (Cleveland Cohort)", pad=20, weight='bold', fontsize=14)
    plt.grid(False) # Remove grid lines inside matrix
    plt.tight_layout()
    corr_path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(corr_path)
    plt.close()
    print(f"[EDA] Successfully generated {corr_path}")

    # 3. Clinical Parameter Boxplots (Grouped by target Outcome)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    features_to_plot = [
        ('thalach', "Maximum Heart Rate Achieved (thalach)", "Beats Per Minute (BPM)"),
        ('oldpeak', "ST Depression Induced by Exercise (oldpeak)", "Depression in mm"),
        ('age', "Patient Age Distribution", "Age (Years)"),
        ('chol', "Serum Cholesterol Levels (chol)", "Cholesterol (mg/dl)")
    ]
    
    for idx, (feat, title, ylabel) in enumerate(features_to_plot):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        # Prepare data groups
        group_healthy = df[df['target'] == 0][feat].values
        group_disease = df[df['target'] == 1][feat].values
        
        # Draw boxplots
        bp = ax.boxplot([group_healthy, group_disease], patch_artist=True, widths=0.5)
        
        # Apply custom colors to boxes
        for patch, color in zip(bp['boxes'], [PALETTE[0], PALETTE[1]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('#222222')
            
        # Customize median line style
        for median in bp['medians']:
            median.set_color('#ffffff')
            median.set_linewidth(2.0)
            
        ax.set_title(title, weight='bold', fontsize=11, pad=10)
        ax.set_xticklabels(['Healthy', 'Heart Disease'])
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.5)
        
    plt.suptitle("Continuous Clinical Parameter Comparison by Cardiac Outcome", y=0.98, weight='bold', fontsize=14)
    plt.tight_layout()
    clinical_path = os.path.join(output_dir, "clinical_distributions.png")
    plt.savefig(clinical_path)
    plt.close()
    print(f"[EDA] Successfully generated {clinical_path}")
    print("[EDA] Clinical visual analysis complete!")

if __name__ == "__main__":
    df = load_data()
    df_cleaned = analyze_and_clean_outliers(df)
    generate_eda_reports(df_cleaned)
