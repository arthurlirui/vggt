import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Set scientific plotting style
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 10
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10

# Sample data - replace with your actual results
num_views = np.array([2, 4, 6, 8, 10, 12, 14, 16])

# PSNR data (dB)
psnr_data = {
    'Proposed Method': np.array([18.5, 22.3, 25.1, 27.8, 29.5, 31.2, 32.1, 32.8]),
    'NeRF': np.array([16.2, 19.8, 22.5, 24.3, 26.1, 27.4, 28.2, 28.9]),
    'Gaussian Splatting': np.array([17.8, 21.5, 23.9, 26.2, 27.8, 29.1, 29.9, 30.5]),
    'VGGT': np.array([15.9, 19.2, 21.8, 23.7, 25.3, 26.6, 27.3, 27.9]),
    'Random Selection': np.array([14.5, 17.8, 20.1, 22.3, 23.8, 25.1, 25.9, 26.5])
}

# SSIM data
ssim_data = {
    'Proposed Method': np.array([0.65, 0.78, 0.85, 0.89, 0.92, 0.94, 0.95, 0.96]),
    'NeRF': np.array([0.58, 0.71, 0.79, 0.83, 0.86, 0.89, 0.90, 0.91]),
    'Gaussian Splatting': np.array([0.62, 0.75, 0.82, 0.86, 0.89, 0.91, 0.92, 0.93]),
    'VGGT': np.array([0.56, 0.69, 0.77, 0.81, 0.84, 0.87, 0.88, 0.89]),
    'Random Selection': np.array([0.52, 0.64, 0.72, 0.77, 0.81, 0.84, 0.85, 0.86])
}

# 3D Overlapping data
overlap_data = {
    'Proposed Method': np.array([0.72, 0.83, 0.89, 0.92, 0.94, 0.95, 0.96, 0.97]),
    'NeRF': np.array([0.65, 0.76, 0.82, 0.86, 0.89, 0.91, 0.92, 0.93]),
    'Gaussian Splatting': np.array([0.68, 0.79, 0.85, 0.88, 0.91, 0.93, 0.94, 0.95]),
    'VGGT': np.array([0.63, 0.74, 0.80, 0.84, 0.87, 0.89, 0.90, 0.91]),
    'Random Selection': np.array([0.58, 0.69, 0.76, 0.80, 0.83, 0.86, 0.87, 0.88])
}

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Define colors and markers for different methods
colors = {
    'Proposed Method': '#E41A1C',  # Red
    'NeRF': '#377EB8',  # Blue
    'Gaussian Splatting': '#4DAF4A',  # Green
    'VGGT': '#984EA3',  # Purple
    'Random Selection': '#FF7F00'  # Orange
}

markers = {
    'Proposed Method': 'o',
    'NeRF': 's',
    'Gaussian Splatting': '^',
    'VGGT': 'D',
    'Random Selection': 'v'
}

# Plot PSNR
ax1 = axes[0]
for method in psnr_data:
    ax1.plot(num_views, psnr_data[method],
             label=method, color=colors[method], marker=markers[method],
             markersize=6, linewidth=2, markeredgecolor='white', markeredgewidth=0.5)

ax1.set_xlabel('Number of Views')
ax1.set_ylabel('PSNR (dB)')
ax1.set_title('(a) Reconstruction Quality (PSNR)')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(num_views)

# Plot SSIM
ax2 = axes[1]
for method in ssim_data:
    ax2.plot(num_views, ssim_data[method],
             label=method, color=colors[method], marker=markers[method],
             markersize=6, linewidth=2, markeredgecolor='white', markeredgewidth=0.5)

ax2.set_xlabel('Number of Views')
ax2.set_ylabel('SSIM')
ax2.set_title('(b) Structural Similarity (SSIM)')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(num_views)

# Plot 3D Overlapping
ax3 = axes[2]
for method in overlap_data:
    ax3.plot(num_views, overlap_data[method],
             label=method, color=colors[method], marker=markers[method],
             markersize=6, linewidth=2, markeredgecolor='white', markeredgewidth=0.5)

ax3.set_xlabel('Number of Views')
ax3.set_ylabel('3D Overlapping Score')
ax3.set_title('(c) 3D Geometry Accuracy')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(num_views)

# Create a single legend for all subplots
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',
           bbox_to_anchor=(0.5, 0.05), ncol=5, frameon=True,
           fancybox=True, shadow=True)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Make room for the legend

# Save figure with high quality
plt.savefig('view_selection_performance.pdf', dpi=300, bbox_inches='tight')
plt.savefig('view_selection_performance.png', dpi=300, bbox_inches='tight')

plt.show()

# Additional analysis: Performance improvement table
print("Performance Improvement Analysis (Proposed vs Best Baseline):")
print("=" * 60)

# Calculate improvements at different view counts
view_indices = [1, 3, 5, 7]  # Corresponding to 4, 8, 12, 16 views
view_counts = num_views[view_indices]

print(f"{'Metric':<15} {'Views':<8} {'Proposed':<10} {'Best Baseline':<15} {'Improvement':<12}")
print("-" * 60)

for idx, view_count in zip(view_indices, view_counts):
    # PSNR improvement
    baseline_psnr = max(psnr_data['NeRF'][idx], psnr_data['Gaussian Splatting'][idx],
                        psnr_data['VGGT'][idx], psnr_data['Random Selection'][idx])
    psnr_improvement = psnr_data['Proposed Method'][idx] - baseline_psnr

    # SSIM improvement
    baseline_ssim = max(ssim_data['NeRF'][idx], ssim_data['Gaussian Splatting'][idx],
                        ssim_data['VGGT'][idx], ssim_data['Random Selection'][idx])
    ssim_improvement = ssim_data['Proposed Method'][idx] - baseline_ssim

    # Overlap improvement
    baseline_overlap = max(overlap_data['NeRF'][idx], overlap_data['Gaussian Splatting'][idx],
                           overlap_data['VGGT'][idx], overlap_data['Random Selection'][idx])
    overlap_improvement = overlap_data['Proposed Method'][idx] - baseline_overlap

    print(
        f"{'PSNR':<15} {view_count:<8} {psnr_data['Proposed Method'][idx]:<10.2f} {baseline_psnr:<15.2f} {psnr_improvement:<12.2f}")
    print(
        f"{'SSIM':<15} {view_count:<8} {ssim_data['Proposed Method'][idx]:<10.3f} {baseline_ssim:<15.3f} {ssim_improvement:<12.3f}")
    print(
        f"{'3D Overlap':<15} {view_count:<8} {overlap_data['Proposed Method'][idx]:<10.3f} {baseline_overlap:<15.3f} {overlap_improvement:<12.3f}")
    print("-" * 60)