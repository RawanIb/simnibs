import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from simnibs import ElementTags
from simnibs.mesh_tools import mesh_io
from simnibs.utils.simnibs_logger import logger
from neck_configs_simulation import results_path, elec1_center, elec2_center

ROI = 'ROI'
NON_ROI = 'Non-ROI'
SCALP = 'Scalp'


def load_simulation_result(simulation_path):
    """
    Load the simulation result mesh file.

    Args:
        simulation_path (Path): Path to the simulation results directory

    Returns:
        mesh: SimNIBS mesh object containing simulation results
    """
    # Find the mesh file containing results
    results_files = list(simulation_path.glob('*.msh'))
    if not results_files:
        raise FileNotFoundError(f"No .msh files found in {simulation_path}")
    return mesh_io.read_msh(results_files[0])


def get_skin_roi_around_electrodes(mesh, center_point_list, radius_mm=50):
    """
    Get ROI in the scalp around the electrode center.

    Args:
        mesh: SimNIBS mesh object
        center_point_list (list): List of [x, y, z] coordinates of electrode center
        radius_mm (float): Radius in mm for ROI (default 50mm = 5cm)

    Returns:
        numpy array: Boolean mask of elements within ROI
    """
    # Crop the mesh so we only have scalp
    scalp_mesh = mesh.crop_mesh(ElementTags.SCALP)
    scalp_centroids = scalp_mesh.elements_baricenters()[:]
    elm_volumes = scalp_mesh.elements_volumes_and_areas()[:]

    def get_roi_mask(center_point):
        distances = np.linalg.norm(scalp_centroids - np.array(center_point), axis=1)
        return distances <= radius_mm

    roi_mask = np.zeros(len(scalp_centroids), dtype=bool)
    roi_volumes = {}

    for i, center_point in enumerate(center_point_list):
        electrode_mask = get_roi_mask(center_point)
        roi_mask = electrode_mask | roi_mask
        roi_volumes[f'Electrode_{i + 1}'] = np.sum(elm_volumes[electrode_mask])

    roi_volumes['Total_ROI'] = np.sum(elm_volumes[roi_mask])
    roi_volumes['Total_Scalp'] = np.sum(elm_volumes)
    roi_volumes['ROI_Fraction'] = roi_volumes['Total_ROI'] / roi_volumes['Total_Scalp'] * 100

    scalp_mesh.add_element_field(roi_mask, 'roi')
    return roi_mask, scalp_mesh, roi_volumes


def plot_field_distribution(field_values_dict, config_name, output_dir):
    """
    Create visualization plots for field distribution using basic matplotlib.

    Args:
        field_values_dict: Dictionary containing field values for ROI and non-ROI
        config_name: Name of the current configuration
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Set default font sizes
    plt.rcParams.update({'font.size': 12})

    # Calculate optimal number of bins based on data
    max_field = max(np.max(field_values_dict['ROI']), np.max(field_values_dict['Non-ROI']))
    min_field = min(np.min(field_values_dict['ROI']), np.min(field_values_dict['Non-ROI']))

    # Create bins with smaller intervals for better resolution
    bins = np.linspace(min_field, max_field, 100)

    # 1. Histogram Plot
    plt.figure(figsize=(12, 6))
    for region, values in field_values_dict.items():
        if region != 'Scalp':  # Skip total scalp for clarity
            plt.hist(values, bins=bins, alpha=0.7, label=region, density=True)

    plt.yscale('log')  # Use log scale for y-axis
    plt.title(f'Electric Field Distribution - {config_name}')
    plt.xlabel('Electric Field Magnitude (V/m)')
    plt.ylabel('Density')
    plt.legend(loc='upper right',
               framealpha=0.9,
               edgecolor='gray')
    plt.grid(True, alpha=0.4, which='both', linestyle=':')
    # Calculate and format statistics
    roi_stats = {
        'mean': np.mean(field_values_dict['ROI']),
        'std': np.std(field_values_dict['ROI']),
        'median': np.median(field_values_dict['ROI']),
        'p95': np.percentile(field_values_dict['ROI'], 95)
    }

    non_roi_stats = {
        'mean': np.mean(field_values_dict['Non-ROI']),
        'std': np.std(field_values_dict['Non-ROI']),
        'median': np.median(field_values_dict['Non-ROI']),
        'p95': np.percentile(field_values_dict['Non-ROI'], 95)
    }

    # Create detailed stats text
    stats_text = (
        f'ROI: mean={roi_stats["mean"]:.2f}±{roi_stats["std"]:.2f} V/m\n'
        f'     median={roi_stats["median"]:.2f}, 95th={roi_stats["p95"]:.2f} V/m\n'
        f'\n Non-ROI: mean={non_roi_stats["mean"]:.2f}±{non_roi_stats["std"]:.2f} V/m\n'
        f'         median={non_roi_stats["median"]:.2f}, 95th={non_roi_stats["p95"]:.2f} V/m'
    )

    # Add stats text box
    plt.text(0.85, 0.97, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round',
                       facecolor='white',
                       alpha=0.9,
                       edgecolor='gray'),
             fontsize=12)

    # Labels and title
    plt.xlabel('Electric Field Magnitude (V/m)')
    plt.ylabel('Log Density')

    # Format title to be more readable
    config_name_formatted = config_name.replace('simu_neck_config_', '')
    plt.title(f'Electric Field Distribution - {config_name_formatted}')

    # Add vertical lines for means
    plt.axvline(roi_stats['mean'], color='skyblue', linestyle='--', alpha=0.8,
                label='ROI mean')
    plt.axvline(non_roi_stats['mean'], color='orange', linestyle='--', alpha=0.8,
                label='Non-ROI mean')

    # Tight layout to prevent text cutoff
    plt.tight_layout()

    plt.savefig(output_dir / f'{config_name}_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Box Plot
    plt.figure(figsize=(8, 6))
    data = []
    labels = []
    for region, values in field_values_dict.items():
        if region != 'Scalp':  # Skip total scalp for clarity
            data.append(values)
            labels.extend([region] * len(values))

    plt.boxplot([field_values_dict['ROI'], field_values_dict['Non-ROI']],
                labels=['ROI', 'Non-ROI'])
    plt.title(f'Electric Field Distribution by Region - {config_name}')
    plt.ylabel('Electric Field Magnitude (V/m)')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f'{config_name}_boxplot.png')
    plt.close()


def plot_summary_heatmap(focality_df, output_dir):
    """
    Create a heatmap summary of focality metrics across configurations.
    """
    plt.figure(figsize=(12, 6))

    # Create heatmap data
    data = focality_df.values

    # Plot heatmap
    plt.imshow(data, aspect='auto', cmap='viridis')

    # Add text annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f'{data[i, j]:.2f}',
                     ha='center', va='center')

    # Configure axes
    plt.xticks(range(len(focality_df.columns)),
               focality_df.columns, rotation=45)
    plt.yticks(range(len(focality_df.index)),
               focality_df.index)

    plt.title('Focality Metrics Comparison Across Configurations')
    plt.colorbar(label='Value')
    plt.tight_layout()
    plt.savefig(output_dir / 'focality_comparison.png')
    plt.close()


def analyze_field_in_roi(scalp_mesh, roi_mask, field_name='magnE'):
    """
    Analyze the electric field in the ROI with visualization.
    """
    scalp_field_values = scalp_mesh.field[field_name][:]
    elm_volumes = scalp_mesh.elements_volumes_and_areas()[:]

    values = {
        SCALP: scalp_field_values,
        ROI: scalp_field_values[roi_mask],
        NON_ROI: scalp_field_values[~roi_mask]
    }
    volumes = {
        SCALP: elm_volumes,
        ROI: elm_volumes[roi_mask],
        NON_ROI: elm_volumes[~roi_mask]
    }

    results = {}
    for name, field_values in values.items():
        vol = volumes[name]
        results[name] = {
            'mean': np.average(field_values, weights=vol),
            'std': np.sqrt(np.average((field_values - np.average(field_values, weights=vol)) ** 2, weights=vol)),
            'median': np.median(field_values),
            'p80': np.percentile(field_values, 80),
            'p90': np.percentile(field_values, 90),
            'p95': np.percentile(field_values, 95),
            'p99': np.percentile(field_values, 99),
            'max': np.max(field_values),
            'min': np.min(field_values)
        }

    focality_metrics = {
        'ROI_to_NonROI_mean': results[ROI]['mean'] / results[NON_ROI]['mean'],
        'ROI_to_Total_mean': results[ROI]['mean'] / results[SCALP]['mean'],
        'ROI_to_NonROI_max': results[ROI]['max'] / results[NON_ROI]['max'],
        'ROI_above_total_p80': np.mean(values[ROI] > results[SCALP]['p80']) * 100,
        'ROI_above_total_p90': np.mean(values[ROI] > results[SCALP]['p90']) * 100
    }

    return pd.DataFrame(results), focality_metrics, values


def plot_roi_comparison(all_results, output_dir):
    """
    Create refined grouped bar plot comparing ROI mean±std and max values across configurations.
    """
    # Extract ROI means, stds, and max values
    configs = []
    means = []
    stds = []
    maxs = []

    for result in all_results:
        # Format configuration names
        config_name = result['Configuration'].iloc[0].replace('simu_neck_config_ellipse_', '')
        # Split into dimensions and conductivity
        dims, cond = config_name.split('mm_')
        config_name = f"{dims} mm\n{cond.replace('Sm-1', ' S/m')}"
        configs.append(config_name)
        means.append(result['ROI']['mean'])
        stds.append(result['ROI']['std'])
        maxs.append(result['ROI']['max'])

    # Create figure
    plt.figure(figsize=(12, 8))

    # Set width of bars and positions
    width = 0.35
    x = np.arange(len(configs))

    # Create bars with refined colors
    bars1 = plt.bar(x - width / 2, means, width, yerr=stds, capsize=5,
                    label='Mean ± Std', color='lightblue', alpha=0.8,
                    error_kw=dict(ecolor='black', capthick=1, capsize=5))
    bars2 = plt.bar(x + width / 2, maxs, width,
                    label='Max', color='salmon', alpha=0.8)

    # Customize plot
    plt.xlabel('Configuration', fontsize=12, labelpad=10)
    plt.ylabel('Electric Field Magnitude (V/m)', fontsize=12, labelpad=10)
    plt.title('ROI Electric Field Statistics Across Configurations', fontsize=14, pad=20)

    # Set x-axis labels
    plt.xticks(x, configs, fontsize=10)

    # Add value labels on top of bars
    for bar1, bar2, mean, std, max_val in zip(bars1, bars2, means, stds, maxs):
        # Mean±std label
        plt.text(bar1.get_x() + bar1.get_width() / 2., mean,
                 f'{mean:.2f}\n±{std:.2f}',
                 ha='center', va='bottom', fontsize=10)

        # Max label
        plt.text(bar2.get_x() + bar2.get_width() / 2., max_val,
                 f'{max_val:.2f}',
                 ha='center', va='bottom', fontsize=10)

    # Add legend with refined position and style
    plt.legend(loc='upper right', framealpha=0.9, fontsize=10)

    # Add grid for easier comparison
    plt.grid(True, axis='y', alpha=0.3, linestyle='--', color='gray')

    # Add some padding to y-axis
    ymax = max(maxs) * 1.15
    plt.ylim(0, ymax)

    # Adjust layout
    plt.tight_layout()

    # Save plot with high resolution
    plt.savefig(output_dir / 'roi_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    open_in_gmsh = False
    field_name = 'magnE'

    # Create results directory for plots
    plots_dir = Path('analysis_plots')
    plots_dir.mkdir(exist_ok=True)

    # Store results for all configurations
    all_results = []
    all_focality = []

    # Process each simulation result
    for config_dir in results_path.iterdir():
        if config_dir.is_dir():
            logger.info(f"Processing {config_dir.name}")

            # Load mesh
            mesh = load_simulation_result(config_dir)

            # Get ROI mask
            roi_mask, scalp_mesh, roi_volumes = get_skin_roi_around_electrodes(
                mesh=mesh,
                center_point_list=[elec1_center, elec2_center],
                radius_mm=25
            )

            if open_in_gmsh:
                scalp_mesh.open_in_gmsh()

            # Analyze electric field in ROI
            results_df, focality_metrics, field_values = analyze_field_in_roi(scalp_mesh=scalp_mesh,
                                                                              roi_mask=roi_mask,
                                                                              field_name=field_name)

            # Store results
            results_df['Configuration'] = config_dir.name
            focality_df = pd.DataFrame([focality_metrics], index=[config_dir.name])

            all_results.append(results_df)
            all_focality.append(focality_df)

            # Create plots for this configuration
            plot_field_distribution(field_values, config_dir.name, plots_dir)

            # Log results
            logger.info(f"Results for {config_dir.name}:")
            logger.info("\nField Statistics:\n" + results_df.to_markdown())
            logger.info("\nFocality Metrics:\n" + focality_df.to_markdown())
            logger.info("\nROI Volumes:")
            for key, value in roi_volumes.items():
                logger.info(f"{key}: {value:.2f} mm³")

    # Combine all results
    if all_results:
        # Create summary plots across all configurations
        combined_results = pd.concat(all_results)
        combined_focality = pd.concat(all_focality)

        # Save combined results
        combined_results.to_csv(plots_dir / 'combined_results.csv')
        combined_focality.to_csv(plots_dir / 'focality_results.csv')

        # Add this line to create the ROI comparison plot
        plot_roi_comparison(all_results, plots_dir)

        # Create summary heatmap
        plot_summary_heatmap(combined_focality, plots_dir)


if __name__ == "__main__":
    main()
