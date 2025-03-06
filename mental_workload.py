import matplotlib as plt
import numpy as np


def compute_rms_time_series3(videodata, map_voxel, region_data):
    """
    Sort the voxels to their cortical regions and then compute the RMS for each region along
    all 11250 time activations. Next, sort each region to its corresponding MRT based on the CSV file,
    compute the average RMS per MRT, and prepare data for visualization.

    Parameters:
        videodata (np.ndarray): Brain activation data with shape (voxels, time).
        map_voxel (np.ndarray): 1D array mapping each voxel to a cortical region (Cerebra_ID).
        region_data (pd.DataFrame): DataFrame with region mapping data. Required columns are:
            'Cerebra_ID', 'Region_name', 'Multiple resource theory ID', 'MRT ID Name'

    Returns:
        avg_rms_per_mrt (dict): Dictionary mapping each MRT ID to its average RMS time series.
        mwl_index (np.ndarray): Overall index (sum of all MRT time series) over time.
        mrt_regions_map (dict): Mapping from MRT ID to list of region names.
        mrt_name_map (dict): Mapping from MRT ID to MRT ID Name.
    """
    # Check required columns.
    required_columns = {"Cerebra_ID", "Region_name", "Multiple resource theory ID", "MRT ID Name"}
    missing_columns = required_columns - set(region_data.columns)
    if missing_columns:
        raise ValueError(f"❌ CSV file must contain the following columns: {missing_columns}")

    # Create mapping dictionaries.
    valid_cerebra_ids = set(region_data["Cerebra_ID"].unique())
    region_name_map = region_data.set_index("Cerebra_ID")["Region_name"].to_dict()
    mrt_id_map = region_data.set_index("Cerebra_ID")["Multiple resource theory ID"].to_dict()
    mrt_name_map = region_data.set_index("Multiple resource theory ID")["MRT ID Name"].to_dict()

    # Build mapping from MRT to its constituent region names.
    mrt_regions_map = {}
    for cerebra_id, mrt_id in mrt_id_map.items():
        region_name = region_name_map.get(cerebra_id, f"Region {cerebra_id}")
        mrt_regions_map.setdefault(mrt_id, []).append(region_name)

    # Ensure the voxel mapping length matches the number of voxels in videodata.
    if len(map_voxel) != videodata.shape[0]:
        print(f"Warning: map_voxel length ({len(map_voxel)}) does not match videodata voxel count ({videodata.shape[0]}). Trimming map_voxel.")
        map_voxel = map_voxel[:videodata.shape[0]]

    # Compute RMS for each cortical region.
    rms_per_region = {}
    for region in valid_cerebra_ids:
        indices = np.where(map_voxel == region)[0]
        if len(indices) == 0:
            continue
        # Compute RMS across voxels at each time point.
        rms_per_region[region] = np.sqrt(np.mean(videodata[indices, :]**2, axis=0))

    # Group the regional RMS time series by MRT.
    rms_per_mrt = {}
    for region, rms_series in rms_per_region.items():
        mrt_id = mrt_id_map.get(region)
        if mrt_id is None:
            continue
        rms_per_mrt.setdefault(mrt_id, []).append(rms_series)

    # Compute the average RMS time series for each MRT.
    avg_rms_per_mrt = {mrt_id: np.mean(rms_list, axis=0) for mrt_id, rms_list in rms_per_mrt.items()}

    # Compute overall Mental Workload Index (MWL Index) as the sum over MRT time series.
    mwl_index = np.sum(list(avg_rms_per_mrt.values()), axis=0)

    return avg_rms_per_mrt, mwl_index, mrt_regions_map, mrt_name_map

def plot_rms_time_series3(avg_rms_per_mrt, mwl_index, mrt_regions_map, mrt_name_map):
    """
    Plot the average RMS time series for each MRT separately in red and overlay the overall
    average (Mental Workload Index) in green.
    """
    # Plot each MRT's average RMS time series in red.
    for mrt_id, rms_values in avg_rms_per_mrt.items():
        mrt_name = mrt_name_map.get(mrt_id, f"Unknown MRT {mrt_id}")
        regions_str = ", ".join(mrt_regions_map.get(mrt_id, []))
        plt.figure(figsize=(16, 5))
        plt.plot(range(len(rms_values)), rms_values, color='red', alpha=0.7,
                 label=f"{mrt_name} (MRT ID {mrt_id})")
        plt.xlabel("Time Activation (0 - 11250)", fontsize=12)
        plt.ylabel("Average RMS Value", fontsize=12)
        plt.title(f"Average RMS for {mrt_name} (MRT ID {mrt_id})\nRegions: {regions_str}", fontsize=14)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    # Plot overall MWL Index in green.
    plt.figure(figsize=(16, 5))
    plt.plot(range(len(mwl_index)), mwl_index, color='green', linewidth=2,
             label="Mental Workload Index")
    plt.xlabel("Time ", fontsize=12)
    plt.ylabel("Sum of Average RMS of Cortical Regions", fontsize=12)
    plt.title("Mental Workload Index", fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()