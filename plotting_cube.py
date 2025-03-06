import matplotlib as plt
import numpy as np

def compute_rms_time_series2(videodata, map_voxel, region_data):
    """
    Sorts 31,553 points into regions based on Cerebra ID,
    Compute the RMS for a given video dataset across all time locations.
    calculating RMS per time activation for each region,
    and visualizing them separately over time.
    """
    required_columns = {"Cerebra_ID", "Region_name"}
    missing_columns = required_columns - set(region_data.columns)

    if missing_columns:
        raise ValueError(f"❌ CSV file must contain the following columns: {missing_columns}")

    valid_cerebra_ids = set(region_data["Cerebra_ID"].unique())
    region_name_map = region_data.set_index("Cerebra_ID")["Region_name"].to_dict()

    avg_rms_time_series = {region: np.zeros(videodata.shape[1]) for region in valid_cerebra_ids}
    count_voxels = {region: np.zeros(videodata.shape[1]) for region in valid_cerebra_ids}

    for time_location in range(videodata.shape[1]):
        videodata_selected = videodata[:, time_location]
        voxel_rms = np.sqrt(videodata_selected**2)

        for region in valid_cerebra_ids:
            region_indices = np.where(map_voxel == region)[0]
            if len(region_indices) == 0:
                continue

            rms_values = voxel_rms[region_indices]
            avg_rms_time_series[region][time_location] += np.sum(rms_values)
            count_voxels[region][time_location] += len(rms_values)

    for region in valid_cerebra_ids:
        valid_indices = count_voxels[region] > 0
        avg_rms_time_series[region][valid_indices] /= count_voxels[region][valid_indices]

    return avg_rms_time_series, region_name_map

def plot_rms_time_series2(avg_rms_time_series, region_name_map):
    """
    Visualize each region's RMS separately in 102 diagrams with both Region Name and Cerebra ID.
    """
    for region, rms_values in avg_rms_time_series.items():
        region_name = region_name_map.get(region, f"Region {region}")
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(rms_values)), rms_values, label=f"{region_name} (ID: {region})")
        plt.xlabel("Time Activation (0 - 11250)")
        plt.ylabel("RMS Value")
        plt.title(f"{region_name} (ID: {region})")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.show()