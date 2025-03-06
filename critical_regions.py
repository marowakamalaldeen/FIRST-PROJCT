
import matplotlib as plt
import numpy as np

def compute_rms_time_series1(videodata, map_voxel, region_data):
    """
    Compute the RMS for a given video dataset across all time locations,
    sorting 31,553 points into regions based on Cerebra ID,
    calculating RMS per time activation for each region,
    and visualizing them separately over time.
    """
    if "Cerebra_ID" not in region_data.columns or "Region_name" not in region_data.columns:
        raise ValueError("CSV file must contain 'Cerebra_ID' and 'Region_name' columns.")

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

def plot_rms_time_series1(avg_rms_time_series, region_name_map, subject_id, video_type):
    """
    Visualize each region's RMS separately in 102 diagrams with both Region Name and Cerebra ID,
    and include Subject ID and Video Type in titles.
    """
    for region, rms_values in avg_rms_time_series.items():
        region_name = region_name_map.get(region, f"Region {region}")
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(rms_values)), rms_values, label=f"{region_name} (ID: {region})")
        plt.xlabel("Time Activation (0 - 11250)")
        plt.ylabel("RMS Value")
        plt.title(f"{region_name} (ID: {region})\nSubject: {subject_id} | Video Type: {video_type}")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.show()

def get_number_of_voxel_per_region(cerebra_id, map_voxel, region_data):
    """
    Get the number of voxels for a given cerebral region (Cerebra ID) and its region name.

    Parameters:
        cerebra_id (int): The ID of the cerebral region.

    Returns:
        tuple: (voxel_count (int), region_name (str))
    """
    voxel_count = np.sum(map_voxel == cerebra_id)  # Count occurrences of the Cerebra ID
    mapping_region_id_to_name = region_data.set_index("Cerebra_ID")["Region_name"].to_dict()
    region_name = mapping_region_id_to_name.get(cerebra_id, "Unknown Region")
    return voxel_count, region_name


def plot_voxel_activation_for_region(time_location, cerebra_id, video_data, map_voxel):
    """
    Plot voxel activation histograms for the selected region and time location,
    for each video type in video_data.

    Parameters:
        time_location (int): The time index to analyze.
        cerebra_id (int): The Cerebra ID for the region of interest.
        video_data (dict): Dictionary of video datasets.
    """
    # Get indices for voxels in the desired region.
    region_indices = np.where(map_voxel == cerebra_id)[0]

    plt.figure(figsize=(12, 5))
    for video_label, data in video_data.items():
        # Restrict activation data to the region of interest and given time
        voxel_activation = data[region_indices, time_location]
        plt.hist(voxel_activation, bins=100, alpha=0.5, label=f"{video_label} - Time {time_location}")

    plt.title(f"Voxel Activation Distribution for Region {cerebra_id} at Time {time_location}")
    plt.xlabel("Voxel Activation Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def plot_number_of_voxel_per_region(map_voxel):
    """
    Plot a histogram showing the overall voxel counts per cortical region.
    """
    unique_regions = np.unique(map_voxel)
    plt.figure(figsize=(20, 5))
    plt.hist(map_voxel, bins=len(unique_regions), color="royalblue", alpha=0.7)
    plt.title("Number of Voxels per Cortical Region")
    plt.xlabel("Cortical Regions")
    plt.ylabel("Number of Voxels")
    plt.show()