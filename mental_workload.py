import matplotlib as plt
import numpy as np

def compute_rms_time_series3(videodata, map_voxel, region_data):
    """
    Compute the RMS for each region along all 11250 time activations,
    sort each region to its corresponding MRT based on the CSV file,
    compute the average RMS per MRT, and prepare data for visualization.
    """
    required_columns = {"Cerebra_ID", "Region_name", "Multiple resource theory ID", "MRT ID Name"}
    missing_columns = required_columns - set(region_data.columns)
    if missing_columns:
        raise ValueError(f"CSV file must contain the following columns: {missing_columns}")

    valid_cerebra_ids = set(region_data["Cerebra_ID"].unique())
    region_name_map = region_data.set_index("Cerebra_ID")["Region_name"].to_dict()
    mrt_id_map = region_data.set_index("Cerebra_ID")["Multiple resource theory ID"].to_dict()
    mrt_name_map = region_data.set_index("Multiple resource theory ID")["MRT ID Name"].to_dict()

    mrt_regions_map = {}
    rms_per_mrt = {}
    avg_rms_per_mrt = {}

    for cerebra_id, mrt_id in mrt_id_map.items():
        region_name = region_name_map.get(cerebra_id, f"Region {cerebra_id}")
        if mrt_id not in mrt_regions_map:
            mrt_regions_map[mrt_id] = []
        mrt_regions_map[mrt_id].append(region_name)

    rms_per_region = {cerebra_id: np.sqrt(np.mean(videodata[map_voxel == cerebra_id]**2, axis=0)) for cerebra_id in valid_cerebra_ids}

    for cerebra_id, mrt_id in mrt_id_map.items():
        if mrt_id not in rms_per_mrt:
            rms_per_mrt[mrt_id] = []
        rms_per_mrt[mrt_id].append(rms_per_region.get(cerebra_id, np.zeros(11250)))

    for mrt_id, rms_values_list in rms_per_mrt.items():
        avg_rms_per_mrt[mrt_id] = np.mean(rms_values_list, axis=0)

    mwl_index = np.sum(list(avg_rms_per_mrt.values()), axis=0)

    return avg_rms_per_mrt, mwl_index, mrt_regions_map, mrt_name_map

def plot_mrt_time_series3(rms_per_mrt, mwl_index, mrt_regions_map, mrt_name_map):
    """
    Plot the average time series for each MRT ID separately in red,
    and plot the overall average RMS in green.
    """
    for mrt_id, avg_rms_values in rms_per_mrt.items():
        mrt_name = mrt_name_map.get(mrt_id, "Unknown MRT Name")
        regions_str = ", ".join(mrt_regions_map.get(mrt_id, []))

        plt.figure(figsize=(16, 5))
        plt.plot(range(11250), avg_rms_values, color='red', label=f"{mrt_name} (MRT ID {mrt_id})")
        plt.xlabel("Time (0 - 11250)")
        plt.ylabel("Average RMS Value")
        plt.title(f"Average RMS for {mrt_name} (MRT ID {mrt_id})\nRegions: {regions_str}")
        plt.legend(loc='upper right', fontsize=8)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(11250), mwl_index, color='green', label="MWL Index ")
    plt.xlabel("Time (0 - 11250)")
    plt.ylabel("Sum of Averages of RMS of cortical regions")
    plt.title("Mental Workload Index")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()