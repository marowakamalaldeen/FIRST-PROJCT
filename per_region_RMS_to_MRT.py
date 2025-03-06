from data_functions import load_csv_data
from data_functions import load_video_data
from mental_workload import compute_rms_time_series3
from mental_workload import plot_mrt_time_series3

# --------------------- Execution ---------------------
while True:
    # Load video data.
    videodata = load_video_data()
    # Load region mapping CSV.
    region_data = load_csv_data()
    if region_data is None:
        continue  # Retry if no valid CSV file was provided.

    # Simulated voxel-to-region mapping (ensure size matches videodata's voxel count).
    map_voxel = np.random.randint(1, 103, size=videodata.shape[0])

    subject_id = input("Enter the Subject ID: ").strip()
    video_type = input("Enter the Video Type: ").strip()

    try:
        avg_rms_per_mrt, mwl_index, mrt_regions_map, mrt_name_map = compute_rms_time_series3(videodata, map_voxel, region_data)
        print("RMS Computation and MRT Sorting Complete.")
        plot_rms_time_series3(avg_rms_per_mrt, mwl_index, mrt_regions_map, mrt_name_map, subject_id, video_type)
    except (KeyError, FileNotFoundError, ValueError) as e:
        print(f"❌ Error: {e}")

    cont = input("Do you want to process another dataset? (yes/no): ").strip().lower()
    if cont != 'yes':
        break

print("\nThank you for using the program. Run ended.")