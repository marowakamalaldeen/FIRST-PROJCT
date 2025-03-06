from data_functions import load_subject_video_data
from critical_regions import plot_number_of_voxel_per_region, get_number_of_voxel_per_region, plot_voxel_activation_for_region


program_exit = False  # Flag to control overall program exit

while not program_exit:
    # ----- Outer Loop: Subject Selection -----
    subject_input = input("\nEnter Subject ID (or type 'exit' to quit): ").strip()
    if subject_input.lower() == "exit":
        program_exit = True
        break  # Exit the outer loop

    subject_id = subject_input  # Use the entered subject ID

    # Load video data for the subject
    try:
        video_data = {
            "Baseline": load_subject_video_data(subject_id, "baseline"),
            "Video1": load_subject_video_data(subject_id, "video1"),
            "Video2": load_subject_video_data(subject_id, "video2"),
            "Video3": load_subject_video_data(subject_id, "video3"),
        }
    except Exception as e:
        print(f"Error loading video data for subject {subject_id}: {e}")
        continue  # Ask for a new subject ID

    # Optionally, plot the overall voxel distribution per cortical region
    plot_number_of_voxel_per_region()

    # ----- Middle Loop: Cerebra ID Analysis -----
    while True:
        cerebra_input = input("\nEnter Cerebra ID (or type 'back' to choose another subject, or 'exit' to quit): ").strip()
        if cerebra_input.lower() == "back":
            print("Returning to subject selection.")
            break  # Return to the subject selection loop
        if cerebra_input.lower() == "exit":
            program_exit = True
            break  # Exit the middle loop, then outer loop

        try:
            cerebra_id = int(cerebra_input)
            voxel_count, region_name = get_number_of_voxel_per_region(cerebra_id)
            if voxel_count == 0:
                print(f"⚠️ Region '{region_name}' (ID: {cerebra_id}) has 0 voxels. Please try another Cerebra ID.")
                continue  # Remain in the Cerebra ID loop
            else:
                print(f"✅ Selected Region: {region_name} (ID: {cerebra_id}) with {voxel_count} voxels.")
        except ValueError:
            print("❌ Invalid input. Please enter a valid Cerebra ID (an integer).")
            continue  # Remain in the Cerebra ID loop

        # ----- Inner Loop: Time Location Analysis -----
        while True:
            time_input = input("\nEnter a Time Location (or type 'back' to choose another Cerebra ID, or 'exit' to quit): ").strip()
            if time_input.lower() == "back":
                print("Returning to Cerebra ID selection.")
                break  # Break inner loop to re-enter a new Cerebra ID
            if time_input.lower() == "exit":
                program_exit = True
                break  # Break inner loop, then break out to end program

            try:
                time_location = int(time_input)
                # Validate time location using the Baseline video shape
                num_timestamps = video_data["Baseline"].shape[1]
                if time_location < 0 or time_location >= num_timestamps:
                    print(f"⚠️ Invalid Time Location. Please enter a value between 0 and {num_timestamps - 1}.")
                    continue

                # Plot voxel activation histograms for the selected region and time location
                plot_voxel_activation_for_region(time_location, cerebra_id, video_data)
            except ValueError:
                print("❌ Invalid input. Please enter a valid time location (integer).")

        if program_exit:
            break  # Break out of the Cerebra ID loop if user requested to exit

    if program_exit:
        break  # Break out of the subject loop if user requested to exit

print("\nThank you for using the program. Run ended.")