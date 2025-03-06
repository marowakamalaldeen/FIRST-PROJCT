import numpy as np
import pandas as pd
import os

def load_subject_video_data(subject_id, video_type):
    """
    Load brain activation data for a given subject and video type.
    """
    file_path = f"/content/drive/MyDrive/Data TU PHD DUBLIN/subjects Data/{subject_id}/evaluation/{video_type}.npy"
    print(f"🔍 Checking file path: {file_path}")  # Debugging print
    if not os.path.exists(file_path):
        print(f"❌ Warning: File not found for subject {subject_id}, video type {video_type}. Returning dummy data.")
        return np.random.randn(31335, 11250)  # Dummy fallback data
    return np.load(file_path)


def load_subject_video_data1():
    """
    Load brain activation data for a given subject and video type.
    Allows user to enter the file path manually.
    """
    file_path = input("Enter the full file path for the video data (.npy file): ").strip()

    if not os.path.exists(file_path):
        print(f"❌ Warning: File not found at {file_path}. Returning dummy data.")
        return np.random.randn(31335, 11250)  # Dummy fallback data

    # Check if the provided path is a directory. If so, list the .npy files within it.
    if os.path.isdir(file_path):
        npy_files = [f for f in os.listdir(file_path) if f.endswith(".npy")]
        if npy_files:
            print("Found the following .npy files in the directory:")
            for i, file in enumerate(npy_files):
                print(f"{i + 1}. {file}")
            file_index = int(input(f"Select the file to load (1-{len(npy_files)}): ")) - 1
            file_path = os.path.join(file_path, npy_files[file_index])
        else:
            print(f"❌ Error: No .npy files found in the directory {file_path}. Returning dummy data.")
            return np.random.randn(31335, 11250)

    return np.load(file_path)



def load_region_data1():
    """
    Load the region mapping data (CSV file) manually entered by the user.
    """
    csv_path = input("Enter the full file path for the region CSV file: ").strip()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Error: CSV file not found at {csv_path}. Please check the path.")

    return pd.read_csv(csv_path)


def load_video_data():
    """
    Manually enter the file path to load brain activation data.
    """
    file_path = input("Enter the full file path for the video data (.npy file): ").strip()
    if not os.path.exists(file_path):
        print(f"❌ Warning: File not found at {file_path}. Returning dummy data.")
        return np.random.randn(31335, 11250)  # Dummy fallback data
    return np.load(file_path)


def load_csv_data():
    """
    Manually enter the file path to load the CSV file.
    """
    file_path = input("Enter the full file path for the CSV file: ").strip()
    if not os.path.exists(file_path):
        print(f"❌ Warning: File not found at {file_path}. Please enter a valid path.")
        return None
    return pd.read_csv(file_path)