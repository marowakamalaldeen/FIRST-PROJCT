a
    ���g�  �                   @   sD   d dl Zd dlZd dlZdd� Zdd� Zdd� Zdd	� Zd
d� Z	dS )�    Nc                 C   sZ   d| � d|� d�}t d|� �� tj�|�sPt d| � d|� d�� tj�dd	�S t�|�S )
zH
    Load brain activation data for a given subject and video type.
    z8/content/drive/MyDrive/Data TU PHD DUBLIN/subjects Data/z/evaluation/�.npyu   🔍 Checking file path: u(   ❌ Warning: File not found for subject z, video type �. Returning dummy data.�gz  ��+  )�print�os�path�exists�np�random�randn�load)�
subject_id�
video_type�	file_path� r   �oc:\Users\D23129149\OneDrive - Technological University Dublin\Documents\projects\FIRST-PROJCT\data_functions.py�load_subject_video_data   s    r   c                  C   s�   t d��� } tj�| �s6td| � d�� tj�dd�S tj�	| �r�dd� t�
| �D �}|r�td� t|�D ]\}}t|d	 � d
|� �� qjtt dt|�� d���d	 }tj�| || �} ntd| � d�� tj�dd�S t�| �S )zy
    Load brain activation data for a given subject and video type.
    Allows user to enter the file path manually.
    �9Enter the full file path for the video data (.npy file): �   ❌ Warning: File not found at r   r   r   c                 S   s   g | ]}|� d �r|�qS )r   )�endswith)�.0�fr   r   r   �
<listcomp>   �    z,load_subject_video_data1.<locals>.<listcomp>z0Found the following .npy files in the directory:�   z. zSelect the file to load (1-z): u0   ❌ Error: No .npy files found in the directory )�input�stripr   r   r	   r   r
   r   r   �isdir�listdir�	enumerate�int�len�joinr   )r   Z	npy_files�i�fileZ
file_indexr   r   r   �load_subject_video_data1   s    r&   c                  C   s2   t d��� } tj�| �s(td| � d���t�| �S )zO
    Load the region mapping data (CSV file) manually entered by the user.
    z2Enter the full file path for the region CSV file: u!   ❌ Error: CSV file not found at z. Please check the path.)r   r   r   r   r	   �FileNotFoundError�pd�read_csv)Zcsv_pathr   r   r   �load_region_data1-   s    r*   c                  C   s@   t d��� } tj�| �s6td| � d�� tj�dd�S t�	| �S )zE
    Manually enter the file path to load brain activation data.
    r   r   r   r   r   )
r   r   r   r   r	   r   r
   r   r   r   �r   r   r   r   �load_video_data9   s
    r,   c                  C   s6   t d��� } tj�| �s,td| � d�� dS t�| �S )z<
    Manually enter the file path to load the CSV file.
    z+Enter the full file path for the CSV file: r   z. Please enter a valid path.N)r   r   r   r   r	   r   r(   r)   r+   r   r   r   �load_csv_dataD   s
    r-   )
Znumpyr
   Zpandasr(   r   r   r&   r*   r,   r-   r   r   r   r   �<module>   s   