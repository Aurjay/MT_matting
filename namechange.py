import os

# Folder path
folder_path = r'I:\Werkstudenten\Deepak_Raj\Javed_Results\JavedSegmentationsForEvaluation\JavedSegmentationsForEvaluation_reduced_frames\SiemensGehen20mv2'

# Suffix to add to each file name
name_suffix = '_binary_mask'  # Change this to the desired suffix

# List files in the folder
files = os.listdir(folder_path)

# Iterate through files and add suffix
for file_name in files:
    # Skip directories
    if os.path.isdir(os.path.join(folder_path, file_name)):
        continue
    
    # Split file name and extension
    name, extension = os.path.splitext(file_name)
    
    # New file name with suffix
    new_file_name = f'{name}{name_suffix}{extension}'
    
    # Rename file
    os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))

print("Suffix added to file names.")
