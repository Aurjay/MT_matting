import os

def reorder_and_rename_images(directory):
    # Get a list of all files in the directory, sort them, and ignore .db files
    files = sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and not f.endswith('.db')])
    
    # Iterate through the files and rename them
    for i, filename in enumerate(files):
        # Extract the file extension
        file_extension = os.path.splitext(filename)[1]

        # Create the new filename with the correct format
        #new_filename = f"ann-img{i+1:04}_gt{file_extension}"
        new_filename = f"output-img{i+1:04}{file_extension}"
         #new_filename = f"org-img{i+1:04}{file_extension}"
        # Get the full paths for the current and new filenames
        current_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(current_path, new_path)
        
        print(f"Renamed {current_path} to {new_path}")

# Set the directory containing the images
directory = r'I:\Werkstudenten\Deepak_Raj\DATASETS\Results_all_models_final\public\MattingV2\cycle-noisy-bg'

reorder_and_rename_images(directory)
