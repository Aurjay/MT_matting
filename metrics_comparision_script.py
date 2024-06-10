import os
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Function to resize an image to the specified dimensions
def resize_image(image, target_size=(380, 244)):
    return image.resize(target_size)

# Paths to the folders containing images
output_folder = r'I:\Werkstudenten\Deepak_Raj\Javed_Results\JavedSegmentationsForEvaluation\JavedSegmentationsForEvaluation_reduced_frames\SiemensGehen20mv2'
ground_truth_folder = r'I:\Werkstudenten\Deepak_Raj\Javed_Results\JavedSegmentationsForEvaluation\GT\SiemensGehen20m'

# List all files in both folders
output_files = [file for file in os.listdir(output_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
ground_truth_files = [file for file in os.listdir(ground_truth_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# Initialize variables to store averages
avg_accuracy = 0
avg_precision = 0
avg_recall = 0
avg_f1 = 0

num_samples = 0

# Iterate through the images in the output folder
for output_file in output_files:
    # Extract the identifier from the filename
    identifier = output_file.split('_')[0]  # Assuming the identifier comes before '_'

    # Search for the corresponding ground truth file
    ground_truth_file = [file for file in ground_truth_files if identifier in file]

    # Ensure there is exactly one corresponding ground truth file

    # Load the output and ground truth images
    try:
        output_image = Image.open(os.path.join(output_folder, output_file)).convert('L')
        ground_truth_image = Image.open(os.path.join(ground_truth_folder, ground_truth_file[0])).convert('L')

        # Resize images to the same dimensions
        output_image = resize_image(output_image)
        ground_truth_image = resize_image(ground_truth_image)

        # Convert images to numpy arrays
        output_image_np = np.array(output_image)
        ground_truth_image_np = np.array(ground_truth_image)

        # Ensure both images have the same shape
        assert output_image_np.shape == ground_truth_image_np.shape, "Images must have the same dimensions"

        # Convert images to binary
        output_binary = np.where(output_image_np > 127, 1, 0)
        ground_truth_binary = np.where(ground_truth_image_np > 127, 1, 0)

        # Flatten images to 1D arrays
        output_flat = output_binary.flatten()
        ground_truth_flat = ground_truth_binary.flatten()

        # Calculate true positives, false positives, and false negatives
        TP = np.sum(np.logical_and(output_flat == 1, ground_truth_flat == 1))
        FP = np.sum(np.logical_and(output_flat == 1, ground_truth_flat == 0))
        FN = np.sum(np.logical_and(output_flat == 0, ground_truth_flat == 1))

        # Calculate accuracy
        accuracy = accuracy_score(ground_truth_flat, output_flat)

        # Calculate precision, recall, and F1 score
        precision = precision_score(ground_truth_flat, output_flat, zero_division='warn')
        recall = recall_score(ground_truth_flat, output_flat)
        f1 = f1_score(ground_truth_flat, output_flat)

        # Update averages
        avg_accuracy += accuracy
        avg_precision += precision
        avg_recall += recall
        avg_f1 += f1

        num_samples += 1
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
    except AssertionError as e:
        print(f"Assertion error: {e}")

# Calculate averages
avg_accuracy /= num_samples
avg_precision /= num_samples
avg_recall /= num_samples
avg_f1 /= num_samples

# Create DataFrame with averages
results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Average': [avg_accuracy, avg_precision, avg_recall, avg_f1]
})

# Save the DataFrame to an Excel file
excel_file_path = os.path.join(output_folder, 'javed_metrics_results.xlsx')
results_df.to_excel(excel_file_path, index=False)

print("Metrics averages saved to:", excel_file_path)
