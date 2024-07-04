import cv2
import numpy as np
import os

def resize_image(image, target_size):
    return cv2.resize(image, target_size)

def calculate_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    TP = np.sum(np.logical_and(pred == 1, gt == 1))
    FP = np.sum(np.logical_and(pred == 1, gt == 0))
    TN = np.sum(np.logical_and(pred == 0, gt == 0))
    FN = np.sum(np.logical_and(pred == 0, gt == 1))

    IoU = TP / float(TP + FP + FN) if (TP + FP + FN) != 0 else 0.0
    Precision = TP / float(TP + FP) if (TP + FP) != 0 else 0.0
    Recall = TP / float(TP + FN) if (TP + FN) != 0 else 0.0
    F1 = 2 * (Precision * Recall) / float(Precision + Recall) if (Precision + Recall) != 0 else 0.0
    Specificity = TN / float(TN + FP) if (TN + FP) != 0 else 0.0

    return {
        'IoU': IoU,
        'Precision': Precision,
        'Recall': Recall,
        'F1 Score': F1,
        'Specificity': Specificity
    }

def read_and_threshold_image(image_path, target_size=(256, 256), threshold=127):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    resized_image = resize_image(image, target_size)
    _, binary_image = cv2.threshold(resized_image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def process_folder(pred_folder, gt_folder):
    pred_files = sorted([file for file in os.listdir(pred_folder) if file.lower().endswith('.jpg')])
    gt_files = sorted([file for file in os.listdir(gt_folder) if file.lower().endswith('.jpg')])

    if len(pred_files) != len(gt_files):
        raise ValueError("The number of images in the prediction and ground truth folders do not match.")

    # Extract numbers from filenames
    pred_numbers = [int(file.split('-img')[1].split('.')[0]) for file in pred_files]
    gt_numbers = [int(file.split('-img')[1].split('_gt')[0]) for file in gt_files]

    # Sort both lists to ensure corresponding files match
    pred_files = [file for _, file in sorted(zip(pred_numbers, pred_files))]
    gt_files = [file for _, file in sorted(zip(gt_numbers, gt_files))]

    total_metrics = {
        'IoU': 0,
        'Precision': 0,
        'Recall': 0,
        'F1 Score': 0,
        'Specificity': 0
    }
    total_images = 0

    for pred_file, gt_file in zip(pred_files, gt_files):
        pred_path = os.path.join(pred_folder, pred_file)
        gt_path = os.path.join(gt_folder, gt_file)

        pred_image = read_and_threshold_image(pred_path)
        gt_image = read_and_threshold_image(gt_path)

        metrics = calculate_metrics(pred_image, gt_image)
        
        for key, value in metrics.items():
            total_metrics[key] += value
        
        total_images += 1
    
    average_metrics = {key: value / total_images for key, value in total_metrics.items()}

    output_path = r'I:\Werkstudenten\Deepak_Raj\DATASETS\Results_all_models_final\public\MattingV2\metrics\WavingTrees_output.txt'
    
    with open(output_path, 'w') as file:
        for metric_name, value in average_metrics.items():
            file.write(f"{metric_name}: {value}\n")

    return average_metrics

pred_folder = r'I:\Werkstudenten\Deepak_Raj\DATASETS\Results_all_models_final\public\MattingV2\WavingTrees_output'
gt_folder = r'I:\Werkstudenten\Deepak_Raj\DATASETS\Public\GT\WavingTrees_output'

average_metrics = process_folder(pred_folder, gt_folder)
print(average_metrics)
