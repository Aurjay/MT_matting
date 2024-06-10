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
    gt_files = sorted([file.replace('_gt', '').replace('.jpg', '.jpg') for file in os.listdir(gt_folder) if file.lower().endswith('.jpg')])

    print("Prediction Files:")
    print(pred_files)
    print("Ground Truth Files:")
    print(gt_files)

    total_metrics = {
        'IoU': 0,
        'Precision': 0,
        'Recall': 0,
        'F1 Score': 0,
        'Specificity': 0
    }
    total_images = 0

    for pred_file in pred_files:
        if pred_file in gt_files:
            gt_file = pred_file.replace('.jpg', '_gt.jpg')

            pred_path = os.path.join(pred_folder, pred_file)
            gt_path = os.path.join(gt_folder, gt_file)

            pred_image = read_and_threshold_image(pred_path)
            gt_image = read_and_threshold_image(gt_path)

            metrics = calculate_metrics(pred_image, gt_image)
            
            for key, value in metrics.items():
                total_metrics[key] += value
            
            total_images += 1
    
    average_metrics = {key: value / total_images for key, value in total_metrics.items()}

    with open('average_results_deepftsg.txt', 'w') as file:
        for metric_name, value in average_metrics.items():
            file.write(f"{metric_name}: {value}\n")

    return average_metrics

pred_folder = r'I:\Werkstudenten\Deepak_Raj\DATASETS\Results_all_models\DeepFTSG\SiemensGehen20m\o'
gt_folder = r'I:\Werkstudenten\Deepak_Raj\DATASETS\Private\GT\SiemensGehen20m'

average_metrics = process_folder(pred_folder, gt_folder)
