import cv2

def resize_image(image, target_size):
    return cv2.resize(image, target_size)

def read_and_threshold_image(image_path, target_size=(256, 256), threshold=127):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = resize_image(grayscale_image, target_size)
    _, binary_image = cv2.threshold(resized_image, threshold, 255, cv2.THRESH_BINARY)

    return image, grayscale_image, binary_image

def process_image(image_path):
    pred_image, pred_grayscale, pred_thresholded = read_and_threshold_image(image_path)

    # Display images
    cv2.imshow('Original Image', pred_image)
    cv2.waitKey(0)

    cv2.imshow('Grayscale Image', pred_grayscale)
    cv2.waitKey(0)

    cv2.imshow('Thresholded Image', pred_thresholded)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

image_path = r'i:\Werkstudenten\Deepak_Raj\DATASETS\Results_all_models_final\public\MattingV2\Walking-camera-shaking\output-img0032.jpg'

process_image(image_path)
