import os
import dlib
import cv2
import time
import shutil


def face_detection(input_dir, output_dir, model_path):
    # Load the CNN face detector
    cnn_face_detector = dlib.cnn_face_detection_model_v1(model_path)

    # Delete and recreate the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Start timing the execution
    start_time = time.time()

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check for image file types
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            # Check if the image is loaded successfully
            if image is None:
                print(f"Could not read image: {image_path}")
                continue

            # Convert the image to RGB (dlib uses RGB format)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect faces in the image
            detections = cnn_face_detector(rgb_image, 1)  # 1 is the number of times to upscale the image

            # Draw rectangles around detected faces
            for detection in detections:
                x, y, w, h = (
                detection.rect.left(), detection.rect.top(), detection.rect.right(), detection.rect.bottom())
                cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

            # Save the image with bounding boxes to the output directory
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, image)

    # End timing the execution
    end_time = time.time()
    total_time = end_time - start_time

    return total_time


# Example usage
input_directory = 'input_images'  # Replace with your input directory path
output_directory = 'output_images_dlib'  # Replace with your output directory path
model_file_path = 'mmod_human_face_detector.dat'  # Replace with the path to the model file

time_taken = face_detection(input_directory, output_directory, model_file_path)
print(f"Total time taken: {time_taken:.2f} seconds")
