import os
import time
import cv2
from mtcnn import MTCNN

# Input and output directory paths
input_dir = 'input_images'  # Change this to your input directory path
output_dir = 'output_images_MTCNN'  # Change this to your output directory path

# Delete and recreate the output directory
if os.path.exists(output_dir):
    os.rmdir(output_dir)  # Remove the output directory
os.mkdir(output_dir)  # Create a new output directory

# Initialize the face detector
detector = MTCNN()

# Start the timer
start_time = time.time()

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image file types
        img_path = os.path.join(input_dir, filename)

        # Load the image and convert it from BGR to RGB
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        faces = detector.detect_faces(img)

        # Draw bounding boxes around detected faces
        for face in faces:
            x, y, width, height = face['box']
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Convert the image back to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Save the modified image in the output directory
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, img_bgr)

# End the timer
end_time = time.time()
total_time = end_time - start_time

# Print the total time taken
print(f"Total time taken: {total_time:.2f} seconds")
