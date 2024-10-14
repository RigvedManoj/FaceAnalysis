import os
import time
import cv2

# Input and output directory paths
input_dir = 'input_images'  # Change this to your input directory path
output_dir = 'output_images_haar_cascade'  # Change this to your output directory path

# Delete and recreate the output directory
if os.path.exists(output_dir):
    for file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, file))
    os.rmdir(output_dir)  # Remove the output directory
os.mkdir(output_dir)  # Create a new output directory

# Load the Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the timer
start_time = time.time()

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image file types
        img_path = os.path.join(input_dir, filename)

        # Load the image
        img = cv2.imread(img_path)

        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw bounding boxes around detected faces
        for (x, y, width, height) in faces:
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Save the modified image in the output directory
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, img)

# End the timer
end_time = time.time()
total_time = end_time - start_time

# Print the total time taken
print(f"Total time taken: {total_time:.2f} seconds")
