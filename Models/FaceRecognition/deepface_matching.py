import cv2
from deepface import DeepFace
import numpy as np


def detect_faces_and_get_embeddings(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Use DeepFace to find faces
    results = DeepFace.represent(image_path, model_name='ArcFace', detector_backend='yolov8', enforce_detection=True)

    # Initialize a dictionary to store embeddings corresponding to each face
    face_embeddings = {}

    # Iterate over the results and extract embeddings
    for i, result in enumerate(results):
        # Extract the embedding
        # print(result)
        embedding = result['embedding']

        # Get face region for drawing bounding box
        x, y, width, height = result['facial_area']['x'], result['facial_area']['y'], result['facial_area']['w'], \
        result['facial_area']['h']

        # Use the bounding box coordinates as a key or simply use the index
        key = f"face_{i + 1}"  # Creating a unique identifier for each face
        face_embeddings[key] = {
            'embedding': embedding,
            'region': (x, y, width, height)
        }

        # Draw a rectangle around the detected face
        cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 2)
        cv2.putText(img, key, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convert embeddings to a numpy array for easier manipulation
    embeddings_array = np.array([face['embedding'] for face in face_embeddings.values()])

    # Display the image with bounding boxes
    cv2.imshow("Detected Faces", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return face_embeddings, embeddings_array

# Example usage
face_embeddings, embeddings_array = detect_faces_and_get_embeddings('input_Images/lots.jpg')

# Print the embeddings for each detected face
print("Face Embeddings:")
for key, value in face_embeddings.items():
    print(f"{key}: {value['embedding']}")
