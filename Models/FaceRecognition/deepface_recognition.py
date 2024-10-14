import csv

from deepface import DeepFace
import time

output_csv = "deepface_results.csv"

models = [
    "VGG-Face",
    "Facenet",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "Dlib",
    "ArcFace"
]

backends = [
    'opencv',
    'ssd',
    'dlib',
    'mtcnn',
    'fastmtcnn',
    'retinaface',
    'mediapipe',
    'yolov8',
    'yunet',
    'centerface',
]

with open(output_csv, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Model", "Detector", "Result", "Time"])
    for model in models:
        for backend in backends:
            start_time = time.time()
            answer = []
            for image1 in ['Anju.jpeg', 'Me.png']:
                for image2 in ['Anju2.jpeg', 'Me2.jpg']:
                    try:
                        result = DeepFace.verify(
                            img1_path="input_Images/" + image1,
                            img2_path="input_Images/" + image2,
                            model_name=model,
                            enforce_detection=True,
                            detector_backend=backend
                        )
                        answer.append(result['verified'])
                    except Exception as e:
                        answer.append(False)
                        pass
            end_time = time.time()
            total_time = end_time - start_time
            print(answer)
            writer.writerow([model, backend, answer, total_time])
