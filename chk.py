import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import os
import pickle

embeddings_file = "models/known_faces_embeddings.pkl"

# Initialize known face encodings and names
known_face_encodings = []
known_face_names = []

# Load known faces if the embeddings file exists
if os.path.exists(embeddings_file):
    with open(embeddings_file, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    print("No known faces data found. Starting with empty lists.")

def add_new_face(face_encoding, name):
    """
    Adds a new face encoding and its name to the known faces list
    and saves the updated data to the embeddings file.
    """
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)
    with open(embeddings_file, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)

def process_image(input_image_path, output_image_path, image_url):
    """
    Processes the input image to detect and recognize faces,
    draws rectangles and names on recognized faces, and saves the output image.
    """
    if not os.path.exists(input_image_path):
        print(f"Input image file does not exist: {input_image_path}")
        return {"image": image_url, "names": []}

    image_file = Image.open(input_image_path)
    image = np.array(image_file)

    face_locations = face_recognition.face_locations(image, model="hog")
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if not face_locations:
        print("No faces detected in the image.")
        return {"image": image_url, "names": []}

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    names = []

    base_name = os.path.basename(input_image_path)
    name_from_filename = os.path.splitext(base_name)[0]

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if True in matches:
            match_index = np.argmin(face_distances)
            name = known_face_names[match_index]
            print(f"Existing face recognized as: {name}")
        else:
            name = name_from_filename
            print(f"New face detected. Assigned name: {name}")

            add_new_face(face_encoding, name)

        names.append(name)
        draw.rectangle(((left, top), (right, bottom)), outline="red", width=3)
        draw.text((left + 6, bottom - 10), name, fill="red")

    pil_image.save(output_image_path)
    print(f"Processed image saved to: {output_image_path}")

    return {"image": image_url, "names": names}
