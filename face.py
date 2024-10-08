from ultralytics import YOLO
import os
import json
import pickle
from PIL import Image, ImageDraw
import numpy as np
import face_recognition

# Load a pre-trained YOLO model for person detection
model = YOLO("yolov10n.pt")

# Directory containing the images to process
image_dir = r"E:\Projects\Face-Ai\face-recog\Images"

# Directory to save unique faces
unique_faces_dir = os.path.join(image_dir, "unique_faces")
os.makedirs(unique_faces_dir, exist_ok=True)

# Initialize known faces lists
known_face_encodings = []  # List to hold known face encodings
known_face_names = []      # List to hold names corresponding to known face encodings
embeddings_file = "models/known_faces_embeddings.pkl"

# Load known faces if the embeddings file exists
if os.path.exists(embeddings_file):
    with open(embeddings_file, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)

# Function to add a new face to the known faces database
def add_new_face(face_encoding, name, face_crop):
    thumbnail_path = os.path.join(unique_faces_dir, f"{name}.png")
    face_crop.save(thumbnail_path)  # Save the cropped face image
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)
    return thumbnail_path

# Function to process an image and detect/recognize faces
def process_image(image, image_path):
    face_locations = face_recognition.face_locations(image, model="hog")
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if not face_locations:
        print("No faces detected in the image.")
        return {"names": []}, []  # Return empty unique faces info if no faces

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    names = []
    face_info = []
    unique_faces_info = []  # To store unique face data for output JSON

    base_name = os.path.basename(image_path)
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
            face_crop = pil_image.crop((left, top, right, bottom))  # Crop the detected face
            thumbnail_path = add_new_face(face_encoding, name, face_crop)
            unique_faces_info.append({
                "name": name,
                "thumbnail": thumbnail_path,
                "encoding": face_encoding.tolist()
            })

        names.append(name)
        face_info.append({"name": name, "box": {"top": top, "right": right, "bottom": bottom, "left": left}})

        # Draw rectangle and name on the image
        draw.rectangle(((left, top), (right, bottom)), outline="red", width=4)
        draw.text((left + 6, bottom - 10), name, fill="red")

    # Save the processed image
    processed_image_path = os.path.join(image_dir, "processed", os.path.basename(image_path))
    os.makedirs(os.path.dirname(processed_image_path), exist_ok=True)
    pil_image.save(processed_image_path)
    print(f"Processed image saved to: {processed_image_path}")

    # Prepare JSON output data
    output_data = {
        "image": processed_image_path,
        "names": names,
        "faces": face_info
    }

    return output_data, unique_faces_info

def main():
    # List all image files in the image directory (filtering common image formats)
    image_extensions = ['.jpg', '.jpeg', '.png']  # Add more formats if needed
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in image_extensions]

    # Main processing loop: YOLO detection followed by face recognition
    all_results = []
    unique_faces_list = []

    for img_path in image_paths:
        class_to_count = [0]  # Person class
        results = model(img_path, classes=class_to_count)  # Perform YOLO detection
        print(f"Detection complete for {img_path}.")

        # Process the first result image for face detection and recognition
        result_image = results[0].plot()  # Get the processed image from YOLO results
        try:
            result, unique_faces_info = process_image(result_image, img_path)
            all_results.append(result)
            unique_faces_list.extend(unique_faces_info)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Save all recognition results to a single JSON file
    json_output_path = os.path.join(image_dir, 'all_results.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(all_results, json_file, indent=4)
    print(f"All recognition results saved to: {json_output_path}")

    # Save unique faces data to another JSON file
    unique_faces_json_path = os.path.join(unique_faces_dir, 'unique_faces.json')
    with open(unique_faces_json_path, 'w') as json_file:
        json.dump(unique_faces_list, json_file, indent=4)
    print(f"Unique faces data saved to: {unique_faces_json_path}")

if __name__ == "__main__":
    main()
