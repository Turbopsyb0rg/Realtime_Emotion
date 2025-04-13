import cv2
import numpy as np
import threading
import time
from keras.models import model_from_json
from mtcnn.mtcnn import MTCNN

# -------------------------------
# Load the pre-trained Emotion Model
# -------------------------------
# Load model architecture
with open('emotionmodel.json', 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
# Load weights into the model
model.load_weights('emotionmodel.h5')

# Emotion labels dictionary
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# -------------------------------
# Define face preprocessing function
# -------------------------------
def preprocess_face(face_img):
    """
    Preprocess the face image for emotion classification:
    - Convert to grayscale
    - Apply CLAHE for adaptive histogram equalization
    - Resize to 48x48 as expected by the model
    - Normalize pixel values
    - Expand dimensions to match model input shape
    """
    # Convert face region to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    # Initialize CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    # Resize to match the model input size
    face_resized = cv2.resize(gray_eq, (48, 48))
    # Normalize pixel values
    face_norm = face_resized.astype("float32") / 255.0
    # Add channel and batch dimensions: (1, 48, 48, 1)
    face_input = np.expand_dims(face_norm, axis=-1)
    face_input = np.expand_dims(face_input, axis=0)
    return face_input

# -------------------------------
# Create a threaded video stream class
# -------------------------------
class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        # Read the first frame to initialize
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        # Use a lock for thread safety
        self.lock = threading.Lock()

    def start(self):
        # Start the thread for reading frames
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
            # Small sleep to reduce excessive CPU usage
            time.sleep(0.01)

    def read(self):
        # Safely return the latest frame
        with self.lock:
            frame_copy = self.frame.copy()
        return frame_copy

    def stop(self):
        self.stopped = True
        self.stream.release()

# -------------------------------
# Initialize MTCNN face detector and video stream
# -------------------------------
detector = MTCNN()
video_stream = WebcamVideoStream(src=0).start()

# -------------------------------
# Main Loop: Process each frame, detect faces and predict emotion
# -------------------------------
while True:
    try:
        # Read frame from the threaded video stream
        frame = video_stream.read()
        if frame is None:
            continue

        # MTCNN expects an RGB image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect faces in the current frame
        faces = detector.detect_faces(rgb_frame)

        # Process each detected face
        for face in faces:
            x, y, w, h = face['box']
            # Ensure the bounding box is within frame dimensions
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue
            # Extract the face region from the frame
            face_img = frame[y:y+h, x:x+w]

            # Check if face region is valid
            if face_img.size == 0:
                continue

            # Preprocess the face image for emotion classification
            face_input = preprocess_face(face_img)
            # Predict emotion using the model
            pred = model.predict(face_input)
            prediction_label = labels_dict[np.argmax(pred)]

            # Draw bounding box and emotion label on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, prediction_label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Display the output frame
        cv2.imshow("Real-time Emotion Detection", frame)

        # Exit if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        # Print unexpected errors and continue looping
        print("Error encountered:", e)
        continue

# -------------------------------
# Cleanup: Release resources
# -------------------------------
video_stream.stop()
cv2.destroyAllWindows()
