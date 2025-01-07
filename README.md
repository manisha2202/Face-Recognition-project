# Face-Recognition-project
face
from keras.models import load_model
import cv2
import numpy as np
import sqlite3
from datetime import datetime

# Load the trained model
model = load_model("keras_Model.h5", compile=False)

# Load class labels
class_names = open("labels.txt", "r").readlines()

# Initialize the SQLite database connection
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

# Create the attendance table with more details
cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    confidence REAL NOT NULL,
    remarks TEXT,
    UNIQUE(name, date)  -- Composite unique constraint
)
""")
conn.commit()

# Pre-populate the database with sample input data (can be skipped if already populated)
sample_data = [
    ('manisha', '2025-01-06', '09:00:00', 0.95, 'On Time'),
    ('kavya', '2025-01-06', '09:45:12', 0.92, 'Late'),
    ('dhanya', '2025-01-06', '08:30:00', 0.99, 'On Time')
]

# Insert sample data into the table if it's not already there
for record in sample_data:
    cursor.execute("""
    INSERT OR IGNORE INTO attendance (name, date, time, confidence, remarks) 
    VALUES (?, ?, ?, ?, ?)
    """, record)
conn.commit()

# Function to mark attendance
def mark_attendance(name, confidence):
    """Marks attendance for the detected person in the database."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    remarks = "On Time" if current_time < "09:30:00" else "Late"

    try:
        # Check if the person is already marked for today
        cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, current_date))
        existing_record = cursor.fetchone()

        if existing_record:
            print(f"Attendance already marked for {name} on {current_date}.")
        else:
            cursor.execute("""
            INSERT INTO attendance (name, date, time, confidence, remarks) 
            VALUES (?, ?, ?, ?, ?)
            """, (name, current_date, current_time, confidence, remarks))
            conn.commit()
            print(f"Attendance marked for {name} on {current_date} at {current_time} (Confidence: {confidence:.2f})")

    except sqlite3.Error as e:
        print(f"Error marking attendance: {e}")

# Open the webcam
camera = cv2.VideoCapture(1)

print("Press ESC to exit the program.")

while True:
    # Capture frame from webcam
    ret, frame = camera.read()

    if not ret:
        print("Failed to grab the frame from the camera.")
        break

    # Resize the frame to (224, 224) for model input
    image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    # Display the frame
    cv2.imshow("Webcam Image", frame)

    # Prepare the image for prediction
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1  # Normalize the image

    # Predict the class
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Print the prediction
    print(f"Class: {class_name} | Confidence Score: {confidence_score:.2f}")

    # Mark attendance if confidence is above threshold
    if confidence_score > 0.9:
        mark_attendance(class_name, confidence_score)

    # Exit on pressing ESC key
    if cv2.waitKey(1) & 0xFF == 27:  # ASCII for ESC key
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
conn.close()
print("Program terminated.")
