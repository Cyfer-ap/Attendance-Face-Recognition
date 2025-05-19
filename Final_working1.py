import sys
import cv2
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1, prewhiten
import torch
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image
import csv
import pandas as pd
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import base64
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QDialog,
    QLineEdit, QMessageBox, QInputDialog, QPlainTextEdit, QSizePolicy,  QGridLayout
)
# Load pre-trained models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MTCNN for face detection
mtcnn = MTCNN()

# Inception Resnet V1 for face recognition
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# MTCNN for face detection
mtcnn = MTCNN()

# Inception Resnet V1 for face recognition
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# File to store face data
face_data_file = 'face_data.csv'

# List to store known faces and their embeddings
known_faces = []

class AdminPasswordDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Admin Password')
        self.setGeometry(200, 200, 300, 150)

        self.password_label = QLabel('Enter Admin Password:', self)
        self.password_edit = QLineEdit(self)
        self.password_edit.setEchoMode(QLineEdit.Password)
        
        self.ok_button = QPushButton('OK', self)
        self.ok_button.clicked.connect(self.check_password)

        self.cancel_button = QPushButton('Cancel', self)
        self.cancel_button.clicked.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_edit)
        layout.addWidget(self.ok_button)
        layout.addWidget(self.cancel_button)

    def check_password(self):
        admin_password = 'Cyfer'  # Replace with your actual admin password

        if self.password_edit.text() == admin_password:
            self.accept()  # Accept the dialog if the password is correct
        else:
            QMessageBox.warning(self, 'Incorrect Password', 'Incorrect admin password. Please try again.')

# Class for the GUI application
class FaceRecognitionApp(QWidget):
    attendance_file = 'Record_Attendance.xlsx'  # Define attendance_file as a class variable
    
    def __init__(self):
        super().__init__()

        self.video_capture = cv2.VideoCapture(0)

        # Initialize known faces and embeddings
        self.face_data_file = 'face_data.csv'
        self.load_embeddings()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Face Recognition System')

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.start_button = QPushButton('Start Recognition', self)
        self.start_button.clicked.connect(self.start_recognition)
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white;")

        self.quit_button = QPushButton('Quit', self)
        self.quit_button.clicked.connect(self.quit_application)
        self.quit_button.setStyleSheet("background-color: #f44336; color: white;")

        self.message_box = QPlainTextEdit(self)
        self.message_box.setReadOnly(True)
        self.message_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.message_box.setStyleSheet("QPlainTextEdit { background-color: #e0e0e0; color: #333; }")

        layout = QGridLayout(self)
        layout.addWidget(self.image_label, 0, 0, 1, 2)
        layout.addWidget(self.start_button, 1, 0)
        layout.addWidget(self.quit_button, 1, 1)
        layout.addWidget(self.message_box, 2, 0, 1, 2)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.show()

    
    def print_to_gui(self, message):
        self.message_box.appendPlainText(message)
        
    
    def crop_and_align_face(self, img, box, target_size=(160, 160)):
        x, y, w, h = [int(coord) for coord in box]
        face_roi = img[y:y+h, x:x+w]

        # Resize the face to the target size
        face_pil = Image.fromarray(face_roi)
        face_resized = F.resize(face_pil, target_size)

        # Convert the PIL image to a torch tensor
        face_tensor = F.to_tensor(face_resized).unsqueeze(0)

        # Prewhiten the face tensor
        prewhitened_face = prewhiten(face_tensor)

        return prewhitened_face
    
    def base64_to_numpy(self, string, dtype=np.float32):
        decoded_bytes = base64.b64decode(string)
        return np.frombuffer(decoded_bytes, dtype=dtype)

    def match_face(self, embeddings, known_embeddings, threshold=0.6):
        if known_embeddings is not None:
            if len(embeddings) == len(known_embeddings):
                # Calculate L2 distance between the embeddings
                distance = np.linalg.norm(embeddings - known_embeddings)

                # Check if the distance is below the threshold
                return distance < threshold

        return False
    
    def start_recognition(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform face recognition for each detected face
            for face in mtcnn.detect_faces(frame):
                box = face['box']
                x, y, w, h = [int(coord) for coord in box]

                # Recognize face using FaceNet model
                known_face = self.recognize_face(frame[y:y+h, x:x+w])

                # Dummy attendance marking
                if known_face:
                    self.mark_attendance(known_face['name'])

            # Display the resulting frame
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)

    def update_frame(self):
        ret, frame = self.video_capture.read()

        # Display the resulting frame
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)

    def quit_application(self):
        self.video_capture.release()
        self.close()

    def recognize_face(self, face_roi):
        # Detect faces in the image using MTCNN
        faces = mtcnn.detect_faces(face_roi)

        if faces:
            for face in faces:
                # Take each face detected
                box = face['box']

                # Extract face embeddings
                aligned = self.crop_and_align_face(face_roi, box)
                embeddings = model(aligned.to(device)).detach().cpu().numpy().flatten()  # Flatten the embeddings

                # Compare with all known faces
                for known_face in self.known_faces:
                    if self.match_face(embeddings, known_face.get('embeddings')):
                        print(f"Person recognized: {known_face['name']}")
                        return known_face

                # If the face is not recognized, ask if the user wants to add it
                print("Person not recognized.")
                self.print_to_gui("Person not recognized.")  # Add this line to display in the message box
                reply = QMessageBox.question(self, 'Recognition', 'Do you want to add this face to the record?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

                if reply == QMessageBox.Yes:
                    self.show_admin_password_dialog(embeddings)

        return None
    
    def show_admin_password_dialog(self, embeddings):
        dialog = AdminPasswordDialog()
        result = dialog.exec_()

        if result == QDialog.Accepted:
            password = dialog.password_edit.text()
            self.handle_admin_password(embeddings, password, dialog)

    def add_face_to_csv(self, name, embeddings):
        try:
            with open(self.face_data_file, 'a', newline='') as csvfile:
                fieldnames = ['Name', 'Embeddings', 'Timestamp']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Check if the file is empty
                if csvfile.tell() == 0:
                    writer.writeheader()

                # Serialize embeddings to base64
                embeddings_base64 = self.numpy_to_base64(embeddings)

                # Append face data
                writer.writerow({'Name': name, 'Embeddings': embeddings_base64, 'Timestamp': pd.Timestamp.now()})

            self.known_faces.append({'name': name, 'embeddings': embeddings.flatten()})  # Flatten the embeddings
            print("Face added to the record.")
        except Exception as e:
            print(f"Error: {e}")

    def numpy_to_base64(self, array):
            return base64.b64encode(array.tobytes()).decode('utf-8')
    
    def update_message(self, message):
        self.message_box.appendPlainText(message)
    
    def handle_admin_password(self, embeddings, password, dialog):
        # Check if the password is correct
        correct_password = "Cyfer"  # Replace with your actual admin password

        if password == correct_password:
            # Get name from the user
            name, ok_pressed = QInputDialog.getText(self, 'Add Face', 'Enter the name for this face:')
            if ok_pressed:
                self.add_face_to_csv(name, embeddings)
                dialog.accept()
        else:
            # Display an error message or take appropriate action
            QMessageBox.warning(self, 'Incorrect Password', 'Incorrect admin password. Please try again.')
            
    def mark_attendance(self, name):
        try:
            # Read existing attendance data
            try:
                attendance_data = pd.read_excel(self.attendance_file)  # Use self.attendance_file
            except FileNotFoundError:
                # Create a new DataFrame if the file doesn't exist
                attendance_data = pd.DataFrame(columns=['Name', 'Time', 'Date', 'Status'])

            # Get current date and time
            current_date = datetime.now().strftime('%Y-%m-%d')
            current_time = datetime.now().strftime('%H:%M:%S')

            # Check if attendance for the student on the current date already exists
            existing_attendance = attendance_data[(attendance_data['Name'] == name) & (attendance_data['Date'] == current_date)]

            if not existing_attendance.empty:
                # Attendance already marked for the day
                self.print_to_gui(f"Attendance for {name} already marked today at {existing_attendance.iloc[0]['Time']}.")
            else:
                # Set the required time for attendance
                required_time = datetime.strptime('09:00:00', '%H:%M:%S')

                # Check if the attendance is early, on time, or late
                status = 'On Time'
                if datetime.strptime(current_time, '%H:%M:%S') > required_time:
                    status = 'Late'
                elif datetime.strptime(current_time, '%H:%M:%S') < required_time:
                    status = 'Early'

                # Append the attendance record to the DataFrame
                new_record = {'Name': name, 'Time': current_time, 'Date': current_date, 'Status': status}
                attendance_data = pd.concat([attendance_data, pd.DataFrame([new_record])], ignore_index=True)

                # Save the updated DataFrame to the Excel file
                attendance_data.to_excel(self.attendance_file, index=False)

                self.print_to_gui(f"Attendance marked for {name} ({status}) at {current_time} on {current_date}.")

        except Exception as e:
            print(f"Error: {e}")

    
    
    def load_embeddings(self):
        try:
            self.known_faces = []  # Clear the known_faces list
            with open(self.face_data_file, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    name = row['Name']
                    embeddings_base64 = row['Embeddings']

                    # Deserialize embeddings from base64
                    embeddings = self.base64_to_numpy(embeddings_base64)
                    self.known_faces.append({'name': name, 'embeddings': embeddings})  # Do not flatten the embeddings
        except Exception as e:
            print(f"Error: {e}")
    
if __name__ == '__main__':
    app = QApplication([])
    face_recognition_app = FaceRecognitionApp()
    sys.exit(app.exec_())
