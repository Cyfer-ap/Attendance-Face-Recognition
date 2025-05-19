# Face Recognition Attendance System

This project is a **Face Recognition Attendance System** built using Python. It uses facial recognition to mark attendance and provides a graphical user interface (GUI) for ease of use.

## Features

- **Face Detection and Recognition**: Uses MTCNN for face detection and FaceNet for face recognition.
- **Attendance Marking**: Automatically marks attendance in an Excel file (`Record_Attendance.xlsx`) with details like name, time, date, and status (Early, On Time, or Late).
- **Admin Authentication**: Allows only authorized users to add new faces to the system.
- **GUI Interface**: Built with PyQt5 for an interactive user experience.
- **Face Data Storage**: Stores face embeddings in a CSV file (`face_data.csv`) for future recognition.

## Requirements

- Python 3.7 or higher
- Libraries:
  - `opencv-python`
  - `mtcnn`
  - `facenet-pytorch`
  - `torch`
  - `numpy`
  - `pandas`
  - `PyQt5`
  - `Pillow`
  - `torchvision`
  - `openpyxl`

Install the required libraries using:

```bash
pip install opencv-python mtcnn facenet-pytorch torch numpy pandas PyQt5 Pillow torchvision openpyxl
