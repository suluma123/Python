import cv2
import numpy as np
import pyautogui

# Load the Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Set the screen size
screen_width, screen_height = pyautogui.size()

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the frame
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # If eyes are detected, move the mouse
    if len(eyes) > 0:
        # Get the coordinates of the first detected eye
        (x, y, w, h) = eyes[0]
        
        # Calculate the center of the eye
        eye_center_x = x + w // 2
        eye_center_y = y + h // 2

        # Normalize the eye position to screen size
        mouse_x = np.interp(eye_center_x, [0, frame.shape[1]], [0, screen_width])
        mouse_y = np.interp(eye_center_y, [0, frame.shape[0]], [0, screen_height])

        # Move the mouse
        pyautogui.moveTo(mouse_x, mouse_y)

    # Display the resulting frame
    cv2.imshow('Eye Controlled Mouse', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
