import urllib.request
import os

# Download the pre-trained model
url = "https://github.com/oarriaga/face_classification/blob/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5?raw=true"
filename = "fer2013_mini_XCEPTION.102-0.66.hdf5"
if not os.path.isfile(filename):
    urllib.request.urlretrieve(url, filename)

import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

# Define the emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the frame to match the input size of the model
    gray = cv2.resize(gray, (64, 64))
    
    # Reshape the frame to match the input shape of the model
    gray = np.reshape(gray, (1, 64, 64, 1))
    
    # Normalize the pixel values
    gray = gray / 255.0
    
    # Make a prediction using the model
    prediction = model.predict(gray)
    
    # Get the emotion with the highest probability
    index = np.argmax(prediction)
    emotion = emotions[index]
    
    # Display the emotion on the frame
    cv2.putText(frame, emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('frame', frame)
    
    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()