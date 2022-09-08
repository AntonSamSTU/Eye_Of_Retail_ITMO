import cv2
from deepface import DeepFace
import numpy as np  # this will be used later in the process
from time import time

# emotions: ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

for imgpath in ['many.jpg', 'happy.jpg', 'sad.jpg']:
    s = time()
    image = cv2.imread(imgpath)
    analyze = DeepFace.analyze(image, actions=['emotion'])  #here the first parameter is the image we want to analyze #the second one there is the action
    print('*'*50, '\n', analyze, time() - s)
