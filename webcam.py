import os
import math
import cv2
from deepface import DeepFace
import numpy as np
from uuid import uuid4
import torch
import torchvision.ops.boxes as bops
from time import time
from push import Visit, ELK_INDEX
from datetime import datetime


detected = {

}

IOU_THRESHOLD = 0.6


def logic(faces, width, height):
    last_upd_global = time()
    candidates = set()

    for frame, x1, y1, w1, h1 in faces:
        same_idx = None
        prev_iou = -1
        for idx, (_, x2, y2, w2, h2, _, _, _) in detected.items():
            box1 = torch.tensor([[x1, y1, x1 + w1, y1 + h1]], dtype=torch.float)
            box2 = torch.tensor([[x2, y2, x2 + w2, y2 + h2]], dtype=torch.float)
            iou = bops.box_iou(box1, box2)[0]
            if iou > IOU_THRESHOLD and iou > prev_iou:
                same_idx = idx
                prev_iou = iou

        if same_idx and same_idx not in candidates:
            candidates.add(same_idx)
            image = frame[y1:y1+h1,x1:x1+w1]
            face_aspect = round((w1*h1)/(width*height), 3)
            if face_aspect > 0.05:
                cv2.imwrite(f'faces/{str(same_idx)[:7]}_{face_aspect > 0.05}_{face_aspect}.jpg', image)
            detected[same_idx] = (
                image, x1, y1, w1, h1, detected[same_idx][5], detected[same_idx][6], last_upd_global
            )
        else:
            face_id = uuid4()
            image = frame[y1:y1+h1,x1:x1+w1]
            face_aspect = round((w1*h1)/(width*height), 3)
            cv2.imwrite(f'faces/{str(face_id)[:7]}_{face_aspect > 0.05}_{face_aspect}.jpg', image)
            detected[face_id] = (image, x1, y1, w1, h1, [], last_upd_global, last_upd_global)

    left_faces = []
    for idx, (frame, x, y, w, h, emotions, creation_date, last_upd) in detected.items():
        # if last_upd - creation_date < 0.5:
        #     img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)  # making a recentangle to show up and detect the face and setting it position and colour

        if last_upd_global - last_upd > 1.5:
            left_faces.append(idx)
        # making a try and except condition in case of any errors
        if last_upd_global == last_upd:
            face_aspect = (w*h)/(width*height)
            if face_aspect < 0.05:
                continue
            try:
                analyze = DeepFace.analyze(frame, actions=['emotion'], prog_bar=False)  # same thing is happing here as the previous example, we are using the analyze class from deepface and using ‘frame’ as input
                emo = analyze['dominant_emotion']
                # print(emo)  # here we will only go print out the dominant emotion also explained in the previous example
                emotions.append(emo)
            except:
                print("no face")

        # this is the part where we display the output to the user
        # cv2.imshow('video', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):   # here we are specifying the key which will stop the loop and stop all the processes going
            break
    if len(left_faces):
        print('LEFT:', len(left_faces))
    for idx in left_faces:
        data = detected.pop(idx, None)
        if data is None:
            print('Warn: no data!')
            continue

        _, x, y, w, h, emotions, creation_date, last_upd = data
        if last_upd - creation_date < 5 or len(emotions) == 0:
            print('Warn: last upd or emos!', last_upd - creation_date)
            continue
        print('SEND!',
            datetime.fromtimestamp(int(creation_date)),
            datetime.fromtimestamp(int(last_upd)),
            os.environ.get('SHOP_NAME', 'default'),
            max(set(emotions), key=emotions.count),
            sep='\n\t')
        act = Visit(
            creation_date=datetime.fromtimestamp(int(creation_date) - 5*3600),
            last_update=datetime.fromtimestamp(int(last_upd) - 5*3600),
            shop_name=os.environ.get('SHOP_NAME', 'Default'),
            emotion=max(set(emotions), key=emotions.count)
        )
        act.save(index=ELK_INDEX)


if __name__ == '__main__':

    face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'  # getting a haarcascade xml file
    face_cascade = cv2.CascadeClassifier()  # processing it for our project
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):  # adding a fallback event
        print("Error loading xml file")

    video = cv2.VideoCapture('IMG_1083.mov')  #requisting the input from the webcam or camera
    # checking if are getting video feed and using it

    frame_rate = video.get(cv2.CAP_PROP_FPS)
    width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    print('FRAME RATE:', frame_rate)
    print('WIDTH:', width)
    print('HEIGHT:', height)

    fps = 10
    if fps > frame_rate:
        fps = frame_rate

    current_frame = 0

    while video.isOpened():
        _, frame = video.read()

        current_frame += 1
        if current_frame % (math.floor(frame_rate / fps)) != 0:
            continue

        # changing the video to grayscale to make the face analisis work properly
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        # print('Count of faces:', len(faces))
        logic(faces, width, height)

    video.release()
