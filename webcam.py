import math
from time import time
from uuid import uuid4
from datetime import datetime

import cv2
import torch
import torchvision.ops.boxes as bops

from deepface import DeepFace

from push import Visit
import settings


detected = {}


def logic(faces, width, height):
    last_upd_global = time()
    candidates = set()

    for frame, x1, y1, w1, h1 in faces:
        same_idx = None
        prev_iou = -1
        for face_id, (_, x2, y2, w2, h2, _, _, _) in detected.items():
            box1 = torch.tensor([[x1, y1, x1 + w1, y1 + h1]], dtype=torch.float)
            box2 = torch.tensor([[x2, y2, x2 + w2, y2 + h2]], dtype=torch.float)
            iou = bops.box_iou(box1, box2)[0]
            if iou > settings.IOU_THRESHOLD and iou > prev_iou:
                same_idx = face_id
                prev_iou = iou

        if same_idx and same_idx not in candidates:
            candidates.add(same_idx)
            image = frame[y1:y1+h1,x1:x1+w1]
            detected[same_idx] = (
                image, x1, y1, w1, h1, detected[same_idx][5], detected[same_idx][6], last_upd_global
            )
        else:
            face_id = uuid4()
            image = frame[y1:y1+h1, x1:x1+w1]
            detected[face_id] = (image, x1, y1, w1, h1, [], last_upd_global, last_upd_global)

    left_faces = []
    for face_id, (frame, x, y, w, h, emotions, creation_date, last_upd) in detected.items():
        if last_upd_global - last_upd > 1.5:
            left_faces.append(face_id)

        if last_upd_global == last_upd and len(emotions) < 8:
            face_aspect = (w * h) / (width * height)
            if face_aspect < settings.FACE_ASPECT_RATIO:
                continue
            try:
                # we are using the analyze class from deepface and using ‘frame’ as input
                analyze = DeepFace.analyze(frame, actions=['emotion'], prog_bar=False)
                emo = analyze['dominant_emotion']
                emotions.append(emo)
            except Exception:
                pass

        key = cv2.waitKey(1)
        if key == ord('q'):   # here we are specifying the key which will stop the loop and stop all the processes going
            break

    for face_id in left_faces:
        data = detected.pop(face_id, None)
        if data is None:
            continue

        *_, emotions, creation_date, last_upd = data

        if last_upd - creation_date < 5 or len(emotions) == 0:
            print('Warn: last upd or emos!', last_upd - creation_date)
            continue

        print('SEND!',
            datetime.fromtimestamp(int(creation_date)),
            datetime.fromtimestamp(int(last_upd)),
            settings.SHOP_NAME,
            max(set(emotions), key=emotions.count),
            sep='\n\t'
        )

        act = Visit(
            creation_date=datetime.fromtimestamp(int(creation_date) - 5*3600),
            last_update=datetime.fromtimestamp(int(last_upd) - 5*3600),
            shop_name=settings.SHOP_NAME,
            emotion=max(set(emotions), key=emotions.count)
        )
        act.save(index=settings.ELK_INDEX)



if __name__ == '__main__':

    face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'  # getting a haarcascade xml file
    face_cascade = cv2.CascadeClassifier()  # processing it for our project
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):  # adding a fallback event
        print("Error loading xml file")

    video = cv2.VideoCapture(settings.VIDEO_CAPTURE)  #requisting the input from the webcam or camera
    # checking if are getting video feed and using it

    frame_rate = video.get(cv2.CAP_PROP_FPS)
    width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    fps = settings.CAM_FMS
    if fps > frame_rate:
        fps = frame_rate

    current_frame = 0

    while video.isOpened():
        _, frame = video.read()

        current_frame += 1
        if current_frame % (math.floor(frame_rate / fps)) != 0:
            continue

        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        faces = [[frame, *coords] for coords in faces]

        logic(faces, width, height)

    video.release()
