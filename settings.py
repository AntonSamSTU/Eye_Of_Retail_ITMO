import os
import base64

# DEVICE
SHOP_NAME = os.environ.get('SHOP_NAME', 'Default')

CAM_FMS = int(os.environ.get('MAX_FPS', 8))
IOU_THRESHOLD = float(os.environ.get('IOU_THRESHOLD', 0.3))
VIDEO_INPUT = os.environ.get('VIDEO_INPUT', 0)
FACE_ASPECT_RATIO = float(os.environ.get('FACE_ASPECT_RATIO', 0.08))

# ELASTIC
ELASTIC_URL = os.environ.get('ELASTIC_HOST', 'tts.korpus.io:443/elastic/')

ELASTIC_USER = os.environ.get('ELASTIC_USER', 'elastic')
ELASTIC_PASSWORD = os.environ.get('ELASTIC_PASSWORD', 'elastic')

BASIC_KEY = base64.b64encode(f"{ELASTIC_USER}:{ELASTIC_PASSWORD}".encode('ascii')).decode('utf-8')
ELASTIC_HEADERS = {'Content-Type': 'application/json', 'kbn-xsrf': 'true', 'Authorization': f'Basic {BASIC_KEY}'}
ELK_INDEX = 'video-analytics-demo'
