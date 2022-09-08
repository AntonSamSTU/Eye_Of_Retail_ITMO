import os
import json
import base64
from elasticsearch_dsl import Document, Text, connections, Date


ELASTIC_USER = os.environ.get('ELASTIC_USER', 'elastic')
ELASTIC_PASSWORD = os.environ.get('ELASTIC_PASSWORD', 'elastic')

BASIC_KEY = base64.b64encode(f"{ELASTIC_USER}:{ELASTIC_PASSWORD}".encode('ascii')).decode('utf-8')
ELASTIC_HOST = 'tts.korpus.io/elastic/'
ELASTIC_PORT = '80'
ELASTIC_HEADERS = {'Content-Type': 'application/json', 'kbn-xsrf': 'true', 'Authorization': f'Basic {BASIC_KEY}'}
ELK_INDEX = 'video-analytics-demo'

connections.create_connection(headers=ELASTIC_HEADERS, hosts=['tts.korpus.io:443/elastic/'], use_ssl=True)


class Visit(Document):
    # age = Long()
    # gender = Text()
    creation_date = Date()
    last_update = Date()
    emotion = Text()
    shop_name = Text()


if __name__ == '__main__':
    with open('test.json', 'r') as js:
        for line in js.readlines():
            try:
                j = json.loads(line)
                j.pop('time', '')
                act = Visit(**j)
                act.save(index=ELK_INDEX)
            except Exception as exc:
                print('Unexpected exception occurred', exc)
