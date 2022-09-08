import json
from elasticsearch_dsl import Document, Text, connections, Date

import settings


connections.create_connection(headers=settings.ELASTIC_HEADERS, hosts=[settings.ELASTIC_URL], use_ssl=True)


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
                act.save(index=settings.ELK_INDEX)
            except Exception as exc:
                print('Unexpected exception occurred', exc)
