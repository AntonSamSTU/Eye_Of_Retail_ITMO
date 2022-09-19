# PeopleTracking

## Getting started
```bash
conda create -n py38 python=3.8
conda activate py38
pip3 install -r requirements.txt
python3 yolo.py
```

## Project status
MVP for hackaton

## About this project  
Eye of Retail is a CV project that provides business (cafe, grocery for example)
track customer's emotional feedback.
The project has been developed by our team "post-AIrony" within product huckaton
DataProductHack (https://ai.itmo.ru/dataproducthack).

Our decesion uses yolo5n model for tracking people and deepface technology for identify human emotion. We used NoSQL DB Elastic Search for containing the data and Kibana for visualisation.

used stack - python(openCV & deepface lib.), elastic search, kibana  
presentation - https://docs.google.com/presentation/d/1DbPLfpapLt_TFfGvdyuVqu4J5v-PH0ZQD6DIcT_bTyU/edit?usp=sharing 
