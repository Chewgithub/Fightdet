FROM python:3.8.12-buster

COPY final_model_gray_op /final_model_gray_op
COPY Fightdet /Fightdet
COPY api /api
COPY requirements.txt /requirements.txt
COPY wagon-bootcamp-347403-d06e1020e122.json /credentials.json

RUN apt-get update
RUN apt-get install -y libgl1
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
