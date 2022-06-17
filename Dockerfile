FROM python:3.8.12

WORKDIR /app
# COPY final_model_gray_op /final_model_gray_op
# # COPY Fightdet /Fightdet
# # COPY api /api
COPY requirements.txt ./requirements.txt

# # COPY frontend.py /frontend.py
# # COPY setup.py /setup.py
COPY . /app

RUN apt-get update
RUN apt-get install -y libgl1
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


CMD streamlit run frontend.py --server.port $PORT
