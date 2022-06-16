FROM python:3.8.12

COPY final_model_gray_op /final_model_gray_op
COPY Fightdet /Fightdet
COPY api /api
COPY requirements.txt /requirements.txt
# COPY eastern-entity-347317-0aa4d779e347.json /credentials.json
COPY frontend.py /frontend.py
COPY setup.py /setup.py
# EXPOSE 8501

RUN apt-get update
RUN apt-get install -y libgl1
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ENTRYPOINT ["streamlit","run"]
CMD streamlit run frontend.py
