FROM python:3.11
WORKDIR /app
COPY . /app
COPY models/fnn_model.h5 /app/models/
COPY data/ /app/data/
COPY generate_new_data.py /app/
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]