FROM python:3.9-slim
WORKDIR /app
# Copy the application and the model file into the Docker image
COPY app/ /app/
COPY model.pkl /app/
RUN pip install --default-timeout=1000 -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]