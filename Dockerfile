# Dockerfile, Image, Container
FROM python:3.9-slim

ADD ./src/main.py .
ADD ./src/glue_data_module.py .
ADD ./src/glue_transformer.py .
ADD ./src/model_train_controller.py .
ADD ./requirements.txt .

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

CMD ["python", "./main.py", "-lr", "0.00001", "-adam_epsilon", "0.00003", "-train_batch_size", "8", "-val_batch_size", "8", "-epochs", "3"]