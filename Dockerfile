# Dockerfile, Image, Container
FROM python:3.9-slim

ADD ./src/main.py .
ADD ./src/glue_data_module.py .
ADD ./src/glue_transformer.py .
ADD ./src/model_train_controller.py .
ADD ./requirements.txt .

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

# environment variables
ENV LEARNING_RATE=0.0001
ENV ADAM_EPSILON=0.00000001
ENV PROJECT_NAME=default_project
ENV TRAIN_BATCH_SIZE=32
ENV VAL_BATCH_SIZE=64
ENV EPOCHS=3
ENV WARMUP_STEPS=0
ENV WEIGHT_DECAY=0
ENV LOG_STEP_INTERVAL=50
ENV SEED=42
ENV MODEL_SAVE_PATH=models/

ENV WANDB_API_KEY=<your_wandb_api_key>

CMD python ./main.py -wandb_key $WANDB_API_KEY -lr $LEARNING_RATE -adam_epsilon $ADAM_EPSILON -project_name $PROJECT_NAME -train_batch_size $TRAIN_BATCH_SIZE -val_batch_size $VAL_BATCH_SIZE -epochs $EPOCHS -warmup_steps $WARMUP_STEPS -weight_decay $WEIGHT_DECAY -log_step_interval $LOG_STEP_INTERVAL -seed $SEED -model_save_path $MODEL_SAVE_PATH
