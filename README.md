# MLOps_Project_2

This repository contains a Machine Learning Operations (MLOps) project with a Dockerized Python application for training a GLUETransformerModel. The project utilizes Weights and Biases (WandB) for experiment tracking.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- [Docker](https://www.docker.com/)
- [Git](https://git-scm.com/)

## Clone the Repository

```bash
git clone https://github.com/your-username/MLOps_Project_2.git
cd MLOps_Project_2
```

## Build Docker Image

```bash
docker build -t mlops_project_2 .
```

## Run Docker Container

Replace `<your_wandb_api_key>` with your WandB API key.

```bash
docker run -e WANDB_API_KEY=<your_wandb_api_key> mlops_project_2
```

This command runs a training job with default parameters and logs the results to WandB.

The Docker container uses environment variables for configuration. Here are the available environment variables:

- `LEARNING_RATE`: Learning rate (default: 0.0001)
- `ADAM_EPSILON`: Adam epsilon (default: 0.00000001)
- `PROJECT_NAME`: Project name for WandB (default: "default_project")
- `TRAIN_BATCH_SIZE`: Training batch size (default: 32)
- `VAL_BATCH_SIZE`: Validation batch size (default: 64)
- `EPOCHS`: Number of training epochs (default: 3)
- `WARMUP_STEPS`: Number of warmup steps (default: 0)
- `WEIGHT_DECAY`: Weight decay parameter (default: 0)
- `LOG_STEP_INTERVAL`: Logging step interval (default: 50)
- `SEED`: Random seed for reproducibility (default: 42)
- `MODEL_SAVE_PATH`: Path to save the trained model (default: "models/")
- `WANDB_API_KEY`: WandB API key (required)

You can set these environment variables when running the Docker container to customize the training parameters. For example:

```bash
docker run -e WANDB_API_KEY=<your_wandb_api_key> -e LEARNING_RATE=0.001 -e EPOCHS=5 mlops_project_2
```

Feel free to customize the environment variables based on your requirements.

## Dockerfile

The `Dockerfile` is included in the repository and uses the `python:3.9-slim` base image. It installs the required dependencies specified in `requirements.txt` and sets environment variables for the training parameters.

## Main Python Script (main.py)

The `main.py` script is the entry point for running the training job. It reads the configuration from the environment variables and uses the `ModelTrainController` class from `model_train_controller.py` to perform the training run.

Happy training!