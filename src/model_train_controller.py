import torch
import wandb
from pytorch_lightning import Trainer, seed_everything

from glue_data_module import GLUEDataModule
from glue_transformer import GLUETransformer
from pathlib import Path
import os

class ModelTrainController:
    def __init__(self,
            project_name: str = "GLUETransformer",
            model_save_path: Path = "",
            train_batch_size: int = 32,
            val_batch_size: int = 32,
            epochs: int=3,
            warmup_steps: int = 0,
            weight_decay: int = 0,
            log_step_interval: int = 50,
            seed: int = 42):

        if not model_save_path:
            model_save_path = Path(os.getcwd()) / project_name

        self.project_name = project_name
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.log_step_interval = log_step_interval
        self.seed = seed

    def train_sweep(self,
                    sweep_count = 20,
                    learning_rate_values: list = [1e-4, 1e-5, 1e-6, 1e-7],
                    batch_size_distribution: str = 'q_log_uniform_values',
                    batch_size_q: int = 8,
                    batch_size_min: int = 8,
                    batch_size_max: int = 64,
                    adam_epsilon_distribution: str = 'uniform',
                    adam_epsilon_min: float = 1e-9,
                    adam_epsilon_max: float = 1e-4,
                    method: str = "bayes",
                    metric_name: str = "f1_score",
                    sweep_goal: str = "maximize"):
        seed_everything(self.seed)

        sweep_config = {
            'method': method
        }

        metric = {
            'name': metric_name,
            'goal': sweep_goal
            }

        
        parameters_dict = {
            'epochs': {
                'value': self.epochs
            },
            'warmup_steps': {
                'value': self.warmup_steps
            },
            'weight_decay': {
                'value': self.weight_decay
            },
            'learning_rate': {
                'values': learning_rate_values
            },
            'batch_size': {
                'distribution': batch_size_distribution,
                'q': batch_size_q,
                'min': batch_size_min,
                'max': batch_size_max,
            },
            'adam_epsilon': {
                'distribution': adam_epsilon_distribution,
                'min': adam_epsilon_min,
                'max': adam_epsilon_max,
            }
        }

        sweep_config['parameters'] = parameters_dict
        sweep_config['metric'] = metric

        sweep_id = wandb.sweep(sweep_config, project=self.project_name)
        wandb.agent(sweep_id, self._train, count=sweep_count)

    def _train(self, config=None):
        with wandb.init(config=config):

            config = wandb.config

            dm = GLUEDataModule(
                model_name_or_path="distilbert-base-uncased",
                task_name="mrpc",
                train_batch_size=config.batch_size,
                eval_batch_size=config.batch_size,
            )
            dm.setup("fit")
            model = GLUETransformer(
                model_name_or_path="distilbert-base-uncased",
                num_labels=dm.num_labels,
                eval_splits=dm.eval_splits,
                task_name=dm.task_name,
                train_batch_size=config.batch_size,
                eval_batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                adam_epsilon=config.adam_epsilon,
                weight_decay=config.weight_decay,
                warmup_steps=config.warmup_steps
            )

            trainer = Trainer(
                max_epochs=config.epochs,
                accelerator="auto",
                devices=1 if torch.cuda.is_available() else None,
                log_every_n_steps=self.log_step_interval
            )
            trainer.fit(model, datamodule=dm)

    def train_run(self, 
                  learning_rate: float = 1e-4,
                  adam_epsilon: float = 1e-8):
        with wandb.init(project=self.project_name):

            dm = GLUEDataModule(
                model_name_or_path="distilbert-base-uncased",
                task_name="mrpc",
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.val_batch_size,
            )
            dm.setup("fit")
            model = GLUETransformer(
                model_name_or_path="distilbert-base-uncased",
                num_labels=dm.num_labels,
                eval_splits=dm.eval_splits,
                task_name=dm.task_name,
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.val_batch_size,
                learning_rate=learning_rate,
                adam_epsilon=adam_epsilon,
                weight_decay=self.weight_decay,
                warmup_steps=self.warmup_steps
            )

            trainer = Trainer(
                max_epochs=self.epochs,
                accelerator="auto",
                devices=1 if torch.cuda.is_available() else None,
                log_every_n_steps=self.log_step_interval
            )
            
            trainer.fit(model, datamodule=dm)