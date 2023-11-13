import argparse
import wandb
from model_train_controller import ModelTrainController

def main():
    parser = argparse.ArgumentParser(description="Runs a single training run of the GLUETransformerModel")
    parser.add_argument("-lr", dest="learning_rate", type=float, default=1e-4, help="The learning rate used for training. Default is 1e-4")
    parser.add_argument("-adam_epsilon", dest="adam_epsilon", type=float, default=1e-8, help="The Adam epsilon used for training. Default is 1e-8")
    parser.add_argument("-project_name", dest="project_name", default="default_project", help="The project name used in weights and biases. Default is 'default_project'")
    parser.add_argument("-train_batch_size", dest="train_batch_size", type=int, default=32, help="The training batch size. Default is 32")
    parser.add_argument("-val_batch_size", dest="val_batch_size", type=int, default=64, help="The validation batch size. Default is 64")
    parser.add_argument("-epochs", dest="epochs", type=int, default=3, help="The number of training epochs. Default is 3")
    parser.add_argument("-warmup_steps", dest="warmup_steps", type=int, default=0, help="The number of warmup steps. Default is 0")
    parser.add_argument("-weight_decay", dest="weight_decay", type=float, default=0, help="The weight decay parameter. Default is 0")
    parser.add_argument("-log_step_interval", dest="log_step_interval", type=int, default=50, help="The logging step interval. Default is 50")
    parser.add_argument("-seed", dest="seed", type=int, default=42, help="The random seed for reproducibility. Default is 42")
    parser.add_argument("-model_save_path", dest="model_save_path", default="models/", help="The path to save the trained model. Default is 'models/'")
    parser.add_argument("-wandb_key", dest="wandb_key", required=True, help="WandB API key")

    args = parser.parse_args()

    learning_rate = args.learning_rate
    adam_epsilon = args.adam_epsilon
    project_name = args.project_name
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    epochs = args.epochs
    warmup_steps = args.warmup_steps
    weight_decay = args.weight_decay
    log_step_interval = args.log_step_interval
    seed = args.seed
    model_save_path = args.model_save_path

    print("Parameters:")
    for arg, value in vars(args).items():
        if arg != "wandb_key":
            print(f"{arg}: {value}")

    wandb.login(key=args.wandb_key)

    mct = ModelTrainController(project_name=project_name, 
                               model_save_path=model_save_path,
                               train_batch_size=train_batch_size,
                               val_batch_size=val_batch_size,
                               epochs=epochs,
                               warmup_steps=warmup_steps,
                               weight_decay=weight_decay,
                               log_step_interval=log_step_interval,
                               seed=seed)
    mct.train_run(learning_rate=learning_rate, adam_epsilon=adam_epsilon)

if __name__ == "__main__":
    main()
