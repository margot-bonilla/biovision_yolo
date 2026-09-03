import argparse
from ultralytics import YOLO


def train_baseline(
        data_config: str,
        epochs: int = 100,
        img_size: int = 640,
        model_name: str = "yolo11n-seg.pt",
        project: str = "models",
        name: str = "baseline",
) -> None:
    """
    Train a YOLOv8 model on the provided dataset.

    Args:
        data_config (str): Path to the data configuration file.
        epochs (int): Number of training epochs.
        img_size (int): Image size for training.
        model_name (str): Pretrained model name or path.
        project (str): Project directory to save training results.
        name (str): Name of the training run.
    """
    # Initialize the YOLO model
    model = YOLO(model_name)

    # Start training
    model.train(
        data=data_config,
        epochs=epochs,
        imgsz=img_size,
        project=project,
        name=name,
    )

    print(f"Training completed. Results saved in {project}/{name}.")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Train a YOLOv8 model on the dataset.")
    arg_parser.add_argument("--data_config", type=str, required=True, help="Path to the data configuration file.")
    arg_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100).")
    arg_parser.add_argument("--img_size", type=int, default=640, help="Image size for training (default: 640).")
    arg_parser.add_argument("--model_name", type=str, default="yolo11n-seg.pt", help="Pretrained model name or path (default: 'yolo11n-seg.pt').")
    arg_parser.add_argument("--project", type=str, default="models", help="Project directory to save training results (default: 'models').")
    arg_parser.add_argument("--name", type=str, default="baseline", help="Name of the training run (default: 'baseline').")
    args = arg_parser.parse_args()

    train_baseline(
        data_config=args.data_config,
        epochs=args.epochs,
        img_size=args.img_size,
        model_name=args.model_name,
        project=args.project,
        name=args.name,
    )