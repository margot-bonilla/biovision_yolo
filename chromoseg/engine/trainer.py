import argparse
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils.loss import v8SegmentationLoss
from chromoseg.models.losses import CytogeneticsLoss


class CustomCytogeneticsLoss(v8SegmentationLoss):
    def __init__(self, model):
        super().__init__(model)
        self.bio_loss = CytogeneticsLoss(boundary_weight=0.5)

    def single_mask_loss(self, gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
        pred_mask = torch.einsum("in,nhw->ihw", pred,proto)
        pred_prob = pred_mask.sigmoid()

        return self.bio_loss(pred_prob, gt_mask)

class CytogeneticsTrainer(SegmentationTrainer):
    def set_model_attributes(self):
        super().set_model_attributes()
        self.model.criterion = CustomCytogeneticsLoss(self.model)


def train_cytogenetics(
        data_config: str, 
        epochs: int = 50, 
        img_size: int = 256,
        model_name: str = "yolo11n-seg.pt",
        project: str = "models",
        name: str = "baseline"
    ):
    model = YOLO(model_name)

    model.train(
        data=data_config,
        epochs=epochs,
        imgsz=img_size,
        trainer=CytogeneticsTrainer,
        project=project,
        name=name,
    )
    print(f"Training completed. Results saved in {project}/{name}.")



def train_baseline(
        data_config: str,
        epochs: int = 100,
        img_size: int = 256,
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
    arg_parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50).")
    arg_parser.add_argument("--img_size", type=int, default=256, help="Image size for training (default: 256).")
    arg_parser.add_argument("--model_name", type=str, default="weights/yolo11n-seg.pt", help="Pretrained model name or path (default: 'weights/yolo11n-seg.pt').")
    arg_parser.add_argument("--project", type=str, default="models", help="Project directory to save training results (default: 'models').")
    arg_parser.add_argument("--name", type=str, default="chromoseg_2class", help="Name of the training run (default: 'chromoseg_2class').")
    args = arg_parser.parse_args()

    train_cytogenetics(
        data_config=args.data_config,
        epochs=args.epochs,
        img_size=args.img_size,
        model_name=args.model_name,
        project=args.project,
        name=args.name,
    )