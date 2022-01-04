import torch
import glob
import sys

from constants import device, CHECKPOINT_PATH, n_classes, priors_config
from dataset import get_data
from loss_function import MultiBoxLoss
from predictor import Predictor
from ssd_model import SSD
from trainer import Trainer
from utils import create_embedded_metadata, create_prior_boxes, get_models_to_evaluate
from visualization import show_losses, show_example

params = [
    {
        "overlap_threshold": 0.5,
        "neg_pos_ratio": 3,
        "lr": 0.001,
        "lr_decay": 0.1,
        "momentum": 0.9,
        "match_min_score": 0.5,
        "max_overlap": 0.5,
        "priors_set": 0,
        "without_empty": False,
    },
]


def main_train():
    for p in params:
        train_loader, test_loader = get_data(batch_size=32, without_empty=p["without_empty"])

        priors = create_prior_boxes(priors_config[p["priors_set"]])

        model = SSD(n_classes=n_classes, priors=priors_config[p["priors_set"]])
        if device == "cuda":
            model.to(device)

        print("model initiated")

        criterion = MultiBoxLoss(
            priors_cxcy=priors,
            threshold=p["overlap_threshold"],
            neg_pos_ratio=p["neg_pos_ratio"],
        ).to(device)

        print("criterion initiated")

        optimizer = torch.optim.SGD(
            model.parameters(), lr=p["lr"], momentum=p["momentum"]
        )
        trainer = Trainer(
            train_loader,
            model,
            criterion,
            optimizer,
            priors,
            epochs=130,
            checkpoint_path=f"{CHECKPOINT_PATH}_new_test",
        )

        trainer.train(
            lr_decay=p["lr_decay"],
            min_score=p["match_min_score"],
            max_overlap=p["max_overlap"],
        )


def main_eval():
    models = get_models_to_evaluate()
    print(f"Models to  evaluate: {models}")
    for i, model in enumerate(models):
        print(f"Model {i}/{len(models)} - {model}")
        train_loader, test_loader = get_data(batch_size=32)
        priors = create_prior_boxes(priors_config[1] if model == "3_More_Priors" else priors_config[0])
        trainer = Trainer(
            test_loader,
            priors=priors,
            checkpoint_path=model,
        )

        trainer.evaluate(test_loader)


def main_predict(checkpoint_path):
    _, test_loader = get_data(batch_size=32)
    priors = create_prior_boxes(priors_config[0])

    trainer = Trainer(
        test_loader,
        priors=priors,
        checkpoint_path=checkpoint_path,
    )

    return show_example(
        Predictor(trainer.model, n_classes=n_classes, priors_cxcy=priors),
        checkpoint_path=checkpoint_path
    )


if __name__ == "__main__":
    try:
        action = sys.argv[1]
    except:
        action = "predict"

    if action == "init":
        print("Create metadata")
        create_embedded_metadata()
    if action == "train":
        print("Start training process")
        main_train()
    if action == "eval":
        print("Start evaluation process")
        main_eval()
    if action == "loss":
        print("Show losses")
        losses = glob.glob("checkpoints/*losses.txt")
        show_losses(paths=losses)
    if action == "predict":
        print("Show prediction examples")
        main_predict(checkpoint_path="1_Basic_model")
