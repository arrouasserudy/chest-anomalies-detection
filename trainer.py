from time import time
import torch
from constants import device, n_classes, CHECKPOINT_PATH
from metrics import calculate_mAP
from predictor import Predictor


class Trainer:
    def __init__(
            self,
            loader,
            model=None,
            criterion=None,
            optimizer=None,
            priors=None,
            epochs=1,
            checkpoint_path="checkpoint",
            test_loader=None,
    ):
        self.loader = loader
        self.test_loader = test_loader
        self.checkpoint_path = f"{checkpoint_path}.pth.tar"
        self.checkpoint_path_before_decay = f"{checkpoint_path}_decay.pth.tar"
        self.checkpoint_losses_path = f"{checkpoint_path}_losses.txt"
        self.n_epoch = epochs
        self.start_epoch = None
        self.model = None
        self.optimizer = None
        self.criterion = criterion

        # Load a trained model if any
        if not self.load_checkpoint():
            self.start_epoch = 0
            self.model = model
            self.optimizer = optimizer

        self.predictor = Predictor(self.model, n_classes=n_classes, priors_cxcy=priors)
        self.losses = []
        self.loaded_model = {}

    def train(self, lr_decay=None, min_score=None, max_overlap=None):
        self.model.to(device)

        self.model.train()

        prev_time = time()
        loss = None
        print(f"start training at {prev_time} - {self.start_epoch}, {self.n_epoch}")
        for epoch in range(self.start_epoch, self.n_epoch):
            for i, (images, boxes, labels) in enumerate(self.loader):
                images = images.to(device)
                boxes = [b.to(device) for b in boxes]
                labels = [l.to(device) for l in labels]

                # Forward pass through the model
                predicted_locs, predicted_scores = self.model(images)

                # Compute loss
                loss = self.criterion(predicted_locs, predicted_scores, boxes, labels)

                # Backward propagation
                self.optimizer.zero_grad()
                loss.backward()

                # Update the weights
                self.optimizer.step()

                new_time = time()
                print(f"{new_time} ({new_time - prev_time}s) - batch {i}/{len(self.loader)} - loss: {loss.item()}")
                print("-------------------------------")
                if i > 0 and i % 50 == 0:
                    # Save the weights
                    self.save_checkpoint(epoch - 1, self.model, self.optimizer)
                    print("Checkpoint saved")
                prev_time = new_time

            # Compute metrics
            APs, mAP = self.get_mAp(
                predicted_locs,
                predicted_scores,
                boxes,
                labels,
                min_score,
                max_overlap,
            )
            self.losses.append((loss.item(), mAP))

            # Save checkpoint
            self.save_checkpoint(epoch, self.model, self.optimizer, loss, mAP, APs=APs)
            print("################################")
            print(f"Checkpoint saved - epoch {epoch} - loss {loss.item()} - mAp {mAP} - APs {APs}")
            print("################################")

            if lr_decay and epoch % 60 == 0:
                # Decrease the learning rate after 60 epochs
                self.save_checkpoint(
                    epoch, self.model, self.optimizer, loss, mAP, decay=True, APs=APs
                )
                lr = 0
                for param_group in self.optimizer.param_groups:
                    lr = param_group["lr"] * lr_decay
                    param_group["lr"] = lr
                print(f"Learning rate setted to {lr}")

    def save_checkpoint(
            self, epoch, model, optimizer, loss=None, mAp=None, decay=False, APs=None
    ):
        state = {"epoch": epoch, "model": model, "optimizer": optimizer}
        if decay:
            torch.save(state, f"{self.checkpoint_path}_epoch_{epoch}")
        else:
            torch.save(state, self.checkpoint_path)
        if loss:
            with open(self.checkpoint_losses_path, "a") as f:
                f.write(f"Epoch: {epoch} - loss: {loss.item()} - mAP: {mAp} - APs: {APs}\n")

    def load_checkpoint(self):
        try:
            checkpoint = torch.load(f"{CHECKPOINT_PATH}/{self.checkpoint_path}", map_location=torch.device(device))
            self.start_epoch = checkpoint["epoch"] + 1
            self.model = checkpoint["model"]
            self.optimizer = checkpoint["optimizer"]
            print(f"Loaded checkpoint from epoch {self.start_epoch}")
            return True
        except Exception as e:
            print(f"load_checkpoint failed - {e}")
            return False

    def get_mAp(
            self,
            predicted_locs,
            predicted_scores,
            true_boxes,
            true_labels,
            min_score,
            max_overlap,
    ):
        """
        Compute metrics
        """
        det_boxes, det_labels, det_scores = self.predictor.detect_objects(
            predicted_locs,
            predicted_scores,
            min_score=min_score,
            max_overlap=max_overlap,
        )
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels)
        return APs, mAP

    def evaluate(self, test_loader):
        """
        Evaluate the model with test images
        """
        if not self.test_loader:
            if test_loader:
                self.test_loader = test_loader
            else:
                print("Please add test loader")
                return
        self.model.eval()
        with open("data/evaluation.txt", "a") as f:
            f.write("Start evaluation\n")

        det_boxes = []
        det_labels = []
        det_scores = []
        true_boxes = []
        true_labels = []

        N = len(self.test_loader)
        with torch.no_grad():
            for i, (images, boxes, labels) in enumerate(self.test_loader):
                print(f"Evaluation: {i}/{N}")
                images = images.to(device)
                boxes = [b.to(device) for b in boxes]
                labels = [l.to(device) for l in labels]

                # Run the model
                predicted_locs, predicted_scores = self.model(images)
                # Predict boxes from the model output
                _det_boxes, _det_labels, _det_scores = self.predictor.detect_objects(predicted_locs, predicted_scores)

                det_boxes.extend(_det_boxes)
                det_labels.extend(_det_labels)
                det_scores.extend(_det_scores)
                true_boxes.extend(boxes)
                true_labels.extend(labels)

            # Compute metrics
            APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels)

        print("APs ---->")
        print(APs)

        print("\nmAP ---->")
        print(mAP)

        with open("data/evaluation.txt", "a") as f:
            f.write(f"{self.checkpoint_path} - mAP: {mAP}\n")
            f.write(f"APs: {APs}\n")
            f.write("\n----------------------------------------------------\n")
