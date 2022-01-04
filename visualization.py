import random
import cv2
import torch
from matplotlib import pyplot as plt, patches
from constants import IMAGES_PATH, embedded_metadata
from dataset import transform_img
from metrics import get_losses_from_file


def plot_values_by_classname(df):
    """
    Plot an histagram of the classes by number of samples
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    x = df["class_name"].value_counts().keys()
    y = df["class_name"].value_counts().values
    ax.bar(x, y)
    ax.set_xticklabels(x, rotation=90)
    ax.set_title("Distribution of the labels")
    plt.grid()
    plt.show()


colors = [
    (1,0.5,0.5),
    (0.5,1,0.5),
    (0.5,0.5,1),
    (1,1,0.5),
    (1,0.5,1),
    (0.5,1,1),
]


def show_custom_imgs(imgs, img_ids, det_boxes, det_labels, checkpoint_path):
    """
    Show images with their true bounding boxes and the predicted ones
    """
    with torch.no_grad():
        fig, ax = plt.subplots(2, len(imgs), figsize=(20, 8))
        for row, img in enumerate(imgs):
            img_metadata = embedded_metadata.loc[
                embedded_metadata["image_id"] == img_ids[row]
                ]
            ax[0][row].imshow(img, cmap="gray")
            ax[1][row].imshow(transform_img(img, "contrast"), cmap="gray")
            print(f"len {len(det_boxes[row])}")
            for i in range(len(img_metadata)):
                case = img_metadata.iloc[i]
                p = patches.Rectangle(
                    ((case["x_min"] * 256), (case["y_min"] * 256)),
                    (case["x_max"] * 256) - (case["x_min"] * 256),
                    (case["y_max"] * 256) - (case["y_min"] * 256),
                    linewidth=1,
                    edgecolor=colors[int(case["class_id"])],
                    facecolor="none",
                )
                ax[0][row].add_patch(p)
            for i, box in enumerate(det_boxes[row]):
                color = colors[int(det_labels[row][0][i])]
                p = patches.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=0.5,
                    edgecolor=color,
                    facecolor=(*color, 0.1),
                )
                ax[1][row].add_patch(p)

            plt.setp(ax[0, row], xlabel=f"{['(a)', '(b)', '(c)', '(d)'][row]}")
            plt.setp(ax[1, row], xlabel=f"{['(a)', '(b)', '(c)', '(d)'][row]}")


        plt.setp(ax[0, 0], ylabel='Original image')
        plt.setp(ax[1, 0], ylabel=f"{checkpoint_path} prediction")
        plt.show()


def show_example(predictor, checkpoint_path):
    """
    Detect bounding boxes for example images from the test set
    """
    ids = ["f908ceca3f24d487d583b5530b127013",
           "e689cdec0469ebf2944bf49955671f2b",
           "f233f426d24061d9584932e52bfdbd49",
           "75dc0e82de7e756942ad4dcdb45c1c8d"]

    imgs = []
    img_ids = []
    det_boxes = []
    det_labels = []

    for img_id in ids:
        img_ids.append(img_id)
        img = cv2.imread(f"{IMAGES_PATH}/{img_id}.png", 0)
        imgs.append(img)
        _det_boxes, _det_labels, _ = predictor.detect(img)
        det_boxes.append(_det_boxes)
        det_labels.append(_det_labels)
    print(img_ids)
    show_custom_imgs(imgs, img_ids, det_boxes, det_labels, checkpoint_path)


def show_losses(paths):
    """
    Plot the losses graph with the different model variantes
    """
    paths = sorted(paths)
    print(paths)
    for i, path in enumerate(paths):
        label = " ".join(path[12:-10].split("_"))
        losses, _ = get_losses_from_file(path=path)
        plt.plot(losses, color=colors[i], label=label)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title("Loss by epoch")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.show()
