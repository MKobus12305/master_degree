import torch
import numpy as np
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
from torch import nn
from models import Resi2
import logging


def rescale_predictions(predictions:np.ndarray, images:list):
    """Scales predictions from range [0,1] to image sizes

    Args:
        predictions (np.ndarray): predictions for each image
        images (list): ordered list of images where each image is a numpy array
    """
    for i, pred in enumerate(predictions):
        image = images[i]
        pred[..., 0] = pred[..., 0] * image.shape[1]
        pred[..., 1] = pred[..., 1] * image.shape[0]
    return predictions


def prepare_model(model_weights, num_points=6, device=torch.device( "cuda" if torch.cuda.is_available() else "cpu" )):
    params = {
        "num_points": num_points,
        "pretrained_encoder": False
    }
    model = Resi2(params)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()
    return model


def predict_old(model: nn.Module, eval_dataloader: DataLoader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    y_hats = []
    for batch in eval_dataloader:
        xb, _ = batch  # Assuming xb is your data and _ is labels or other information you don't need for inference
        xb = xb.to(device)
        output = model(xb)
        y_hats.append(output.cpu().detach().numpy())
    y_hats=np.concatenate(y_hats)
    return y_hats

def predict(model: nn.Module, eval_dataloader: DataLoader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    xs = []
    ys = []
    preds = []
    for batch in eval_dataloader:
        # Assuming each batch is a tuple of (inputs, labels)
        xb, yb = batch
        xb = xb.to(device)
        output = model(xb)
        
        # Append data to lists
        xs.append(xb.cpu().detach().numpy())
        ys.append(yb.cpu().detach().numpy())
        preds.append(output.cpu().detach().numpy())
    
    # Concatenate lists of arrays into single arrays
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    
    return xs, ys, preds


def predict_and_GTs(model, val_dl, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    num_points = val_dl.dataset.num_points
    preds = []
    y_val = []
    x_val = []
    for xb, yb in val_dl:
        xb=xb.to(device)
        yb=yb.to(device)
        output=model(xb)
        preds.append(output.cpu().detach().numpy())
        y_val.append(yb.cpu().detach().numpy())
        x_val.append(xb.cpu().detach().numpy())
    preds=np.concatenate(preds)
    xs = np.concatenate(x_val)
    ys = np.concatenate(y_val)
    preds = preds.reshape(-1, num_points, 2)
    ys = ys.reshape(-1, num_points, 2)
    return xs, ys, preds


def plot_predictions_on_image(img:np.ndarray, preds, axis_i=plt, title=None):
    if title is not None:
        axis_i.set_title(title, fontsize=8)
    axis_i.imshow(img)
    # annotate markers:
    annotate_keys = ["A", "B", "C", "D", "E", "F"]
    axis_i.scatter(preds[..., 0], preds[..., 1], c='r', marker='x', s=5)
    # annotate markers:
    for k in range(preds.shape[0]):
        axis_i.annotate(annotate_keys[k], (preds[k, 0], preds[k, 1]), fontsize=8)


def plot_predictions_on_images(images, preds):
    plt.figure(figsize=(10, 10))
    for i, img in enumerate(images):
        print(img.shape)
        plot_predictions_on_image(img, preds[i])
        plt.savefig(str(i) + "_img.png")


def predictions_to_dicts(predictions:np.ndarray):
    annotate_keys = ["A", "B", "C", "D", "E", "F"]
    dict_predictions = []
    for pred in predictions:
        predis = []
        for i, point in enumerate(pred):
            predi = {
                "label": annotate_keys[i],
                "position": [float(point[0]), float(point[1])]
            }
            predis.append(predi)
        dict_predictions.append({"controlPoints": predis})
    return dict_predictions


def sample_predictions(model, val_ts, data_dir="../dicom_sagittal_2dimages/"):
    val_images = val_ts.images
    val_labels = val_ts.labels
    val_dl = DataLoader(val_ts, batch_size=1, shuffle=False)
    # get random 10 indices
    np.random.seed(423)
    indices = np.random.randint(0, len(val_images), 10)
    sample_images = np.array(val_images)[indices]
    sample_labels = np.array(val_labels)[indices]
    xs,ys,preds = predict_and_GTs(model, val_dl)
    sample_preds = preds[indices]
    sample_ys = ys[indices]
    # read image im_path
    sample_images = [cv2.cvtColor(cv2.imread(data_dir + i), cv2.COLOR_BGR2RGB) for i in sample_images]
    # rescale preds
    for i, im in enumerate(sample_images):
        width = im.shape[1]
        height = im.shape[0]
        sample_preds[i][...,0] = sample_preds[i][...,0] * width
        sample_preds[i][...,1] = sample_preds[i][...,1] * height
        sample_ys = sample_ys.reshape(-1, 2, 2)  # TODO
        sample_ys[i][...,0] = sample_ys[i][...,0] * width
        sample_ys[i][...,1] = sample_ys[i][...,1] * height
        im_preds = im.copy()
        for point in sample_preds[i]:
            point = int(point[0]), int(point[1])
            cv2.circle(im_preds, point, 2, (255, 0, 0), -1)
        # plot image
        plt.figure(figsize=(10,10))
        plt.imshow(im_preds)
        plt.show()
        # plot actual points
        im_ys = im.copy()
        for point in sample_ys[i]:
            point = int(point[0]), int(point[1])
            cv2.circle(im_ys, point, 2, (0, 255, 0), -1)
        # plot image
        plt.figure(figsize=(10,10))
        plt.imshow(im_ys)
        plt.show()
        print(sample_preds[i])
        print(sample_ys[i])
        break


def evalontest(model, criterion, data_loader, device=("cuda" if torch.cuda.is_available() else "cpu")):
    if type(model) != str:
        model.eval()   # Set model to evaluate mode
    running_loss = 0.0

    # Iterate over data.
    n_inputs = 0
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        n_inputs += inputs.shape[0]
        outputs = model(inputs)
        outputs = outputs
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / n_inputs

    logging.info(f'evalontest Loss: {epoch_loss:.4f}')
