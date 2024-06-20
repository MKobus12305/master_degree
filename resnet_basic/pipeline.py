from pointmarker.inference import predict, prepare_model, rescale_predictions, predictions_to_dicts
from pointmarker.dataset import predict_data, open_image
import torch
from torch.utils.data import DataLoader
import json


if __name__ == "__main__":
    # read image
    img_paths = [
        "../dicom_sagittal_2dimages/patient005_6_0.png",
        "../dicom_sagittal_2dimages/patient005_6_2.png"
    ]
    slice_numbers = [i.split('/')[-1].split('_', 1)[1].split(".")[0] for i in img_paths]

    images = []
    for img_path in img_paths:
        img = open_image(img_path)
        images.append(img)
    device = ( "cuda" if torch.cuda.is_available() else "cpu" )

    images_dataset = predict_data(images)
    images_dataloader = DataLoader(images_dataset, batch_size=1, shuffle=False)
    # prepare model
    model = prepare_model("pdector_weights.pt", device=device)
    # predict
    preds = predict(model, images_dataloader, device=device)
    preds = rescale_predictions(preds, images)
    # predictions are in x, y format(x from left, y from up)
    dict_preds = predictions_to_dicts(preds)
    for prediction, img_path in zip(dict_preds, img_paths):
        # save prediction as json
        save_path = img_path.split('/')[-1].split(".")[0] + ".json"
        with open(save_path, "w") as f:
            json.dump(prediction, f, indent=4)
