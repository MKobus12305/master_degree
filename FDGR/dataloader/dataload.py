import numpy as np
import os
import albumentations as A
import torch
import pandas as pd
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def open_image(img_path):
    image = Image.open(img_path).convert('RGB')
    image = np.array(image)
    return image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class pytorch_data(Dataset):
    def __init__(self, num_points, txt_filenames=None, data_dir="../dicom_sagittal_2dimages", transform=None):
        if txt_filenames is None:
            file_names = os.listdir(data_dir)
            file_names = [i for i in file_names if i.endswith('.txt') and i.startswith('patient')]
        else:
            file_names = txt_filenames
        file_names = [i.replace('.txt','') for i in file_names]
        self.labels = [i + '.txt' for i in file_names]
        self.images = [i + '.png' for i in file_names]
        if transform is None:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(),
            ], keypoint_params=A.KeypointParams(format='xy'))
        else:
            self.transform = transform
        self.data_dir = data_dir
        self.num_points = num_points
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        curr_labels = pd.read_csv(os.path.join(self.data_dir, self.labels[idx]), sep='\t', header=None).values
        new_labels = np.zeros_like(curr_labels, dtype=np.float32)
        image = open_image(os.path.join(self.data_dir, self.images[idx]))
        transformed = self.transform(image=image, keypoints=curr_labels)
        if len(transformed['keypoints']) != 6:
            i = 0
            while(len(transformed['keypoints']) != 6):
                transformed = self.transform(image=image, keypoints=curr_labels)
                i += 1
                if i > 10:
                    raise Exception('Could not transform image')
        image = transformed['image']
        new_labels = transformed['keypoints']
        new_labels = np.array(new_labels)
        new_labels[..., 0] = new_labels[..., 0] / image.shape[0]
        new_labels[..., 1] = new_labels[..., 1] / image.shape[1]
        image = torch.tensor(np.moveaxis(image, -1, 0), dtype=torch.float32)
        new_labels = torch.tensor(new_labels, dtype=torch.float32)
        
        # Assuming valid is an additional information you want to include
        valid = True  # Example value, replace with actual logic if needed
        
        return image, new_labels[:self.num_points], valid

class predict_data(Dataset):
    def __init__(self, images: list):
        self.images = images
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize()
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        transformed = self.transform(image=image)
        image = transformed['image']
        image = torch.tensor(np.moveaxis(image, -1, 0), dtype=torch.float32)
        return image

def stratify_fill_datasets(valid_file_names, all_file_names, remaining_file_names, randomseed=23):
    if randomseed:
        random.seed(randomseed)
    train_files, val_files = [], []
    val_files = valid_file_names
    print(len(val_files) / len(all_file_names))
    valid_fraction = 0.185
    n_to_add_to_valid = round(valid_fraction * len(all_file_names)) - len(val_files)
    assert n_to_add_to_valid >= 0
    train_fraction = 1 - (n_to_add_to_valid / len(remaining_file_names))
    remaining_patient_range = sorted(list(set([int(i.split('_', 1)[0][7:]) for i in remaining_file_names])))
    random.shuffle(all_file_names)
    for i in remaining_patient_range:
        patient_files = [j for j in remaining_file_names if int(j.split('_', 1)[0][7:]) == i]
        train_files += patient_files[:int(train_fraction*len(patient_files))]
        val_files += patient_files[int(train_fraction*len(patient_files)):]
    return train_files, val_files

def setup_datasets(num_points, valid_patients=[4,12], fill_valid_to_80percent=False, transform_train=True, data_dir='../dicom_sagittal_2dimages/', val_labels_files=None):
    transformations = A.Compose([
        A.Resize(250, 250),
        A.Affine(scale=(0.9, 1.1), p=1.),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=350.0, p=0.7),
        A.GaussianBlur(blur_limit=3, p=0.5),
        A.Rotate(limit=15, p=1),
        A.Affine(shear=(-10, 10), p=1),
        A.RandomCrop(224, 224),
        A.Normalize()
    ], keypoint_params=A.KeypointParams(format='xy'))
    all_file_names = os.listdir(data_dir)
    all_file_names = [i for i in all_file_names if i.endswith('.txt') and i.startswith('patient')]
    valid_file_names = [fname for fname in all_file_names if np.array([fname.startswith("patient" + str(val_patient).rjust(3, '0')) for val_patient in valid_patients if val_patient]).any()]
    remaining_file_names = [fname for fname in all_file_names if fname not in valid_file_names]
    
    if val_labels_files is None:
        if fill_valid_to_80percent:
            train_files, val_files = stratify_fill_datasets(valid_file_names, all_file_names, remaining_file_names)
        else:
            val_files = valid_file_names
            train_files = [i for i in all_file_names if i not in val_files]
    else:
        val_files = val_labels_files
        train_files = [i for i in all_file_names if i not in val_files]

    if transform_train:
        train_ts = pytorch_data(num_points, train_files, data_dir=data_dir, transform=transformations)
    else:
        train_ts = pytorch_data(num_points, train_files, data_dir=data_dir, transform=None)
    val_ts = pytorch_data(num_points, val_files, data_dir=data_dir, transform=None)
    print("train dataset size:", len(train_ts))
    print("validation dataset size:", len(val_ts))
    print("proportion of datasets:", round(len(train_ts) / (len(val_ts) + len(train_ts)), 3), ':', round(len(val_ts) / (len(val_ts) + len(train_ts)), 3))
    print("---------------------")
    return train_ts, val_ts

def make_image_data_loader(images_files):
    dataset = predict_data(images_files)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    return data_loader