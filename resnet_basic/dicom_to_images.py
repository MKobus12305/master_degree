import os
from PIL import Image
import numpy as np
import pydicom
import json
from time import perf_counter as ctime
import png
import re


def get_thickness(dicom_file: pydicom.dataset.FileDataset) -> float:
    """Get thickness of the slices.

    Args:
        dicom_file (pydicom.dataset.FileDataset): a loaded dicom file.
        
    Returns:
        thickness (float): thickness between slices."""

    all_metadata = str(dicom_file)
    st_start = all_metadata.find('Slice Thickness') # start of line
    st_end = all_metadata[st_start:].find('\n') # end of line
    line_containing_thickness = all_metadata[st_start:st_start+st_end]
    thickness = float(re.findall(r'\d+\.\d+', line_containing_thickness).pop())
    
    return thickness


def dicom_to_images(dicom_path: str) -> np.ndarray:
    """Convert dicom file to images.
    
    Args:
        dicom_path (str): path to dicom file.
        
    Returns:
        numpy.ndarray: array of images."""
    np.seterr(divide='ignore', invalid='ignore')
    ds = pydicom.dcmread(dicom_path) # read the dicom file
    thickness = get_thickness(ds) # get thickness of the slices
    images_lst = []
    # iterate over all sagittal images in the dicom file
    for i in range(int(str(ds[(0x0028, 0x0008)])[-4:-1])):
        # Convert to float to avoid overflow or underflow losses.
        image_2d = ds.pixel_array[..., i].astype(float)
        # Rescaling grey scale between 0-255
        image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
        # Convert to uint
        image_2d_scaled = np.uint8(image_2d_scaled)
        images_lst.append(image_2d_scaled) # append to list

    for it in ds[(0x5200, 0x9230)]:
        for el in it[(0x0020, 0x9113)]:
            # get start of the sagittal plane
            sagittal_start = float(str(el)[-17:-1].split('[', 1)[1].split(',', 1)[0])
            # get start of the moving plane
            moving_start = float(str(el)[-17:-1].split(',', 2)[-1])
            break
        break
    return np.array(images_lst, dtype=np.uint8), thickness, sagittal_start, moving_start


def get_dicom_files(dicom_path):
    dicom_files = []
    for root, dirs, files in os.walk(dicom_path):
        for file in files:
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    return dicom_files


def make_images_and_spacing_files(dicoms_path, images_path, spacing_path):
    stime = ctime()
    all_dicom_files = get_dicom_files(dicoms_path)
    print("All dicoms found:")
    for dicom_path in all_dicom_files:
        print(dicom_path)
    print("--------------------")
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    for dicom_path in all_dicom_files:
        images_arr, thickness, sagittal_start, moving_start = dicom_to_images(dicom_path)
        shift = abs(sagittal_start) - moving_start
        save_patient_name = os.path.normpath(dicom_path) # get the path of the dicom file
        save_patient_name = save_patient_name.split(os.sep)[-1] # get the name of the patient
        save_patient_name = save_patient_name.replace(' ', '')
        save_patient_name = save_patient_name.lower()
        save_patient_name = save_patient_name.replace('.dcm', '')
        # add "patient" and make the number with 0 at the beginning if it is less than 3 digits
        save_patient_name = 'patient' + save_patient_name.zfill(3)
        print(f"Saving {save_patient_name}...")
        with open(spacing_path + save_patient_name + "_scaling.json", 'w') as f:
            json.dump([thickness for i in range(3)], f)
        for i in range(images_arr.shape[0]):
            print(f"{round(i / images_arr.shape[0] * 100, 2)}%", end='\r')
            if moving_start > 0 and sagittal_start < 0:
                depth = -round(moving_start - (i * thickness) + shift, 3)
            elif moving_start > 0 and sagittal_start > 0 or moving_start < 0 and sagittal_start < 0:
                depth = round(moving_start + (i * thickness) + shift, 3)
            A = images_arr[i]
            if depth == -0.0:
                depth = 0.0
            with open(images_path + save_patient_name + '_' + str(depth).replace('.', '_') + '.png', 'wb') as f:
                w = png.Writer(A.shape[1], A.shape[0], greyscale=True)
                w.write(f, A)
    print(f"Done in {ctime() - stime} seconds")


if __name__ == '__main__':
    dicoms_path = '../../data/dicoms/'
    images_path = '../dicom_sagittal_2dimages/'
    spacing_path = '../dicom_sagittal_2dimages/'
    make_images_and_spacing_files(dicoms_path, images_path, spacing_path)
