import os
import pandas as pd
import re
import json
from time import perf_counter as ctime
import numpy as np
import cv2
import random


def make_annotations_next_to_images():
    stime = ctime()
    ann_folder = '../../data/annotations/'  # location of json annotations (in [mm])
    dirname = '../dicom_sagittal_2dimages'  # location of 2d slices
    inlist = os.listdir(ann_folder)
    patients = list(filter(lambda s: bool(re.search(r'\d\d\d$', s)), inlist))
    for patient in patients:
        print('Processing patient', patient)
        patient_files = os.listdir(os.path.join(ann_folder, patient))
        patient_annotations = list(filter(lambda s: s.endswith('.json'), patient_files))
        patientdir = os.path.join(ann_folder, patient)
        patient_annotations_full_paths = []
        for patient_annotated_base in patient_annotations:
            annotated_patient = os.path.join(patientdir, patient_annotated_base)
            patient_annotations_full_paths.append(annotated_patient)
            with open(annotated_patient, "r") as file:
                patient_data = json.load(file)
            assert len(patient_data['markups']) == 1
            cps = patient_data['markups'][0]['controlPoints']
            if len(cps) != 6:
                print(f"Skipping {patient_annotated_base} for patient {patient} because it has {len(cps)} control points. But not 6.")
                continue
            curr_slice_str = patient_annotated_base.split('.mrk')[0]
            curr_slice = float(curr_slice_str.replace('_', '.'))
            curr_slice = round(curr_slice, 8)
            for i in cps:
                assert round(i['position'][0], 8) == curr_slice
            towrite = []
            labels = [i["label"] for i in cps]
            skip_this_annotation = False
            for lab in ["A", "B", "C", "D", "E", "F"]:
                if lab not in labels:
                    print("Skipping", patient_annotated_base, "for patient", patient, "because label", lab, "not found.")
                    skip_this_annotation = True
                    continue
            if skip_this_annotation:
                continue
            # check if no label occures twice:
            assert len(labels) == len(set(labels))
            curr_annotations = [
                {i['label']:i['position'][1:]} for i in cps
            ]
            scaling_file = patient + '_scaling.json'
            with open(os.path.join(dirname, scaling_file), "r") as file:
                scaling = json.load(file)
            for i in curr_annotations:
                for k, v in i.items():
                    i[k] = [round(v[0] / scaling[1], 3), round(v[1] / scaling[2], 3)]
            # extract only first item from each of dictionary in list:
            curr_annotations = [(list(i.keys())[0], list(i.values())[0]) for i in curr_annotations]
            curr_annotations = sorted(curr_annotations, key=lambda x: x[0])
            curr_annotations = [i[1] for i in curr_annotations]
            labels = np.array(curr_annotations)
            # TODO check if for patient 5,8,9 there is no additional shift.
            curr_slice_str = str(round(float(curr_slice), 8)).replace('.', '_')
            img = cv2.imread(os.path.join(dirname, patient + '_' + curr_slice_str + '.png'))
            if img is None:
                print('Image not found:', os.path.join(dirname, patient + '_' + curr_slice_str + '.png'))
                continue
            cx = (img.shape[1] - 1) / 2
            cy = (img.shape[0] - 1) / 2
            labels[:, 0] += cx
            labels[:, 1] =- labels[:, 1]
            labels[:, 1] += cy
            labels = np.round(labels, 3)
            pd.DataFrame(labels).to_csv(os.path.join(dirname, patient + '_' + curr_slice_str + '.txt'), sep='\t', header=False, index=False)
    print('Done in', ctime() - stime, 'seconds')


def make_train_test_split():
    dirname = 'dicom_sagittal_2dimages'
    inlist = os.listdir(dirname)
    patients = list(filter(lambda s: s.lower().startswith('patient') and s.lower().endswith(".txt"), inlist))
    patients = [i.split('.txt')[0] for i in patients]
    random.seed(42)
    random.shuffle(patients)
    train = patients[:int(len(patients) * 0.8)]
    test = patients[int(len(patients) * 0.8):]
    with open(os.path.join(dirname, 'train.txt'), 'w') as f:
        for item in train:
            f.write(f"{item}\n")
    with open(os.path.join(dirname, 'test.txt'), 'w') as f:
        for item in test:
            f.write(f"{item}\n")


if __name__ == "__main__":
    make_annotations_next_to_images()
