

# import torchvision, torchvision.transforms

# import sys, os
# sys.path.insert(0,"../torchxrayvision/")
# import torchxrayvision as xrv
# import matplotlib.pyplot as plt
# import torch
# from torch.nn import functional as F
# import glob
# import numpy as np
# import skimage, skimage.filters
# import captum, captum.attr
# import torch, torch.nn
# import pickle
# import pandas as pd



# def get_data(dataset_str, masks=False, unique_patients=False,
#             transform=None, data_aug=None, merge=True, views = ["PA","AP"],
#             pathologies=None):
    
#     dataset_dir = "/home/groups/akshaysc/joecohen/"      
    
    
#     datasets = []
    
#     if "covid" in dataset_str:
#         dataset = xrv.datasets.COVID19_Dataset(
#             imgpath=dataset_dir + "/covid-chestxray-dataset/images",
#             csvpath=dataset_dir + "/covid-chestxray-dataset/metadata.csv",
#             transform=transform, data_aug=data_aug, semantic_masks=masks,
#             views=views)
#         datasets.append(dataset)

#     if "pc" in dataset_str:
#         dataset = xrv.datasets.PC_Dataset(
#             imgpath=dataset_dir + "/images-512-PC",
#             transform=transform, data_aug=data_aug,
#             unique_patients=unique_patients,
#             views=views)
#         datasets.append(dataset)

#     if "rsna" in dataset_str:
#         dataset = xrv.datasets.RSNA_Pneumonia_Dataset(
#             imgpath=dataset_dir + "/kaggle-pneumonia-jpg/stage_2_train_images_jpg",
#             transform=transform, data_aug=data_aug,
#             unique_patients=unique_patients, pathology_masks=masks,
#             views=views)
#         datasets.append(dataset)
        
#     if "nih" in dataset_str:
#         dataset = xrv.datasets.NIH_Dataset(
#             imgpath=dataset_dir + "/images-512-NIH", 
#             transform=transform, data_aug=data_aug,
#             unique_patients=unique_patients, pathology_masks=masks,
#             views=views)
#         datasets.append(dataset)
        
#     if "siim" in dataset_str: 
#         dataset = xrv.datasets.SIIM_Pneumothorax_Dataset(
#             imgpath=dataset_dir + "SIIM_TRAIN_TEST/dicom-images-train/",
#             csvpath=dataset_dir + "SIIM_TRAIN_TEST/train-rle.csv",
#             transform=transform, data_aug=data_aug,
#             unique_patients=unique_patients, pathology_masks=masks)
#         datasets.append(dataset)
        
#     if "chex" in dataset_str:
#         dataset = xrv.datasets.CheX_Dataset(
#             imgpath=dataset_dir + "/CheXpert-v1.0-small",
#             csvpath=dataset_dir + "/CheXpert-v1.0-small/train.csv",
#             transform=transform, data_aug=data_aug, 
#             unique_patients=False,
#             views=views)
#         datasets.append(dataset)
        
#     if "google" in dataset_str:
#         dataset = xrv.datasets.NIH_Google_Dataset(
#             imgpath=dataset_dir + "/images-512-NIH",
#             transform=transform, data_aug=data_aug,
#             views=views)
#         datasets.append(dataset)
        
#     if "mimic_ch" in dataset_str:
#         dataset = xrv.datasets.MIMIC_Dataset(
#             imgpath="/scratch/users/joecohen/data/MIMICCXR-2.0/files/",
#             csvpath=dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-chexpert.csv.gz",
#             metacsvpath=dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz",
#             transform=transform, data_aug=data_aug,
#             unique_patients=unique_patients,
#             views=views)
#         datasets.append(dataset)
        
#     if "openi" in dataset_str:
#         dataset = xrv.datasets.Openi_Dataset(
#             imgpath=dataset_dir + "/OpenI/images/",
#             transform=transform, data_aug=data_aug,
#             views=views)
#         datasets.append(dataset)
        
#     if "vin" in dataset_str:
#         dataset = xrv.datasets.VinBrain_Dataset(
#             imgpath=dataset_dir + "vinbigdata-chest-xray-abnormalities-detection/train",
#             csvpath=dataset_dir + "vinbigdata-chest-xray-abnormalities-detection/train.csv",
#             pathology_masks=masks, 
#             transform=transform, data_aug=data_aug,
#             views=views)
#         datasets.append(dataset)
        
#     if "objectcxr" in dataset_str:
        
#         dataset = xrv.datasets.ObjectCXR_Dataset(imgzippath=dataset_dir + "/object-CXR/train.zip",
#                             csvpath=dataset_dir + "/object-CXR/train.csv",
#                             pathology_masks=masks, 
#                             transform=transform,
#                             data_aug=data_aug)
#         datasets.append(dataset)
        
        
#     if not pathologies is None:
#         for d in datasets:
#             xrv.datasets.relabel_dataset(pathologies, d)
    
        
#     if merge:
#         newlabels = set()
#         for d in datasets:
#             newlabels = newlabels.union(d.pathologies)
# #         if "Support Devices" in newlabels:
# #             newlabels.remove("Support Devices")
#         print(list(newlabels))
#         for d in datasets:
#             xrv.datasets.relabel_dataset(list(newlabels), d)
            
            
#         dmerge = xrv.datasets.Merge_Dataset(datasets)
#         return dmerge
        
#     else:
#         return datasets
    
import torchvision
import torchxrayvision as xrv
import matplotlib.pyplot as plt
import torch
import glob
import numpy as np
import skimage
import captum.attr
import pandas as pd

# def get_data(dataset_str, masks=False, unique_patients=False,
#              transform=None, data_aug=None, merge=True, views=["PA", "AP"],
#              pathologies=None):

#     dataset_dir = "/torchxrayvision/data"
#     datasets = []

#     # Mapping dataset keys to their respective classes and initialization parameters
#     dataset_map = {
#         "covid": xrv.datasets.COVID19_Dataset,
#         "pc": xrv.datasets.PC_Dataset,
#         "rsna": xrv.datasets.RSNA_Pneumonia_Dataset,
#         "nih": xrv.datasets.NIH_Dataset,
#         "siim": xrv.datasets.SIIM_Pneumothorax_Dataset,
#         "chex": xrv.datasets.CheX_Dataset,
#         "google": xrv.datasets.NIH_Google_Dataset,
#         "mimic_ch": xrv.datasets.MIMIC_Dataset,
#         "openi": xrv.datasets.Openi_Dataset,
#         "vin": xrv.datasets.VinBrain_Dataset,
#         "objectcxr": xrv.datasets.ObjectCXR_Dataset,
#         "custom": xrv.datasets.PneumoniaXRayDataset # This maps our new dataset class in datasets.py
#     }

#     # Initialize datasets based on the keys in dataset_str
#     for key, cls in dataset_map.items():
#         if key in dataset_str:
#             if key == "custom":
#                 # Specific arguments for the PneumoniaXRayDataset can be set here
#                 dataset = cls(img_dir=dataset_dir + "/content/drive/MyDrive/Capstone-Veytel/torchxrayvision/torchxrayvision/data/merged_labels.csv",
#                               csv_file=dataset_dir + "/content/drive/MyDrive/Capstone-Veytel/torchxrayvision/torchxrayvision/data/Sampled_Custom/",
#                               transform=transform, data_aug=data_aug,
#                               unique_patients=unique_patients, views=views)
#             else:
#                 dataset = cls(imgpath=dataset_dir + f"images-{key}",
#                               csvpath=dataset_dir + f"metadata-{key}.csv",
#                               transform=transform, data_aug=data_aug,
#                               semantic_masks=masks, views=views)
#             datasets.append(dataset)

#     # Relabel datasets if specific pathologies are requested
#     if pathologies is not None:
#         for d in datasets:
#             xrv.datasets.relabel_dataset(pathologies, d)

#     if merge:
#         dmerge = xrv.datasets.Merge_Dataset(datasets)
#         return dmerge

#     else:
#         return datasets
import torchvision
import torchxrayvision as xrv
import torch

def get_data(dataset_str, masks=False, unique_patients=False,
             transform=None, data_aug=None, merge=True, views=["PA", "AP"],
             pathologies=None):
    
    dataset_dir = "/content/drive/MyDrive/Capstone-Veytel/torchxrayvision/torchxrayvision/data"
    datasets = []

    # Pathology mapping to your classes
    pathology_map = {
        "COVID-Pneumonia": 2,    # COVID pneumonia
        "Non-COVID-Pneumonia": 1,   # non-COVID pneumonia
        "Non-Pneumonia": 0   # non-pneumonia
    }
    
    
    # Define which datasets to initialize
    dataset_map = {
        "custom": xrv.datasets.PneumoniaXRayDataset
        # Add other datasets if needed
    }

    # Initialize datasets based on the keys in dataset_str
    for key, cls in dataset_map.items():
        if key in dataset_str:
            dataset = cls(img_dir=dataset_dir + "/Sampled_Custom/",
                          csv_file=dataset_dir + "/merged_labels.csv",
                          transform=transform, data_aug=data_aug,
                          unique_patients=unique_patients, views=views)
            datasets.append(dataset)

    # Relabel datasets to use standard classes
    if pathologies:
        for d in datasets:
            relabel_dataset(d, pathology_map)

    if merge:
        dmerge = xrv.datasets.Merge_Dataset(datasets)
        return dmerge
    else:
        return datasets

def relabel_dataset(dataset, pathology_map):
    new_labels = []
    for label in dataset.labels:
        mapped_label = [pathology_map.get(l, -1) for l in label]  # Use -1 for undefined pathologies
        new_labels.append(mapped_label)
    dataset.labels = torch.tensor(new_labels)
  