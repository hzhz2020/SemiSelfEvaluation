import numpy as np
import pandas as pd
import os
import pickle

def save_pickle(directory, filename, file_to_save):
#     make_dir_if_not_exists(directory)
    
    with open(os.path.join(directory, filename), 'wb') as handle:
        pickle.dump(file_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def read_npy(filepath):
    with open(filepath, 'rb') as f:
        data = np.load(f)
    return data


unlabeled_image_path = '/cluster/tufts/hugheslab/zhuang12/Echo_ClinicalManualScript_1112/ML_DATA/ViewClassifier/npy_shared_Unlabeled/Unlabeled_image.npy'
unlabeled_label_path = '/cluster/tufts/hugheslab/zhuang12/Echo_ClinicalManualScript_1112/ML_DATA/ViewClassifier/npy_shared_Unlabeled/Unlabeled_label.npy'

unlabeled_images = read_npy(unlabeled_image_path)
unlabeled_labels = read_npy(unlabeled_label_path)

print('unlabeled_images: {}, unlabeled_labels: {}'.format(unlabeled_images.shape , unlabeled_labels.shape))

unlabeled_set = {"images":unlabeled_images, "labels":unlabeled_labels}

#save the unlabeled set
all_shared_unlabeledset_path = '/cluster/tufts/hugheslab/zhuang12/SSL_Contamination/realistic-ssl-evaluation-pytorch_RE/ML_DATA/TMED2/unnormalized_HWC/echo/all_shared_unlabeledset'
# with open(os.path.join(all_shared_unlabeledset_path, 'u_train.npy'), 'wb') as f:
#     np.save(f, unlabeled_set)
        
# np.save(os.path.join(all_shared_unlabeledset_path, 'u_train.npy'), unlabeled_set) #using 'u_train.npy' as same naming convention as CIFAR10 experiment

# save_pickle(all_shared_unlabeledset_path, 'u_train.pkl', unlabeled_set)
save_pickle(all_shared_unlabeledset_path, 'u_train.npy', unlabeled_set)