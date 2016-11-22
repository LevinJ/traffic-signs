import sys
import os
from __builtin__ import True
import preprocess
sys.path.insert(0, os.path.abspath('..'))

from exploredata import ExploreData
import matplotlib.pyplot as plt
from utility.vis_utils import vis_grid
from utility.vis_utils import vis_grid_withlabels
from preprocess.imageaugmentation import ImageAugmentation
import numpy as np
import math 


class VisualizeImages(ExploreData):
    def __init__(self):
        ExploreData.__init__(self)

        return
    def show_images(self, data):
        
        
        features = data[:,:-1].reshape(-1, 32, 32, 3)
        labels = data[:,-1]
        total_size = labels.shape[0]
        ind= np.random.choice(total_size, size=16)
        
        vis_grid_withlabels(features[ind], self.get_label_names(labels[ind]))

        return
    def show_unique_images(self, data):
        features = data[:,:-1].reshape(-1, 32, 32, 3)
        labels = data[:,-1]
        _, ind = np.unique(labels,  return_index=True)
        vis_grid_withlabels(features[ind], self.get_label_names(labels[ind]))
        return
    def show_augmentedimages(self, data):
        features = data[:,:-1].reshape(-1, 32, 32, 3)
        labels = data[:,-1]
        total_size = labels.shape[0]
        ind= np.random.choice(total_size, size=16)
        images = features[ind]
        labels = self.get_label_names(labels[ind])
        
        aug = ImageAugmentation()
        images = aug.transform_imagebatch(images)
        
        vis_grid_withlabels(images, labels)
        return
    
    
    def run(self):
        train_data, test_data = self.get_train_test_data()
#         self.show_images(test_data)
#         self.show_augmentedimages(train_data)
        self.show_unique_images(train_data)
        plt.show()

        return
    


if __name__ == "__main__":   
    obj= VisualizeImages()
    obj.run()