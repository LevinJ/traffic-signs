from exploredata import ExploreData
import matplotlib.pyplot as plt
from utility.vis_utils import vis_grid
import numpy as np


class VisualizeImages(ExploreData):
    def __init__(self):
        ExploreData.__init__(self)

        return
    def show_images(self):
        train_data, test_data = self.get_train_test_data()
        
        features = train_data[:,:-1].reshape(-1, 32, 32, 3)
        labels = train_data[:,-1]
        val, ind = np.unique(labels,  return_index=True)
        sampled_images = features[ind]
        res = vis_grid(sampled_images)
        plt.imshow(res)
        return
    
    def run(self):
        self.show_images()
        plt.show()

        return
    


if __name__ == "__main__":   
    obj= VisualizeImages()
    obj.run()