
from explore.exploredata import ExploreData
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit

class PrepareData(ExploreData):
    def __init__(self):
        ExploreData.__init__(self)
        return
    def __normalize(self):
        self.X_train = self.X_train.astype(np.float32)
        self.X_val = self.X_val.astype(np.float32)
        self.X_test = self.X_test.astype(np.float32)
        
        mean_image = np.mean(self.X_train, axis=0)
        self.X_train -= mean_image
        self.X_val -= mean_image
        self.X_test -= mean_image
        return
    def __split_dataset(self, train_data):
        # split train data into train and val
        X = train_data[:,:-1]
        y = train_data[:,-1]
        
        
        split = StratifiedShuffleSplit(y, 1, test_size=0.2, random_state=None)
        for train_index, val_index in split:
            self.X_train, self.X_val = X[train_index], X[val_index]
            self.y_train, self.y_val = y[train_index], y[val_index]
        return
    def get_train_validationset(self):
        train_data, test_data = self.get_train_test_data()
        self.__split_dataset(train_data)
        self.X_test = test_data[:,:-1]
        self.y_test = test_data[:,-1]
        self.__normalize()
        return self.X_train, self.y_train,self.X_val,self.y_val, self.X_test,self.y_test
    def run(self):
        self.get_train_validationset()
        return
    

if __name__ == "__main__":   
    obj= PrepareData()
    obj.run()