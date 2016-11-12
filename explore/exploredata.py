from utility.dumpload import DumpLoad
import numpy as np
from sklearn.preprocessing import scale



class ExploreData(object):
    def __init__(self):
    
        return
    
    def __get_data(self, filepath):
        dump_load = DumpLoad(filepath)
        data = dump_load.load()
        features = data['features']
        features = features.reshape(features.shape[0], -1)
        labels = data['labels'][:, np.newaxis]
        data = np.concatenate((features, labels), axis = 1)
        return data
    def get_train_test_statistics(self):
        train_data, test_data = self.get_train_test_data()
        _, _, train_num_bin,train_bins = self.get_data_statistics(train_data)
        _, _, test_num_bin,test_bins = self.get_data_statistics(test_data)
        return train_num_bin,train_bins, test_num_bin,test_bins
    def get_train_test_data(self):
        train_data = self.__get_data('../data/train_2.p')
        test_data = self.__get_data('../data/test_2.p')
        return train_data, test_data
    def get_data_statistics(self, data):
        num_sample = data.shape[0]
        labels = data[:,-1]
        num_class = np.unique(labels).size
        num_bin, bins = np.histogram(labels, np.arange(0, num_class+1))
        return num_sample, num_class, num_bin,bins[:-1]
    def load(self):
        
#         dump_load2 = DumpLoad('../data/test_2.p')
#         dump_load2.dump(data, 2)
#         train_data = train_data['features']
#         img_mean = np.mean(train_data, axis = 0)
#         img_std = np.std(train_data, axis = 0)
#         train_data_np = (train_data-img_mean)/img_std
#         
#         train_data_sk = scale(train_data.reshape(train_data.shape[0], -1))
#         train_data_sk = train_data_sk.reshape(train_data.shape)
#         print(train_data.shape)
        return
        
    def run(self):
        train_data, test_data = self.get_train_test_data()
        sts_train = self.get_data_statistics(train_data)
        print(sts_train)
        sts_test = self.get_data_statistics(test_data)
        print(sts_test)
        return
    

    
if __name__ == "__main__":   
    obj= ExploreData()
    obj.run()