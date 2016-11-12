from exploredata import ExploreData
import matplotlib.pyplot as plt



class VisualizeData(ExploreData):
    def __init__(self):
        ExploreData.__init__(self)

        return
    def disp_class_bar(self):
        train_bin_height,train_bin_ind, test_bin_height,test_bin_ind = self.get_train_test_statistics()
        _, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
        width = 0.8
        ax1.bar(train_bin_ind, train_bin_height, width=width)
        ax1.set_title('train classes')
        ax1.set_xticks(train_bin_ind  + width/2)
        ax1.set_xticklabels(map(str, train_bin_ind))
        ax2.bar(test_bin_ind, test_bin_height, width=width)
        ax2.set_title('test classes')
        ax2.set_xticks(test_bin_ind  + width/2)
        ax2.set_xticklabels(map(str, test_bin_ind))
        return
    
    def run(self):
        self.disp_class_bar()
        plt.show()

        return
    


if __name__ == "__main__":   
    obj= VisualizeData()
    obj.run()