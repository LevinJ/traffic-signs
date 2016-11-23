import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class VisuallieClassification(object):
    def __init__(self):

        return
    def plt_bar_2(self):
        N = 5
        menMeans = (0.8, 0.1, 0.05,0.025, 0.025)
        menStd = (2, 3, 4, 1, 2)

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35       # the width of the bars
        
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, menMeans, width, color='g')
        
        

        
        # add some text for labels, title and axes ticks
        ax.set_ylabel('Scores')
        ax.set_title('Scores by group and gender')
        ax.set_xticks(ind)
        ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
        


        return
    def plt_bar(self, class_score, class_name, ax):
        # Example data
    
        y_pos = np.arange(len(class_score), 0, -1)
        for i in range(len(class_name)):
            class_name[i] = class_name[i] + '(' + str(class_score[i]) + ')'
        class_name = class_name[::-1]
        #just to make sure socre with very little value can be displayed 
        class_score  = np.array(class_score) + 0.02

        ax.barh(y_pos, class_score, align='center', alpha=0.4)
        for y in y_pos:
            ax.text(0, y, class_name[y-1], fontsize=15)
        ax.set_xlabel('probability')
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
        ax.set_title('Top 5 prediction')
        return
    def plt_classification(self, img_title, img, class_score, class_name):
        fig, axarr = plt.subplots(nrows=1, ncols=2,squeeze=False, figsize=(12, 4))
        fig.set
        axarr[0,0].imshow(img)
        axarr[0,0].set_title(img_title)
        self.plt_bar(class_score, class_name, axarr[0,1])
        plt.show()
        
        return
    
  
    
    def run(self):
        class_score = [0.8, 0.1, 0.05,0.025, 0.025]
        class_name= ['A', 'B','C','D','E']
        img = mpimg.imread('../data/01.jpg')
        img_title = 'Image #12\nTrue Lable: Stop'
        
        self.plt_classification(img_title, img, class_score, class_name)
        plt.show()

        return
    


if __name__ == "__main__":   
    obj= VisuallieClassification()
    obj.run()