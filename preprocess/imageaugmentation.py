import numpy as np
import cv2

class ImageAugmentation():
    def __init__(self):
        return
    def __transform_image(self, img,ang_range,shear_range,trans_range):
        '''
        This function transforms images to generate new images.
        The function takes in following arguments,
        1- Image
        2- ang_range: Range of angles for rotation
        3- shear_range: Range of values to apply affine transform to
        4- trans_range: Range of values to apply translations over. 
        
        A Random uniform distribution is used to generate different parameters for transformation
        
        '''
        # Rotation
    
        ang_rot = ang_range * np.random.uniform()-ang_range/2
        rows,cols,ch = img.shape    
        Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)
    
        # Translation
        tr_x = trans_range*np.random.uniform()-trans_range/2
        tr_y = trans_range*np.random.uniform()-trans_range/2
        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    
        # Shear
        pts1 = np.float32([[5,5],[20,5],[5,20]])
    
        pt1 = 5+shear_range*np.random.uniform()-shear_range/2
        pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    
        pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    
        shear_M = cv2.getAffineTransform(pts1,pts2)
            
        img = cv2.warpAffine(img,Rot_M,(cols,rows))
        img = cv2.warpAffine(img,Trans_M,(cols,rows))
        img = cv2.warpAffine(img,shear_M,(cols,rows))
        
        return img
    def transform_image(self, img):
        ang_range = 20 #rotation angle 10 degree
        shear_range = 10 #
        trans_range = 5 #both x an y shift 2.5
        return self.__transform_image(img,ang_range,shear_range,trans_range)
    def transform_imagebatch(self, batch_image):
        N = batch_image.shape[0]
        res =[]
        for n in range(N):
            temp = self.transform_image(batch_image[n])
            res.append(temp)
        return np.array(res)
        
    def run(self):
        
        return
    

if __name__ == "__main__":   
    obj= ImageAugmentation()
    obj.run()