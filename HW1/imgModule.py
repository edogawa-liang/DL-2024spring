import os
import pickle
import numpy as np
import cv2

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 【Python】 3.8.18
# 【OpenCV】 3.4.11
# 【NumPy】 1.24.4
# 【Scikit-learn】 1.3.2

#############################################

class OpenImageFile(object):
    '''
    A class for opening image files and writing the paths and labels to text files.
    '''
    def __init__(self, directory='TinyImageNet/TIN', train_file='train.txt', test_file='test.txt'):
        self.directory = directory
        self.train_file = train_file
        self.test_file = test_file

    def path_totxt(self):
        dd = os.listdir(self.directory)
        with open(self.train_file, 'w') as f1, open(self.test_file, 'w') as f2:
            for i in range(len(dd)):
                dir_path = os.path.join(self.directory, dd[i], 'images')
                if os.path.isdir(dir_path):
                    d2 = os.listdir(dir_path)

                for j in range(len(d2)-2):
                    str1 = os.path.join(self.directory, dd[i], 'images', d2[j])
                    f1.write("%s %d\n" % (str1, i))
                str1 = os.path.join(self.directory, dd[i], 'images', d2[-1])
                f2.write("%s %d\n" % (str1, i))


    def load_img(self, f, img_size=(256, 256)):
        with open(f) as file:
            lines = file.readlines()
        imgs, labels = [], []
        for line in lines:
            fn, label = line.split(' ')
            label = int(label.strip())  # Use strip() to remove newline character
            
            img = cv2.imread(fn)
            if img is not None:
                img = cv2.resize(img, img_size)  # Resize only if image is successfully loaded
                imgs.append(img)
                labels.append(label)
            # else:
                # print(f"Warning: Could not load image {fn}")
    
        return np.array(imgs), np.array(labels)

    

#################################
    
class FeatureExtractor(object):
    '''
    A class for extracting features from images.

    Parameters
    ----------
    feat_extractor: "ColorHist", "HOG", "BRIEF", "ORB"
    K: feature dimensions across different images  (default = 32)

    Methods
    -------
    get_feature(imgs): Extract features from the given images.
  
    '''
    
    def __init__(self, feat_extractor, K=32):
        self.feat_extractor = feat_extractor
        self.K = K if self.feat_extractor in ["BRIEF", "ORB"] else None

    def get_feature(self, imgs):
        '''
        Extract features from the given images.

        Parameters
        ----------
        imgs: A list of images from which to extract features.

        Returns
        -------
            A numpy array of extracted features.
        '''
        if self.feat_extractor in ["ColorHist", "HOG"]:
            return self._fixedDimen(imgs)
        elif self.feat_extractor in ["BRIEF", "ORB"]:
            return self._varyingDimen(imgs)
        else:
            raise ValueError(f"Unsupported feature extractor: {self.feat_extractor}")

    def _fixedDimen(self, imgs):
        '''
        Deal with ColorHist, HOG
        '''
        features = []
        for img in imgs:
            if self.feat_extractor == "ColorHist":
                hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
                features.append(hist)
            
            elif self.feat_extractor == "HOG":
                imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hog = cv2.HOGDescriptor((128, 128), (64, 64), (32, 32), (64, 64), 5) # winsize, blocksize, blockstride, cellsize, nbins
                descriptors = hog.compute(imggray).flatten()
                features.append(descriptors)
        return np.array(features, dtype=np.float32)

                
    def _varyingDimen(self, imgs):
        '''
        Deal with BRIEF, ORB
        '''
        descriptors_list = []
        for img in imgs:
            imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.feat_extractor == "BRIEF":
                fast = cv2.FastFeatureDetector_create()
                brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
                keypoints = fast.detect(imggray, None)
                _, descriptors = brief.compute(imggray, keypoints)
            
            elif self.feat_extractor == "ORB":
                orb = cv2.ORB_create()
                _, descriptors = orb.detectAndCompute(imggray, None)

            # 若沒有從圖像檢測到特徵點，補一個長度為self.K的零向量
            if descriptors is None or len(descriptors) == 0:
                descriptors = np.zeros((1, self.K))  #
            descriptors_list.append(descriptors)

        return self._pooling(descriptors_list)

    def _pooling(self, descriptors_list):
        all_descriptors = np.concatenate(descriptors_list, axis=0)
        kmeans = KMeans(n_clusters=self.K, random_state=0) #將每張圖的所有描述符分群
        kmeans.fit(all_descriptors)
        
        all_predictions = kmeans.predict(all_descriptors)
        
        image_features = []
        start = 0
        for descriptors in descriptors_list:
            end = start + len(descriptors)
            hist = np.bincount(all_predictions[start:end], minlength=self.K)#每張圖特徵的個數
            image_features.append(hist)
            start = end
        image_features = np.array(image_features)
    
        return image_features

######################################
    
class ModelingEvaluate(object):
    '''
    A class for training and evaluating a machine learning model.

    Parameters
    ----------
    model: The machine learning model to be trained and evaluated.
    applypca: Whether to do PCA to reduce dimension, default is False
    n_components: If PCA, the number of principle components, default is 32

    Methods
    -------
    modeling(X_train, y_train):
        Train the model on the given training data and calculate the training accuracy and F1 score.
    evaluate(X_test, y_test):
        Predict the labels of the test data using the trained model and calculate the test accuracy and F1 score.
    save_model(filepath):
        Whether to save the model 
    '''
    
    def __init__(self, model=None, applypca = False, n_components=32):
        self.model = model
        self.applypca = applypca
        self.pca = PCA(n_components=n_components) if applypca else None

    def modeling(self, X_train, y_train):
        if self.applypca:
            X_train = self.pca.fit_transform(X_train)
            print("Use PCA to reduce dimension...")

        self.model.fit(X_train, y_train)
        train_preds = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        train_f1 = f1_score(y_train, train_preds, average="macro")
        return train_acc, train_f1

    def evaluate(self, X_test, y_test):
        if self.applypca:
            X_test = self.pca.transform(X_test)

        test_preds = self.model.predict(X_test)
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average="macro")
        return test_acc, test_f1

    def save_model(self, filepath="best_model.pkl"):
        with open(filepath, 'wb') as file:
            pickle.dump(self.model, file)

