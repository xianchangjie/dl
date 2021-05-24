import pandas as pd
import matplotlib.image as mpimg
from test import ObjectDetector

class NaiveDetectionGT:
    def __init__(self, image):
        self._objects = []
        self.image = image
        img = mpimg.imread('dog.jpg')
        self.w = img.shape[1]
        self.h = img.shape[0]

    def add_object(self, name, xmin, ymin, xmax, ymax, difficult=0):
        self._objects.append({'image': self.image, 'class': name,
                              'xmin': xmin / self.w, 'ymin': ymin / self.h,
                              'xmax': xmax / self.w, 'ymax': ymax / self.h, 'difficult': difficult})

    @property
    def df(self):
        return pd.DataFrame(self._objects)

gt = NaiveDetectionGT('dog.jpg')
gt.add_object('dog', 140, 220, 300, 540)
gt.add_object('bicycle', 120, 140, 580, 420)
gt.add_object('car', 460, 70, 680, 170)
df = gt.df


dataset = ObjectDetector.Dataset(df, classes=df['class'].unique().tolist())
dataset.show_images(nsample=1, ncol=1)