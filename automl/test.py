import autogluon.core as ag
from  autogluon.vision import ObjectDetector

url = 'https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip'
dataset_train = ObjectDetector.Dataset.from_voc(url, splits='trainval')
# or load from coco format, skip as it's too big to download
# dataset_train = ObjectDetector.Dataset.from_coco(annotation_json_file, root='/path/to/root')

ag.utils.download('https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg', path='dog.jpg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
img = mpimg.imread('dog.jpg')
imgplot = plt.imshow(img)
plt.grid()
plt.show()
