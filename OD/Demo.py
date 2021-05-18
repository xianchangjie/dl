from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2
import tensorflow as tf
import numpy as np
import os

# disable gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class TFObjectDetector():
    def __init__(self,
                 path_to_object_detection,
                 path_to_model_checkpoint,
                 path_to_labels,
                 model_name):
        self.model_name = model_name
        self.pipeline_config_path = path_to_object_detection
        self.pipeline_config = os.path.join(f'{self.pipeline_config_path}/{self.model_name}.config')
        self.full_config = config_util.get_configs_from_pipeline_file(self.pipeline_config)
        self.path_to_model_checkpoint = path_to_model_checkpoint
        self.path_to_labels = path_to_labels
        self.setup_model()

    # set up model for usage
    def setup_model(self):
        self.build_model()
        self.restore_checkpoint()
        self.detection_function = self.get_model_detection_function()
        self.prepare_labels()

    # build model
    def build_model(self):
        model_config = self.full_config['model']
        assert model_config is not None
        self.model = model_builder.build(model_config=model_config, is_training=False)
        return self.model

    # restore checkpoint into model
    def restore_checkpoint(self):
        assert self.model is not None
        self.checkpoint = tf.train.Checkpoint(model=self.model)
        self.checkpoint.restore(os.path.join(self.path_to_model_checkpoint, 'ckpt-0')).expect_partial()

    # get a tf.function for detection
    def get_model_detection_function(self):
        assert self.model is not None

        @tf.function
        def detection_function(image):
            image, shapes = self.model.preprocess(image)
            prediction_dict = self.model.predict(image, shapes)
            detections = self.model.postprocess(prediction_dict, shapes)
            return detections, prediction_dict, tf.reshape(shapes, [-1])

    # prepare labels
    def prepare_labels(self):
        label_map = label_map_util.load_labelmap(self.path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=label_map_util.get_max_label_map_index(
                                                                        label_map),
                                                                    use_display_name=True)
        self.category_index - label_map_util.create_category_index(categories)
        self.label_map_dict = label_map_util.get_label_map_dict(label_map,
                                                                use_display_name=True)

    # get key point tuple
    def get_keypoint_tuples(self, eval_config):
        tuple_list = []
        kp_list = eval_config.keypoint_edge
        for edge in kp_list:
            tuple_list.append((edge.start, edge.end))
        return tuple_list

    # prepare image
    def prepare_image(self, image):
        return tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)

    # do detect
    def detect(self,image,label_offset = 1):
        assert self.detection_function is not None
        # prepare image and perform prediction
        image - image.copy()
        image_tensor = self.prepare_image(image)
        detections,prediction_dict,shapes = self.detection_function(image_tensor)
