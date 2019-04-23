import rospy
from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from ssd import SSD300
from ssd_utils import BBoxUtility

REMOTENESS_UNKNOWN = -1

class TLClassifier(object):
    def __init__(self, is_site):
        #TODO load classifier

        self.is_site = is_site

        if self.is_site:
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.45
            set_session(tf.Session(config=config))

            prefix = "keras_model/"
            weights_path = prefix + "weights.01-1.26.hdf5"
            self.voc_classes = ['red','yellow','green']
            NUM_CLASSES = len(self.voc_classes) + 1
            self.input_shape=(300, 300, 3)
            self.model = SSD300(self.input_shape, num_classes=NUM_CLASSES)
            self.model.load_weights(weights_path, by_name=True)
            self.graph = tf.get_default_graph()
            self.bbox_util = BBoxUtility(NUM_CLASSES)
            self.threashold_confidence = 0.6

        else: # simulator
            prefix = "templates/" + "simulator/"
            self.template_red = cv2.imread(prefix+"red.bmp")
            self.template_yellow = cv2.imread(prefix+"yellow.bmp")
            self.template_green = cv2.imread(prefix+"green.bmp")
            self.image_sizes = [(100,75), (120,90), (160,120), (200,150), (240,180), (320,240), (400,300), (480,360)]
            self.threashold_possibility = 0.87

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        if self.is_site:
            return self.get_classification_site(image)
        else:
            return self.get_classification_simulator(image)

    def get_classification_simulator(self, image):
        for remoteness, image_size in enumerate(self.image_sizes):
            possibility_red = possibility_yellow = possibility_green = 0.0

            image_resized = cv2.resize(image,image_size)

            method = cv2.TM_CCORR_NORMED
            result_red = cv2.matchTemplate(image_resized, self.template_red, method)
            result_yellow = cv2.matchTemplate(image_resized, self.template_yellow, method)
            result_green = cv2.matchTemplate(image_resized, self.template_green, method)

            minval_red, maxval_red, minloc_red, maxloc_red = cv2.minMaxLoc(result_red)
            minval_yellow, maxval_yellow, minloc_yellow, maxloc_yellow = cv2.minMaxLoc(result_yellow)
            minval_green, maxval_green, minloc_green, maxloc_green = cv2.minMaxLoc(result_green)

            if possibility_red < maxval_red:
                possibility_red = maxval_red
            if possibility_yellow < maxval_yellow:
                possibility_yellow = maxval_yellow
            if possibility_green < maxval_green:
                possibility_green = maxval_green

            max_possibility = max(possibility_red, possibility_yellow, possibility_green)
            if max_possibility > self.threashold_possibility:
                break

        rospy.loginfo("remoteness: %d, "%remoteness +
                      "possibility_red: %.3f, "%possibility_red +
                      "possibility_yellow: %.3f, "%possibility_yellow +
                      "possibility_green: %.3f"%possibility_green)

        if max_possibility < self.threashold_possibility:
            return TrafficLight.UNKNOWN, REMOTENESS_UNKNOWN
        elif max_possibility == possibility_red:
            return TrafficLight.RED, remoteness
        elif max_possibility == possibility_yellow:
            return TrafficLight.YELLOW, remoteness
        elif max_possibility == possibility_green:
            return TrafficLight.GREEN, remoteness

        return TrafficLight.UNKNOWN, REMOTENESS_UNKNOWN

    def get_classification_site(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_300 = cv2.resize(image_rgb, self.input_shape[:2])
        inputs = preprocess_input(np.array([image_300], dtype='float32'))
        with self.graph.as_default():
            preds = self.model.predict(inputs, batch_size=1)
        result = self.bbox_util.detection_out(preds)[0]

        if len(result) == 0:
            return TrafficLight.UNKNOWN, REMOTENESS_UNKNOWN

        det_label = result[:, 0]
        det_conf  = result[:, 1]
        det_xmin  = result[:, 2]
        det_ymin  = result[:, 3]
        det_xmax  = result[:, 4]
        det_ymax  = result[:, 5]

        # Get detections with confidence higher than self.threashold_confidence.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.threashold_confidence]

        if len(top_indices) == 0:
            return TrafficLight.UNKNOWN, REMOTENESS_UNKNOWN

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        colors = [(0,0,255),(0,255,255),(0,255,0)]

        width = image_bgr.shape[1]
        height = image_bgr.shape[0]

        nearest_tl_width = 0
        nearest_tl_color = TrafficLight.UNKNOWN

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * width))
            ymin = int(round(top_ymin[i] * height))
            xmax = int(round(top_xmax[i] * width))
            ymax = int(round(top_ymax[i] * height))
            tl_width = ymax - ymin
            nearest_tl_width = max(tl_width, nearest_tl_width)
            score = top_conf[i]
            label = int(top_label_indices[i])-1
            label_name = self.voc_classes[label]

            if label == 0:
                nearest_tl_color = TrafficLight.RED
            elif label == 1:
                nearest_tl_color = TrafficLight.YELLOW
            elif label == 2:
                nearest_tl_color = TrafficLight.GREEN
            else:
                nearest_tl_color = TrafficLight.UNKNOWN

            image = cv2.rectangle(image_bgr,(xmin,ymin),(xmax,ymax),colors[label])

        return nearest_tl_color, 150/nearest_tl_width
