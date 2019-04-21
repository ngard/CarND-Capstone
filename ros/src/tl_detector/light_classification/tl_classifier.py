import rospy
from styx_msgs.msg import TrafficLight
import cv2

REMOTENESS_UNKNOWN = -1

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.template_red = cv2.imread("red.bmp")
        self.template_yellow = cv2.imread("yellow.bmp")
        self.template_green = cv2.imread("green.bmp")

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
        closeness_red = closeness_yellow = closeness_green = REMOTENESS_UNKNOWN

        for remoteness, image_size in enumerate(self.image_sizes):
            possibility_red = possibility_yellow = possibility_green = 0.0

            image_resized = cv2.resize(image,image_size)

            result_red = cv2.matchTemplate(image_resized, self.template_red, cv2.TM_CCORR_NORMED)
            result_yellow = cv2.matchTemplate(image_resized, self.template_yellow, cv2.TM_CCORR_NORMED)
            result_green = cv2.matchTemplate(image_resized, self.template_green, cv2.TM_CCORR_NORMED)

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
