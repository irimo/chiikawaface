#coding: utf-8

import cv2
# import numpy as np
# import math
# import copy
# import os
# import random

class compressor:
    xml_face = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml"
    def parts_recognize(self):
        print("parts_recognize() start")
        xml_lefteye = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_lefteye_2splits.xml"
        xml_righteye = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_righteye_2splits.xml"

        law_path = "./images/lenna.png"
        img_origin = cv2.imread(law_path)
        # imgheight = img_origin.shape[0]
        # imgwidth = img_origin.shape[1]
        print("parts_recognize() start")
        img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
        # input_img_path = "./images/lenna.png"
        output_img_path = "./out/lenna.png"
        
        img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
        
        # 輪郭
        classifier = cv2.CascadeClassifier(self.xml_face)
        targets = classifier.detectMultiScale(img_gray)
        for x, y, w, h in targets:
            cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 左目
        # classifier = cv2.CascadeClassifier(xml_lefteye)
        # targets = classifier.detectMultiScale(img_gray)
        # for x, y, w, h in targets:
        #     cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # # 右目
        # classifier = cv2.CascadeClassifier(XML_PATH_RIGHTEYE)
        # targets = classifier.detectMultiScale(img_gray)
        # for x, y, w, h in targets:
        #     cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        cv2.imwrite(output_img_path, img_gray)

    def face_reco(self, img_gray, classifier):
        # XML_PATH_FACE = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml"

        # classifier = cv2.CascadeClassifier(XML_PATH_FACE)
        # del XML_PATH_FACE
        targets = classifier.detectMultiScale(img_gray)
        
compr = compressor()
compr.parts_recognize()