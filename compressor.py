#coding: utf-8

import cv2
# import numpy as np
# import math
# import copy
# import os
# import random

class compressor:
    xml_face = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml"
    # xml_lefteye = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_lefteye_2splits.xml"
    def parts_recognize(self):
        print("parts_recognize() start")
        # xml_righteye = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_righteye_2splits.xml"

        law_path = "./images/lenna.png"
        img_origin = cv2.imread(law_path)
        output_img_path = "./out/lenna.png"
        
        img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
        
        # 輪郭
        face = square_reco(img_gray, self.xml_face)
        targets = face.reco()
        # targets = self.face_reco(img_gray, self.xml_face)
        # classifier = cv2.CascadeClassifier(self.xml_face)
        # targets = classifier.detectMultiScale(img_gray)
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


class square_reco:
    img_gray = ""
    xml_path = ""
    def __init__(self, img_gray, xml_path):
        self.img_gray = img_gray
        self.xml_path = xml_path
    def reco(self):
        classifier = cv2.CascadeClassifier(self.xml_path)
        targets = classifier.detectMultiScale(self.img_gray)
        return targets
    
compr = compressor()
compr.parts_recognize()