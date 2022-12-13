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
        face = face_reco(img_gray)
        targets = face.reco()
        for x, y, w, h in targets:
            cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 左目
        lefteye = lefteye_reco(img_gray)
        targets = lefteye.reco()
        for x, y, w, h in targets:
            cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)


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
    xml_base_dir = "/usr/local/lib/python3.7/site-packages/cv2/data/"
    def __init__(self, img_gray, xml_filename):
        self.img_gray = img_gray
        self.xml_path = self.xml_base_dir + xml_filename
    def reco(self):
        classifier = cv2.CascadeClassifier(self.xml_path)
        targets = classifier.detectMultiScale(self.img_gray)
        return targets

class face_reco(square_reco):
    xml_face = "haarcascade_frontalface_default.xml"
    def __init__(self, img_gray):
        print("face_reco init")
        super().__init__(img_gray, self.xml_face)

class lefteye_reco(square_reco):
    xml_lefteye = "haarcascade_lefteye_2splits.xml"
    def __init__(self, img_gray):
        print("lefteye_reco init")
        super().__init__(img_gray, self.xml_lefteye)

class righteye_reco(square_reco):
    xml_lefteye = "haarcascade_lefteye_2splits.xml"
    def __init__(self, img_gray):
        print("lefteye_reco init")
        super().__init__(img_gray, self.xml_lefteye)
        
compr = compressor()
compr.parts_recognize()