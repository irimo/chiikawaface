#coding: utf-8

import cv2
# import numpy as np
# import math
# import copy
# import os
# import random

class compressor:
    def parts_recognize(self):
        print("parts_recognize() start")
        law_path = "./images/pkts/004.jpg"
        # law_path = "./images/lenna.png"
        img_origin = cv2.imread(law_path)
        output_img_path = "./out/face_lefteye_righteye.png"
        
        img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
        
        # 輪郭
        face = face_reco(img_gray)
        face_targets = face.reco()

        # 左目
        lefteye = lefteye_reco(img_gray)
        lefteye_targets = lefteye.reco()

        # # 右目
        righteye = righteye_reco(img_gray)
        righteye_targets = righteye.reco()

        self.print_rect_at_image(img_gray, face_targets)
        self.print_rect_at_image(img_gray, lefteye_targets)
        self.print_rect_at_image(img_gray, righteye_targets)

        self.img_write(output_img_path, img_gray)

    def print_rect_at_image(self, img_gray, rect):
        for x, y, w, h in rect:
            cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def img_write(self, filename, img):
        cv2.imwrite(filename, img)

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
    filename = "haarcascade_lefteye_2splits.xml"
    def __init__(self, img_gray):
        print("lefteye_reco init")
        super().__init__(img_gray, self.filename)

class righteye_reco(square_reco):
    filename = "haarcascade_righteye_2splits.xml"
    def __init__(self, img_gray):
        print("righteye_reco init")
        super().__init__(img_gray, self.filename)
        
compr = compressor()
compr.parts_recognize()