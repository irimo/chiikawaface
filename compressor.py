#coding: utf-8

import cv2
import numpy as np
import math
import copy
import os
import random

class compressor:
    def parts_recognize(self):
        print("parts_recognize() start")
        law_path = "./images/pkts/003.jpg"
        # law_path = "./images/lenna.png"
        img_origin = cv2.imread(law_path)
        output_img_path = "./out/ck.png"
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

        img_origin = self.face_paste(img_origin)

        self.img_write(output_img_path, img_origin)

    def print_rect_at_image(self, img_gray, rect):
        for x, y, w, h in rect:
            cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def img_write(self, filename, img):
        cv2.imwrite(filename, img)

    def face_paste(self, back_img):
        fore_img = cv2.imread("./images/parts/face.png")
        return back_img
        pass
        dx = 100    # 横方向の移動距離
        dy = 100    # 縦方向の移動距離
        h, w = fore_img.shape[:2]
        face_after_size = (math.floor(h/5), math.floor(w/5))
        fore_img = cv2.resize(fore_img, face_after_size)
        back_img[dy:dy+h, dx:dx+w] = fore_img
        return back_img
        
        # cv2.imshow('img',back_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

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