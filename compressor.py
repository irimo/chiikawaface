#coding: utf-8

import cv2
import numpy as np
import math
import copy
import os
import random

class compressor:
    def parts_recognize(self):
        XML_PATH_FACE = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml"
        XML_PATH_LEFTEYE = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_lefteye_2splits.xml"
        XML_PATH_RIGHTEYE = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_righteye_2splits.xml"

        print("parts_recognize() start")
        # print(cv2.__file__)
        law_path = "./images/lenna.png"
        img_origin = cv2.imread(law_path)
        # imgheight = img_origin.shape[0]
        # imgwidth = img_origin.shape[1]
        img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
        INPUT_IMG_PATH = "./images/lenna.png"
        OUTPUT_IMG_PATH = "./out/lenna.png"
        
        
        # img = cv2.imread(INPUT_IMG_PATH)
        img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
        
        # 輪郭
        classifier = cv2.CascadeClassifier(XML_PATH_FACE)
        targets = classifier.detectMultiScale(img_gray)
        for x, y, w, h in targets:
            cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 左目
        # classifier = cv2.CascadeClassifier(XML_PATH_LEFTEYE)
        # targets = classifier.detectMultiScale(img_gray)
        # for x, y, w, h in targets:
        #     cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # # 右目
        # classifier = cv2.CascadeClassifier(XML_PATH_RIGHTEYE)
        # targets = classifier.detectMultiScale(img_gray)
        # for x, y, w, h in targets:
        #     cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        cv2.imwrite(OUTPUT_IMG_PATH, img_gray)

    def face_reco(self, img_gray, classifier):
        # XML_PATH_FACE = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml"

        # classifier = cv2.CascadeClassifier(XML_PATH_FACE)
        # del XML_PATH_FACE
        targets = classifier.detectMultiScale(img_gray)
        
compr = compressor()
compr.parts_recognize()