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
        img_origin = self.lefteye_paste(img_origin)
        img_origin = self.righteye_paste(img_origin)
        img_origin = self.mouth_paste(img_origin)


        self.img_write(output_img_path, img_origin)

    def print_rect_at_image(self, img_gray, rect):
        for x, y, w, h in rect:
            cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def img_write(self, filename, img):
        cv2.imwrite(filename, img)
    
    def face_paste(self, back_img):
        fore_img = cv2.imread("./images/parts/face.png",  cv2.IMREAD_UNCHANGED)
        h, w = fore_img.shape[:2]
        face_after_size = (math.floor(h/5), math.floor(w/5))
        fore_img = cv2.resize(fore_img, face_after_size)
        return self.paste(back_img, fore_img, 100, 100)

    def lefteye_paste(self, back_img):
        fore_img = cv2.imread("./images/parts/lefteye.png",  cv2.IMREAD_UNCHANGED)
        h, w = fore_img.shape[:2]
        face_after_size = (math.floor(h/5), math.floor(w/5))
        fore_img = cv2.resize(fore_img, face_after_size)
        return self.paste(back_img, fore_img, 100, 100)

    def righteye_paste(self, back_img):
        fore_img = cv2.imread("./images/parts/righteye.png",  cv2.IMREAD_UNCHANGED)
        h, w = fore_img.shape[:2]
        face_after_size = (math.floor(h/5), math.floor(w/5))
        fore_img = cv2.resize(fore_img, face_after_size)
        return self.paste(back_img, fore_img, 100, 100)

    def mouth_paste(self, back_img):
        fore_img = cv2.imread("./images/parts/mouth.png",  cv2.IMREAD_UNCHANGED)
        h, w = fore_img.shape[:2]
        face_after_size = (math.floor(h/5), math.floor(w/5))
        fore_img = cv2.resize(fore_img, face_after_size)
        return self.paste(back_img, fore_img, 100, 100)

    def paste(self, back_img, fore_img, dx, dy):
        # return back_img
        # pass
        # dx = 100    # 横方向の移動距離
        # dy = 100    # 縦方向の移動距離
        h, w = fore_img.shape[:2]
        # face_after_size = (math.floor(h/5), math.floor(w/5))
        # fore_img = cv2.resize(fore_img, face_after_size)
        back_img = self.alpha_blend(back_img, fore_img, (dx, dy))
        # back_img[dy:dy+h, dx:dx+w] = fore_img
        return back_img
        
        # cv2.imshow('img',back_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    def alpha_blend(self, frame: np.array, alpha_frame: np.array, position: (int, int)):
        """
        frame に alpha_frame をアルファブレンディングで描画する。

        :param frame: ベースとなるフレーム。frame に直接、書き込まれるので、中身が変更される。
        :param alpha_frame: 重ね合わる画像フレーム。アルファチャンネルつきで読み込まれている前提。
        :param position: alpha_frame を描画する座標 (x, y)。負の値などはみ出る値も指定可能。
        :return: 戻り値はなし。frame に直接、描画する。

        usage:
        base_frame = cv2.imread("bg.jpg")
        png_image = cv2.imread("alpha.png", cv2.IMREAD_UNCHANGED)  # アルファチャンネル込みで読み込む
        alpha_blend(base_frame, png_image, (1500, 300))
        """
        # 貼り付け先座標の設定 - alpha_frame がはみ出す場合への対処つき
        x1, y1 = max(position[0], 0), max(position[1], 0)
        x2 = min(position[0] + alpha_frame.shape[1], frame.shape[1])
        y2 = min(position[1] + alpha_frame.shape[0], frame.shape[0])
        ax1, ay1 = x1 - position[0], y1 - position[1]
        ax2, ay2 = ax1 + x2 - x1, ay1 + y2 - y1

        # 合成!
        frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - alpha_frame[ay1:ay2, ax1:ax2, 3:] / 255) + \
                            alpha_frame[ay1:ay2, ax1:ax2, :3] * (alpha_frame[ay1:ay2, ax1:ax2, 3:] / 255)
        return frame
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