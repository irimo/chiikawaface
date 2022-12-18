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

        img_origin = self.face_paste(img_origin, face_targets)
        img_origin = self.lefteye_paste(img_origin, lefteye_targets)
        img_origin = self.righteye_paste(img_origin, righteye_targets)
        # img_origin = self.mouth_paste(img_origin)

        self.print_rect_at_image(img_origin, face_targets)
        self.print_rect_at_image(img_origin, lefteye_targets)
        self.print_rect_at_image(img_origin, righteye_targets)


        self.img_write(output_img_path, img_origin)

    def print_rect_at_image(self, img_gray, rect):
        for x, y, w, h in rect:
            cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def img_write(self, filename, img):
        cv2.imwrite(filename, img)
    
    def face_paste(self, back_img, rect):
        px, py, pw, ph = rect[0]
        fore_img = cv2.imread("./images/parts/face.png",  cv2.IMREAD_UNCHANGED)
        h, w = fore_img.shape[:2]
        face_after_size = (pw, ph)
        fore_img = cv2.resize(fore_img, face_after_size)
        return self.paste(back_img, fore_img, px, py)

    def lefteye_paste(self, back_img, rect):
        px, py, pw, ph = rect[0]
        fore_img = cv2.imread("./images/parts/lefteye.png",  cv2.IMREAD_UNCHANGED)
        h, w = fore_img.shape[:2]
        face_after_size = (pw, ph)
        fore_img = cv2.resize(fore_img, face_after_size)
        return self.paste(back_img, fore_img, px, py)

    def righteye_paste(self, back_img, rect):
        px, py, pw, ph = rect[0]
        fore_img = cv2.imread("./images/parts/righteye.png",  cv2.IMREAD_UNCHANGED)
        h, w = fore_img.shape[:2]
        face_after_size = (pw, ph)
        fore_img = cv2.resize(fore_img, face_after_size)
        return self.paste(back_img, fore_img, px, py)

    def mouth_paste(self, back_img, rect):
        px, py, pw, ph = rect[0]
        fore_img = cv2.imread("./images/parts/mouth.png",  cv2.IMREAD_UNCHANGED)
        h, w = fore_img.shape[:2]
        face_after_size = (pw, ph)
        fore_img = cv2.resize(fore_img, face_after_size)
        return self.paste(back_img, fore_img, px, py)

    def paste(self, back_img, fore_img, dx, dy):
        h, w = fore_img.shape[:2]
        back_img = self.alpha_blend(back_img, fore_img, (dx, dy))
        return back_img
        
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

    def rotate(self,img, deg):
        theta = np.deg2rad(deg)
        mat = np.float32([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0]])
        img1 = cv2.warpAffine(img, mat, (w, h))

        img = cv2.hconcat([img, img1])
        del img1
        # cv2_imshow(imgs)
        return img

    def get_radian(self, x, y, x2, y2):
        radian = Math.atan2(y2 - y, x2 - x)
        return radian

    def get_central(self, rect):
        px, py, pw, ph = rect[0]

        return ((px + Math.floor(pw / 2)), (py + Math.floor(ph / 2)))

    def get_degree_from_eyes(rect1, rect2):
        x1, y1 = self.get_central(rect1)
        x2, y2 = self.get_central(rect2)
        radian = get_radian(x1, y1, x2, y2)
        return radian
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