#coding: utf-8

import cv2
import numpy as np
import math
import copy
import os
import random

class compressor:
    # 右目の縮尺決め打ち
    lefteye_ratio = 1.0 # 代入するので適当な値
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

        bigger_rect = self.get_reduction_ratio(lefteye_targets, righteye_targets)

        radian = self.get_degree_from_eyes(lefteye_targets, righteye_targets)

        img_origin = self.face_paste(img_origin, face_targets, radian)
        img_origin = self.lefteye_paste(img_origin, lefteye_targets, radian)
        img_origin = self.righteye_paste(img_origin, righteye_targets, radian)
        # img_origin = self.mouth_paste(img_origin)

        # self.print_rect_at_image(img_origin, face_targets)
        # self.print_rect_at_image(img_origin, lefteye_targets)
        # self.print_rect_at_image(img_origin, righteye_targets)


        self.img_write(output_img_path, img_origin)

    def print_rect_at_image(self, img_gray, rect):
        for x, y, w, h in rect:
            cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def img_write(self, filename, img):
        cv2.imwrite(filename, img)
    
    def face_paste(self, back_img, rect, radian):
        px, py, pw, ph = rect[0]
        fore_img = cv2.imread("./images/parts/face.png",  cv2.IMREAD_UNCHANGED)

        # fore_img = self.rotate(fore_img, radian, pw, ph)
        h, w = fore_img.shape[:2]
        face_after_size = self.get_after_size(w, h, pw, ph)
        # face_after_size = (pw, ph)
        fore_img = cv2.resize(fore_img, face_after_size)
        # return self.putSprite_Affine(back_img, fore_img, (px,py), radian)
        center = self.get_center(rect)
        return self.putSprite_Affine(back_img, fore_img, (px,py), angle=radian, center=center)
        # return self.paste(back_img, fore_img, px, py)

    def lefteye_paste(self, back_img, rect, radian):
        px, py, pw, ph = rect[0]
        # 左側の目、という名称になっている
        fore_img = cv2.imread("./images/parts/righteye.png",  cv2.IMREAD_UNCHANGED)
        h, w = fore_img.shape[:2]
        self.lefteye_ratio = w / pw
        face_after_size = self.get_after_size(w, h, pw, ph)
        fore_img = cv2.resize(fore_img, face_after_size)
        center = self.get_center(rect)
        return self.putSprite_Affine(back_img, fore_img, (px,py), angle=radian, center=center)
        # return self.paste(back_img, fore_img, px, py)

    def righteye_paste(self, back_img, rect, radian):
        px, py, pw, ph = rect[0]
        # 右側の目、という名称になっている
        fore_img = cv2.imread("./images/parts/lefteye.png",  cv2.IMREAD_UNCHANGED)
        h, w = fore_img.shape[:2]
        if (self.lefteye_ratio < 1.0):    # 初期化の値でない（汚い...ごめんなさい）
            self.lefteye_ratio = w / pw
        face_after_size = self.get_after_size(w, h, pw, ph)
        fore_img = cv2.resize(fore_img, face_after_size)
        center = self.get_center(rect)
        return self.putSprite_Affine(back_img, fore_img, (px,py), angle=radian, center=center)
        # return self.paste(back_img, fore_img, px, py)
    # def mouth_paste(self, back_img, rect):
    #     px, py, pw, ph = rect[0]
    #     fore_img = cv2.imread("./images/parts/mouth.png",  cv2.IMREAD_UNCHANGED)
    #     h, w = fore_img.shape[:2]
    #     face_after_size = (pw, ph)
    #     fore_img = cv2.resize(fore_img, face_after_size)
    #     return self.paste(back_img, fore_img, px, py)

    # def paste(self, back_img, fore_img, dx, dy):
    #     h, w = fore_img.shape[:2]
    #     back_img = self.alpha_blend(back_img, fore_img, (dx, dy))
    #     return back_img
    # 顔が全ての人間と比べて横長なので
    def get_after_size(self, w, h, pw, ph):
        aw = math.floor(w * (ph / h))
        # ah = math.floor(ph / h)
        return (aw, ph)
    def get_after_size_eyes(self, w, h, pw, ph):
        aw = math.floor(w * self.lefteye_ratio)
        ah = math.floor(h * self.lefteye_ratio)
        return (aw, ah)
    # duplicated
    def get_reduction_ratio(self, rect, rect2):
        px, py, pw, ph = rect[0]
        px2, py2, pw2, ph2 = rect2[0]

        if (ph < ph2):
            return rect2
        return rect

    def get_center(self, rect):
        px, py, pw, ph = rect[0]
        return [(px + pw / 2), (py + ph / 2)]
    def get_pos(self, center, aw, ah):
        # px, py, pw, ph = rect[0]
        # center = self.get_center(rect)
        # print(center)
        center_x, center_y = center
        x = center_x - aw / 2
        y = center_y - ah / 2
        return (math.floor(x), math.floor(y))
        
    # def alpha_blend(self, frame: np.array, alpha_frame: np.array, position: (int, int)):
    #     """
    #     frame に alpha_frame をアルファブレンディングで描画する。

    #     :param frame: ベースとなるフレーム。frame に直接、書き込まれるので、中身が変更される。
    #     :param alpha_frame: 重ね合わる画像フレーム。アルファチャンネルつきで読み込まれている前提。
    #     :param position: alpha_frame を描画する座標 (x, y)。負の値などはみ出る値も指定可能。
    #     :return: 戻り値はなし。frame に直接、描画する。

    #     usage:
    #     base_frame = cv2.imread("bg.jpg")
    #     png_image = cv2.imread("alpha.png", cv2.IMREAD_UNCHANGED)  # アルファチャンネル込みで読み込む
    #     alpha_blend(base_frame, png_image, (1500, 300))
    #     """
    #     # 貼り付け先座標の設定 - alpha_frame がはみ出す場合への対処つき
    #     x1, y1 = max(position[0], 0), max(position[1], 0)
    #     x2 = min(position[0] + alpha_frame.shape[1], frame.shape[1])
    #     y2 = min(position[1] + alpha_frame.shape[0], frame.shape[0])
    #     ax1, ay1 = x1 - position[0], y1 - position[1]
    #     ax2, ay2 = ax1 + x2 - x1, ay1 + y2 - y1

    #     # 合成!
    #     frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - alpha_frame[ay1:ay2, ax1:ax2, 3:] / 255) + \
    #                         alpha_frame[ay1:ay2, ax1:ax2, :3] * (alpha_frame[ay1:ay2, ax1:ax2, 3:] / 255)
    #     return frame
    # putSprite_Affine(back_img, fore_img, (x,y), radian)
    def putSprite_Affine(self, back, front4, pos, angle=0, center=[0,0]):
        # x, y = pos
        print(angle)
        front3 = front4[:, :, :3]
        mask1 =  front4[:, :, 3]
        mask3 = 255- cv2.merge((mask1, mask1, mask1))
        bh, bw = back.shape[:2]
        ph, pw = front4.shape[:2]
        x, y = self.get_pos(center,pw,ph)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        M[0][2] += x
        M[1][2] += y
        front_rot = cv2.warpAffine(front3, M, (bw,bh))
        mask_rot = cv2.warpAffine(mask3, M, (bw,bh), borderValue=(255,255,255))
        tmp = cv2.bitwise_and(back, mask_rot)
        result = cv2.bitwise_or(tmp, front_rot)
        return result

    def rotate(self,img, deg, w, h):
        theta = np.deg2rad(deg)
        mat = np.float32([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0]])
        img1 = cv2.warpAffine(img, mat, (w, h))

        # img2 = cv2.hconcat([img, img1])
        # del img1
        # del img
        # cv2_imshow(imgs)
        return img1

    def get_radian(self, x, y, x2, y2):
        radian = math.atan2(y2 - y, x2 - x)
        return radian

    def get_central(self, rect):
        px, py, pw, ph = rect[0]

        return ((px + math.floor(pw / 2)), (py + math.floor(ph / 2)))

    def get_degree_from_eyes(self, rect1, rect2):
        x1, y1 = self.get_central(rect1)
        x2, y2 = self.get_central(rect2)
        radian = self.get_radian(x1, y1, x2, y2)
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