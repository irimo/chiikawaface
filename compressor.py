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
    face_ratio = 1.0
    def parts_recognize(self):
        print("parts_recognize() start")
        law_path = "./images/idol/001.jpg"
        # law_path = "./images/lenna.png"
        img_origin = cv2.imread(law_path)
        output_img_path = "./out/ck.png"
        img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
        
        # 輪郭
        face = face_reco(img_gray)
        face_targets = face.reco()

        # 目
        eyes = eyes_reco(img_gray)
        eyes_targets = eyes.reco()

        # # 左目
        # lefteye = lefteye_reco(img_gray)
        # leftsideeye_targets = lefteye.reco()

        # # # 右目
        # righteye = righteye_reco(img_gray)
        # rightsideeye_targets = righteye.reco()

        # bigger_rect = self.get_reduction_ratio(leftsideeye_targets, rightsideeye_targets)
        print(eyes_targets)
        radian = self.get_degree_from_eyes(eyes_targets[0], eyes_targets[1])
        angle = math.degrees(-radian) # 計算汚い

        # 右はどっちか、左はどっちか
        leftsideeye_targets = eyes_targets[0]
        rightsideeye_targets = eyes_targets[1]

        img_origin = self.face_paste(img_origin, face_targets[0], angle)
        img_origin = self.leftsideeye_targets(img_origin, leftsideeye_targets, angle)
        img_origin = self.rightsideeye_targets(img_origin, rightsideeye_targets, angle)

        mouth_targets = self.convert_mouth_rect(face_targets, leftsideeye_targets, rightsideeye_targets)
        img_origin = self.mouth_paste(img_origin, mouth_targets, angle)

        # self.print_rect_at_image(img_origin, face_targets[0])
        # self.print_rect_at_image(img_origin, leftsideeye_targets)
        # self.print_rect_at_image(img_origin, rightsideeye_targets)
        # self.print_rect_at_image(img_origin, eyes_targets[3])
        # self.print_rect_at_image(img_origin, eyes_targets[2])


        self.img_write(output_img_path, img_origin)

    def print_rect_at_image(self, img_gray, rect):
        x, y, w, h = rect
        cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def img_write(self, filename, img):
        cv2.imwrite(filename, img)
    
    def face_paste(self, back_img, rect, angle):
        px, py, pw, ph = rect
        a = 1.2
        pw = math.floor(pw * a)
        ph = math.floor(ph * a)
        fore_img = cv2.imread("./images/parts/face.png",  cv2.IMREAD_UNCHANGED)

        # fore_img = self.rotate(fore_img, radian, pw, ph)
        h, w = fore_img.shape[:2]
        self.face_ratio = pw / w
        face_after_size = self.get_after_size_face(w, h, pw, ph)
        # face_after_size = (pw, ph)
        fore_img = cv2.resize(fore_img, face_after_size)
        # return self.putSprite_Affine(back_img, fore_img, (px,py), radian)
        center = self.get_rotete_point(fore_img.shape[:2])

        # print(center)

        return self.putSprite_Affine(back_img, fore_img, (px,py), rect, angle=angle, center=center)
        # return self.paste(back_img, fore_img, px, py)

    def leftsideeye_targets(self, back_img, rect, angle):
        px, py, pw, ph = rect
        fore_img = cv2.imread("./images/parts/righteye.png",  cv2.IMREAD_UNCHANGED)
        h, w = fore_img.shape[:2]
        self.lefteye_ratio = pw / w
        face_after_size = self.get_after_size_eyes(w, h, pw, ph)
        fore_img = cv2.resize(fore_img, face_after_size)
        center = self.get_rotete_point(fore_img.shape[:2])
        # print(rect)
        return self.putSprite_Affine(back_img, fore_img, (px,py),rect,  angle=angle, center=center)
        # return self.paste(back_img, fore_img, px, py)

    def rightsideeye_targets(self, back_img, rect, angle):
        px, py, pw, ph = rect
        # 右側の目、という名称になっている
        fore_img = cv2.imread("./images/parts/lefteye.png",  cv2.IMREAD_UNCHANGED)
        h, w = fore_img.shape[:2]
        if (self.lefteye_ratio == 1.0):    # 初期化の値でない（汚い...ごめんなさい）
            self.lefteye_ratio = pw / w
        face_after_size = self.get_after_size_eyes(w, h, pw, ph)
        fore_img = cv2.resize(fore_img, face_after_size)
        center = self.get_rotete_point(fore_img.shape[:2])
        print("rightsideeye_targets center")
        print(center)
        print(rect)
        return self.putSprite_Affine(back_img, fore_img, (px,py), rect, angle=angle, center=center)
        # return self.paste(back_img, fore_img, px, py)
    def mouth_paste(self, back_img, rect, angle):
        px, py, pw, ph = rect
        fore_img = cv2.imread("./images/parts/mouth.png",  cv2.IMREAD_UNCHANGED)
        h, w = fore_img.shape[:2]
        if (self.lefteye_ratio == 1.0):    # 初期化の値でない（汚い...ごめんなさい）
            self.lefteye_ratio = pw / w
        face_after_size = (pw, ph)
        fore_img = cv2.resize(fore_img, face_after_size)
        center = self.get_rotete_point(fore_img.shape[:2])
        return self.putSprite_Affine(back_img, fore_img, (px,py), rect, angle=angle, center=center)
    def convert_mouth_rect(self, face_rect, lefteye_rect, righteye_rect):
        fx, fy, fw, fh = face_rect[0]
        lx, ly, lw, lh = lefteye_rect
        rx, ry, rw, rh = righteye_rect
        mc = math.floor((rx + rw + lx) / 2)
        my = math.floor(fy + (fh / 2))
        mx = math.floor(fx + (mc / 3))
        mw = math.floor(fw / 3)
        mh = math.floor(fh / 3)
        return [mx, my, mw, mh]

    # def paste(self, back_img, fore_img, dx, dy):
    #     h, w = fore_img.shape[:2]
    #     back_img = self.alpha_blend(back_img, fore_img, (dx, dy))
    #     return back_img
    # 顔が全ての人間と比べて横長なので
    def get_after_size_face(self, w, h, pw, ph):
        aw = math.floor(w * (ph / h))
        # ah = math.floor(ph / h)
        return (aw, ph)
    def get_after_size_eyes(self, w, h, pw, ph):
        aw = math.floor(w * self.face_ratio)
        ah = math.floor(h * self.face_ratio)
        return (aw, ah)
    # def get_reduction_ratio(self, rect, rect2):
    #     px, py, pw, ph = rect[0]
    #     px2, py2, pw2, ph2 = rect2[0]

    #     if (ph < ph2):
    #         return rect2
    #     return rect

    # duplicated
    def get_rotete_point(self, shape):
        cfh, cfw = shape
        center = (cfw/2, cfh/2)
        return center
        # return [float(px + pw), float(py + ph)]
        # return [(px + pw / 2), (py + ph / 2)]
        # return [(px + pw / 2), float(py + ph / 2)]
    def get_pos(self, center, x, y, shape, rect):
        h, w = shape
        px, py, pw, ph = rect
        # center = self.get_rotete_point(rect)
        # print(center)
        center_x, center_y = center
        rect_center_x = px + pw / 2
        x = rect_center_x - center_x
        rect_center_y = py + ph / 2
        y = rect_center_y - center_y
        print("center_x, center_y")
        print(center_x)
        print(center_y)
        print("rect_center_x/y")
        print(rect_center_x)
        print(rect_center_y)
        print("x,y")
        print(x)
        print(y)
        # x = center_x
        # y = center_y - h / 2
        return (math.floor(x), math.floor(y))
        
    def putSprite_Affine(self, back, front4, pos, rect_target, angle=0, center=[0,0]):
        x, y = pos
        print(angle)
        front3 = front4[:, :, :3]
        mask1 =  front4[:, :, 3]
        mask3 = 255- cv2.merge((mask1, mask1, mask1))
        bh, bw = back.shape[:2]
        ph, pw = front4.shape[:2]
        x, y = self.get_pos(center,x,y,front4.shape[:2], rect_target)
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

    def get_radian_position(self, rect):
        px, py, pw, ph = rect
        return ((px + math.floor(pw / 2)), py)
        # return ((px + math.floor(pw / 2)), (py + math.floor(ph / 2)))

    def get_degree_from_eyes(self, rect1, rect2):
        x1, y1 = self.get_radian_position(rect1)
        x2, y2 = self.get_radian_position(rect2)
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
class eyes_reco(square_reco):
    filename = "haarcascade_eye.xml"
    def __init__(self, img_gray):
        print("eyes_reco init")
        super().__init__(img_gray, self.filename)

        
compr = compressor()
compr.parts_recognize()