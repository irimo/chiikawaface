# chiikawaface
顔写真からちいかわちゃんの画像を生成する（非営利ですが言われたらやめます）

# memo

docker build . -t chiikawaface
docker run -it -v $(pwd)/:/var/www/html --rm chiikawaface bash

'''
root@64777d9b2d64:/usr/local/lib/python3.7/site-packages/cv2/data# ls
__init__.py                              haarcascade_fullbody.xml
__pycache__                              haarcascade_lefteye_2splits.xml
haarcascade_eye.xml                      haarcascade_licence_plate_rus_16stages.xml
haarcascade_eye_tree_eyeglasses.xml      haarcascade_lowerbody.xml
haarcascade_frontalcatface.xml           haarcascade_profileface.xml
haarcascade_frontalcatface_extended.xml  haarcascade_righteye_2splits.xml
haarcascade_frontalface_alt.xml          haarcascade_russian_plate_number.xml
haarcascade_frontalface_alt2.xml         haarcascade_smile.xml
haarcascade_frontalface_alt_tree.xml     haarcascade_upperbody.xml
haarcascade_frontalface_default.xml
'''

- マツコ・デラックス
 - https://naturaleight.co.jp/matsuko/