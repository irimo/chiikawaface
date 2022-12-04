# chiikawaface
顔写真からちいかわちゃんの画像を生成する（非営利ですが言われたらやめます）

# memo

docker build . -t font-convert docker run -it -p 8080:80 -v $(pwd)/:/var/www/html font-convert