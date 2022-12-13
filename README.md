# chiikawaface
顔写真からちいかわちゃんの画像を生成する（非営利ですが言われたらやめます）

# memo

docker build . -t chiikawaface
docker run -it -v $(pwd)/:/var/www/html --rm chiikawaface bash