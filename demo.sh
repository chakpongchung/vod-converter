# python3.6 vod_converter/main.py --from kitti --from-path datasets/mydata-kitti --to voc --to-path datasets/mydata-voc



# /data/home/microway/anaconda3/bin/python \
# vod_converter/main.py --from voc --from-path /data/home/microway/experiments/TFFRCNN/data/KITTI \
# --to voc --to-path /data/home/microway/experiments/TFFRCNN/data/KITTIVOC

py=/data/home/microway/anaconda3/bin/python

# $py \
# vod_converter/main.py --from voc --from-path ../Faster-RCNN_TF/data/VOCdevkit2007/ \
# --to voc --to-path /data/home/microway/experiments/vod-converter/temp


# $py \
# vod_converter/main.py --from rap --from-path /mnt/falcon/local/d2/users/310195626/data/pedestrian \
# --to voc --to-path /data/home/microway/experiments/vod-converter/temp

$py \
vod_converter/main.py --from rap --from-path /mnt/falcon/local/d2/users/310195626/data/pedestrian \
--to voc --to-path /mnt/falcon/local/d2/users/310195626/data/pedestrian/voc


# CAM31_2014-03-18_20140318125002-20140318125546_tarid93_frame2823_line1.png