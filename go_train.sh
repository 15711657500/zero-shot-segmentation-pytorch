backbone='resnet101'
dataset='pascal'
model='deeplabv3+'
norm='gn'
epoch=50
cs=640
ratio=0.75
bs=1
lr=0.01
momentum=0.9
decay=0.0001
gpus=0
resume="./run/bdd/deeplab-efficientnet-b7/experiment_1/checkpoint.pth.tar"
#decoder="./ckpt/herbrand.pth.tar"
#img_list="./10k.txt"
# Using tamakoji syncbn, dont use --sync-bn command

clear

python -W ignore train.py --backbone $backbone --dataset $dataset --model $model --norm $norm --batch-size $bs --momentum $momentum --weight-decay $decay --gpu-ids $gpus #--no-val --ratio $ratio #--resume $resume

#python -W ignore train.py --backbone $backbone --dataset $dataset --model $model --norm $norm --epochs $epoch --crop-size $cs --batch-size $bs --lr $lr --momentum $momentum --weight-decay $decay --gpu-ids $gpus #--decoder $decoder #--img-list $img_list --sync-bn
