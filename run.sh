# data and model path
datapath=data/coco
modelpath=mcar-work-dirs

# setting
dataset='coco2014'            # 'voc2007', 'voc2012', 'coco2014'
imgsize=256                  #  256, 448    change  self.pooling = nn.MaxPool2d(8, 8)  if choose 256
basemodel='resnet101'      # 'mobilenetv2', 'resnet50', 'resnet101'
poolingstyle='avg'           # 'avg', 'gwp'
topN=4
threshold=0.5

TrainPhase=True #true
TestPhase=False #true

savepath=$modelpath/$dataset-$basemodel-$poolingstyle-$imgsize-$topN-$threshold
echo  $savepath
if [ ! -d $savepath ];then
   mkdir $savepath
fi
echo $savefold

# training
if [ $TrainPhase == True ];then
echo 'begining train....'
CUDA_VISIBLE_DEVICES=0  python -u ./src/main.py \
  --data-path $datapath  \
  --dataset-name $dataset \
  --image-size $imgsize \
  --bm $basemodel \
  --ps $poolingstyle  \
  --topN $topN \
  --threshold $threshold \
  --bs 4 \
  --sp $savepath \
#  2>&1 | tee $savepath/train-val.log
fi

# testing
if [ $TestPhase == True ];then
echo 'begining test....'
resumemodel=$savepath/model_best.pth.tar
CUDA_VISIBLE_DEVICES=0,1,2,3  python -u ./src/main.py \
  --data-path $datapath  \
  --dataset-name $dataset \
  --image-size $imgsize \
  --bm $basemodel \
  --ps $poolingstyle  \
  --topN $topN \
  --threshold $threshold \
  --bs 16 \
  --sp $savepath \
  --resume $resumemodel \
  -e \
  2>&1 | tee $savepath/test.log
fi
