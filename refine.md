conda create -n cls python=3.6
conda activate cls
conda install pytorch==0.4.0 -c pytorch
conda install torchvision==0.2.1
pip install tqdm
pip install torchnet
pip install opencv-python

目标：将GCN融入到global和local分支计算中

主要修改models.py  global分支是将gcn嵌入到cnn提取特征后，local是嵌入到经过from local to global处理后提取的特征后，注意这里要进行对应的池化处理（见注释），注意最后产生的global_x local_x对应的数值范围并不是[0,1]，需要将损失函数修改一下成与多类别损失函数（main.py）

main.py 详见注释

util.py 最后两个函数

voc.py 要注意读取标签信息

engine.py：计算loss的时候，model()中要加入inp标签信息（455行可尝试去掉后效果如何变化）


