# keras_deeplearning_example
https://github.com/qiaoguan/keras_deeplearning_example

## 目录介绍：
data_augmentation_by_keras: 用keras自带的函数进行一些data augmentation操作 <br><br>

cnn_augmentation.py:  用keras训练一个简单的二分类，并对训练样本进行data augmentation（使用图片生成器ImageDataGenerator，训练时该函数会无限生成数据，直到达到规定的epoch次数为止） <br><br>

classification_using_pretrained.py:  使用VGG16已经训练好的模型(因为VGG是在imagenet上训练的，而imagenet也有包括我们要分类的物体，故可以这样做)， 我们的方法是这样的，我们将利用网络的卷积层部分，把全连接以上的部分抛掉。然后在我们的训练集和测试集上跑一遍，将得到的输出（即“bottleneck feature”，网络在全连接之前的最后一层激活的feature map）记录在两个numpy array里。然后我们基于记录下来的特征训练一个全连接网络。 然后我们把这个已经feature在我们自己定义的很小的模型上跑一遍，这样速度会快很多. <br><br>

fine-tuning_using_pretrained.py: 为了进一步提高之前的结果，我们可以试着fine-tune网络的后面几层。 Fine-tune仍然以预训练好的VGG16网络为基础，在新的数据集上重新训练一小部分权重。 <br>

* 首先载入VGG16的权重<br>
    
* 接下来在初始化好的VGG网络上添加我们预训练好的模型(classification_using_pretrained.py)<br>
    
* 最后将最后一个卷积块钱的层数冻结，然后以很低的学习率开始训练(我们只选择最后一个卷积块进行训练，是因为训练样本很少，而VGG模型层数很多，全部训练肯定不能训练好，会过拟合。  其次fine-tune时由于是在一个已经训练好的模型上进行的，故权值更新应该是一个小范围的，以免破坏预训练好的特征)<br>

训练数据库和测试数据库及VGG16训练好的权值在链接: https://pan.baidu.com/s/1skGLNOx 密码: tves<br><br>

数据下载后，把权值和数据库放在相应的目录就OK.
