# sketch-dynamic
create dynamic strokes from pixel images

### Review:关于笔画生成的文献调研
- 调研方法：搜索quickdraw数据集引用文献，搜索和工作中，静态图->动态图转换，多层级笔画特征识别与生成两个方面的文章，同时关注和image caption 相关的文章。
- 时间：2018年11月17日 
- 调研结果：按文献排序，尔后总结
========================================================================================================
#### Synthesizing Programs for Images using Reinforced Adversarial Learning
Yaroslav Ganin, arXiv 2018

任务：给定手写字符，将其分解成每一笔画的特征(rendered images)。该任务在视觉领域被成为 逆图像化（Mansinghka et al.2013, Kulkarni et al.,2015a)
方法，强化学习方法，比如在原图上“在(8,13)位置放置一个小的红色箱子”，通过一些仿真工具如libmypaint生成图像，然后利用对抗网络D判定是否为原图来作为reward。
优缺点：该文章提出，已有的图像生成模型（(LeCun et al., 2015; Hinton & Salakhutdinov, 2006; Goodfellow et al., 2014; Ackley et al., 1987; Kingma & Welling, 2013; Oord et al., 2016; Kulkarni et al., 2015b; Eslami et al., 2016; Reed et al., 2017; Gregor et al., 2015)，虽然可以得到图像细节，但并不能推断出图像的结构化特征。该文章缺点在于，由于采用统一笔刷，仅能学习 MNIST Omniglot数据集，在非统一笔刷的人脸数据集 celeba上表现不好。
该文章采用生成模型和对抗网络实现了效果。这类强化学习的思路后续还有几篇文章，不予深入调研。

========================================================================================================
（这个工作系列的我们要对比）
#### Sketch-pix2seq: a Model to Generate Sketches of Multiple Categories
Yajing Chen, The Chinese University of Hong Kong, arXiv2017

任务：Photo-to-sketch，对于sketch-rnn模型生成任务来说，单类别生成效果较好，但是多类别生成效果会下降较多。所以该文主要是针对如何利用同一模型生成多个类别的数据。
方法：较sketch-rnn有两点改动，一是输入数据由原始数据转变为48*48的png图像，编码器由双向LSTM变成CNN；二是去掉VAE部分要求隐向量满足高斯分布的KL散度loss。
做了隐空间插值和t-SNE分析


#### Learning to Sketch with Shortcut Cycle Consistency
Jifei Song, Queen Mary University of London, Nips2018

任务：Photo-to-sketch，该文号称是第一个完成真实图像（224*224）到sketch图像合成工作，且完成了检索任务。将Photo-to-sketch视为一个跨领域图像-图像的翻译问题，且同一像素图可能对应的sketch图像可能区别较大，噪声较大。 P->Ep->Ds->S, S->Es->Dp->p, P->Ep->Dp->p, S->Es->Ds->S。（避免使用长的cycle方法，即不采用 P->Ep->Ds->S->Es->Dp->p，而是采用短的P->Ep->Dp->p，有利于建立domain之间的联系）
数据集：TU-Berlin，Sketchy datasets 20k~70k数据，数据难以用于生成；所以在quick draw数据集上预训练，再在 QMUL-Shoe-Chair-V2 dataset fine-tuning。

#### Universal Sketch Perceptual Grouping
Ke Li, Beijing University of Posts and Telecommunications, ECCV2018

任务：将笔画分组成有含义的几个部分。（在quickdraw上选取了一些图进行了标注，构建了SPG数据集，20000=25类*800条/类）
分组的基本思想：距离，相似性，连续性，闭环，对称。比如图像分割主要采用的是距离和相似性。
该文中，在sketch-rnn模型和损失函数的基础上，将生成的特征f128两两求差，Di,j = |fi − fj |， i, j ∈ [1, N]，将 Di,j 作为分类器输入，判断其是否属于同一组(local group loss)。将Di,j 的第i行的特征作为第i个笔画的全局分组特征，利用triplet ranking loss来保证同组相似性高于异组相似性，以解决仅用local group loss造成的冲突问题(比如 12同组，23不同组，13同组)。
评价指标：Variation of Information，不同组特征距离，Probabilistic Rand Index,组内笔画对之间索引分配的兼容性, Segmentation Covering, 机器分组结果和人类分组结果的差距

#### SKETCHSEGNET: A RNN MODEL FOR LABELING SKETCH STROKES
Xingyuan Wu, MLSP

任务：笔画层面的分割。将输入的笔画标签进行分类，正确率为90%以上。在quickdraw数据集上选取了七类物体（冰淇淋，飞机，机动车，蘑菇，咖啡杯，人脸，天使），每类标注了60个以上的轮廓图。效果比上一篇 Perceptual Grouping好一些，主要是好在连续的笔画上，比如一个车的车身方框较大，中间若有一个断点则上一种方法可能做的不太好。提升点主要在于本方法考虑了笔画连续性和类别信息。

#### Quadtree Convolutional Neural Networks
Pradeep Kumar Jayaraman, ECCV 2018

采用特殊的节点树的方式存储像素图，在此基础上，设计了卷积、池化和上采样过程，利用CNN网络实现了笔画简化 Sketch Simplification，图像尺寸是256×256，但是效果不是特别精细。

#### Multi-scale multi-class conditional generative adversarial network for handwritten character generation
Jin Liu, the journal of Supercomputing

利用GAN生成手写字。可作为对比实验之一。主要说明在生成更多尺度更多类别上，效果要好很多。

#### SketchyGAN: Towards Diverse and Realistic Sketch to Image Synthesis

利用 原始图像->HED->二值化->细化->去除小杂不减->腐蚀，然后去训练。

========================================================================================================
（这些相关性较弱，可以作为一些补充的小对比实验）
#### HUMAN VALIDATION OF COMPUTER VS HUMAN GENERATED DESIGN SKETCHES
Christian Lopez, ASME2018
介绍机器生成sketch效果同人类生成sketch效果对比，结论是测试者无法分辨哪些是人类画的，哪些是机器生成的。

#### DeepScript:_Handwriting_Generation_with_GANs_Architecture
给定句子(字符label序列)，生成手写句子(坐标序列点)

#### Inferring Semantic Layout for Hierarchical Text-to-Image Synthesis
Seunghoon Hong, CVPR2018

给定方框和文本信息，生成图像

#### Context-based Sketch Classification
Jianhui Zhang, ACM

sketch图的场景分类，场景类的还有描述、编辑、分割等等一系列文章

#### A SELF-IMPROVING COLLABORATIVE GAN FOR DECODING SKETCH RNNS
ICLR 2018, 拒稿 

考虑到生成图像偶尔比较差是因为输出序列产生了“坏点”，然而图像的多样性也可能来源于该坏点。那么我们尝试让一个模型(weak generator)产生多样性丰富的点，但是这些点质量可能较差，我们让另一个模型(strong generator)产生一些质量好但多样性不丰富的点。用两个模型(固定weak模型权重，训练strong模型权重）交互生成序列（即某个模型i时刻的输出作为另一个模型(i+1)时刻的输入，生成一段序列，希望D网络去区分这个序列和真实序列的区别。

#### End-to-End Visuomotor Learning of Drawing Sequences using Recurrent Neural Networks
Kazuma Sasaki, IJCNN

建模作图过程，即输入时刻t的图像，以及t时刻画笔的移动方向(dx, dy, p1, p2, p3)，预测下一时刻的图像以及下一时刻的画笔移动方向。

#### VectorDefense:Vectorization as a Defense to Adversarial Examples
在mnist数据集上，通过将噪声样本向量化，恢复原样本来抗攻击。

#### DragonPaint: Rule Based Bootstrapping for Small Data with an Application to Cartoon Coloring
K. Gretchen Greene arXiv 2018

基于规则的卡通图像着色任务，和人工智能大作业很像。

#### Deep Learning for Identifying Potential Conceptual Shifts for Co-creative Drawing
Pegah Karimi,
该文章尝试去寻找quickdraw中不同类型的图里，相似的图有哪些，同一类型的图里面，可以聚类分成多少种情况。

#### I Lead, You Help But Only with Enough Details:Understanding the User Experience of Co-Creation with Artificial Intelligence
CHanghong Oh, CHI2018
一个交互工具DUET DRAW, 提供五个功能，1是帮助完成绘画，2是复制一个风格类似的图画，比如再画一个菠萝，3是提供跟已有图画场景匹配的其他图画，比如给水果加个盘子；4是找到画布空的地方；5是按照用户选择的颜色着色。
