# GNMT模型训练

## 环境配置

本项目基于的python库为`tensorflow`，它是一个采用数据流图的开源库，被广泛应用于各类机器学习算法的实现。

在训练本模型前，要首先完成python相关库的安装和配置，所需要的环境大致如下：

- tensorflow-gpu 1.12.0
- scipy 1.13.0
- cudatoolkit 9.2.0
- cudnn 7.6.0
- numpy 1.16.4

由于需要的库较多，并且库之间有版本依赖关系，建议使用`anaconda`对环境进行配置管理。

另外，tensorflow同时支持CPU和GPU两种模式的运行，只是需要下载不同的库，CPU的计算速度相比于GPU慢了许多，训练时间会成倍增加，对于我们要训练的翻译模型来讲CPU的速度远远不够，所以建议在服务器端配置GPU的运行环境。

接下来我们就对以上提到的环境配置阐述具体步骤。

### 使用Anaconda配置GPU环境(Linux)

#### Anaconda安装与使用

首先，下载anaconda，[这是下载链接](https://www.anaconda.com/distribution/)。

![1575123459980](https://github.com/Lor-na/Train-Doc-For-NMT/blob/master/images/1575123459980.png?raw=true)

我们下载distribution版本即可，在界面中根据自己的操作系统选择相应的版本，本模型语言基于python3，所以选择`Python 3.7 Version`。

下载安装完成后，在命令行输入命令`conda activate`，就会看到在命令行前面出现`(base)`字样，证明你已经启动了conda环境。

此时建议创建一个新环境并命名，将所有与本项目有关的库安装到该环境中，借助conda便于管理，以后有其他项目需要还可以建立新的环境，建立环境有关的命令可以参考[如下链接](https://blog.csdn.net/wdx1993/article/details/83660717)。（该步可以跳过）

#### 选择tensorflow版本

在建立好环境后，我们要对tensorflow等库进行安装。首先交互哟交互要做的是选择一个合适的版本，tensorflow运行于GPU的python库叫做`tensorflow-gpu`，该库需要`Cudnn`和`Cuda`这两个与GPU做交互会使用到的库。这两个库的版本与GPU的型号息息相关，所以一般在服务器端这两个库是提前部署好的，也就确定了其版本。如果tensorflow版本相较于这两个库版本过高则会出现GPU不能兼容的情况，所以我们要做的是在可以兼容的情况下选择尽量高的tensorflow版本进行安装。

以下是用于确定相应版本的网址链接：

+ [查看服务器的CUDA和CUDNN版本](https://blog.csdn.net/u011394059/article/details/78455252)
+ [tensorflow和CUDA/CUDNN的版本对应关系，选择一个当前CUDA/CUDNN版本支持下的最高版本即可](https://blog.csdn.net/qq_27825451/article/details/89082978)

确定版本后我们就可以准备安装了。

#### 安装tensorflow-gpu

这里有两种方式进行安装，一种是使用conda，一种是使用pip。

conda安装非常简单，只需要键入命令`conda install tensorflow-gpu=xx.xx.xx`，即可下载对应版本的tensoflow以及所有其所需要的依赖库。但这种方法的弊端是其下载的依赖库中CUDA或CUDNN与服务器端所需要的版本不一致（即使你上一步已经确定了他们的版本是对应的），如果是这种情况就需要使用pip安装。

pip安装使用命令`pip install tensorflow-gpu=xx.xx.xx`，与conda几乎一致，只是需要自己在分别利用同样的命令格式对`cuda-toolkit`和`cudnn`进行安装，情况因人而异在此不一一列举。在之后的运行中说明缺少什么库也可以利用这种方法把库补安装在环境中。

安装好tensorflow后开始尝试训练模型。

## 项目地址

[这是该模型的Github地址](https://github.com/tensorflow/nmt)

在tensorflow提供的github项目中给出了十分详细的模型说明文档，对翻译模型的架构和训练方法都给出了简单的介绍，本文档的训练过程即按照其步骤进行。

## 项目下载

```bash
git clone https://github.com/tensorflow/nmt/
```

执行上述指令即可克隆整个项目至当前目录下。

## 数据集下载

进入刚才克隆的项目目录下，运行该命令即可将英语-越南语的数据集下载至tmp目录下的nmt_data文件夹中。
```bash
nmt/scripts/download_iwslt15.sh /tmp/nmt_data
```

在脚本后的超参数设定的是数据集下载地址。使用文档中的地址好处为在之后的训练过程中不需要调整参数即可完成大部分训练。该文档中的数据和模型储存位置都是tmp文件夹中，应是为Linux用户使用方便而设计。如果想要自己设定地址可以使用相对地址的方法进行设定，如:
```bash
nmt/scripts/download_iwslt15.sh ./nmt_data
```

该命令就会将数据集新建文件夹并保存在nmt的目录下。

默认数据集共有6个文件。
- train.en
- train.vi 
- tst2012.en 
- tst2012.vi 
- tst2013.en 
- tst2013.vi 
- vocab.en 
- vocab.vi


train文件即训练数据集，tst2012与tst2013为验证数据集，分做test和validation之用。Vocab为词典文件。

在GNMT模型的代码中，使用前缀+后缀的方式定位文件。前缀用于表示该文件用途，如train，后缀表示该文件对应的语言，如en为英语，vi为越南语。前缀与后缀都可以作为超参数进行设置，所以如果之前下载数据集在自定义目录下就可以调整前缀，将相对路径作为前缀的一部分设定超参数就可以了。

## 简单模型训练

Github文档给出的训练命令如下：

```bash
mkdir /tmp/nmt_model
python -m nmt.nmt \
    --src=vi --tgt=en \
    --vocab_prefix=/tmp/nmt_data/vocab  \
    --train_prefix=/tmp/nmt_data/train \
    --dev_prefix=/tmp/nmt_data/tst2012  \
    --test_prefix=/tmp/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_model \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu
```

注：训练命令是基于linux运行的，如果是windows运行要把折行符“\”去掉。

我们首先对超参数进行逐一解释。

+ Src为源语言，tgt为目标语言，该参数只要与数据集文件的后缀相对应即可。

+ vocab_prefix为词典文件前缀，train_prefix为训练集文件前缀，dev_prefix为验证集文件前缀，test_prefix为测试集文件前缀。如果你的数据集文件不在/tmp/nmt_data文件夹下，则将这些前缀参数中的/tmp/nmt_data替换成你的数据集路径即可。

+ out_dir为输出文件路径，输出内容为训练出的模型。

+ num_train_steps为训练轮数。

+ steps_per_stats为checkpoint储存频率，即训练多少轮后保存一次参数。

+ num_layers为网络深度，num_units为神经元个数即网络大小，这两个参数都是较泛的参数，可以通过encoder_units等更详细的超参数对网络结构进行更进一步的约束。

+ Dropout为随机丢弃的概率，可以使得采样更加随机，网络的泛化能力更强。

+ Metrics为测量标准，常用的参数即bleu，分数越高即翻译水准越好。

根据超参数可以对网络进行自定义的约束，运行上述命令即可训练一个简单的翻译模型。训练后的模型储存在tmp目录下的nmt_model文件夹中。

## 尝试翻译

在模型训练结束后会输出对当前训练模型的一些评估数据，默认以bleu得分作为评判标准，如果想要得到更加直观的翻译效果，则需要通过命令重新加载该模型，不训练只输出翻译结果。

首先，新建一个文本文件，输入想要翻译的句子，每个句子结束后换行，保证换行前为一个完整的句子。

也可以像文档中一样，用cat命令打开验证数据集，随机选取一些句子作为测试内容。

在新建好输入文件后运行如下命令，利用训练好的模型进行翻译预测：

```bash
python -m nmt.nmt \
    --out_dir=/tmp/nmt_model \
    --inference_input_file=/tmp/my_infer_file.vi \
    --inference_output_file=/tmp/nmt_model/output_infer
```

- out_dir为刚才训练出的模型的存储位置目录
- inference_input_file为需要进行翻译的源语言输入文件路径
- inference_output_file为模型输出结果的存储路径

在模型运行结束后，使用cat命令对输出文件进行查看，效果如下：

![翻译结果1](https://github.com/Lor-na/Train-Doc-For-NMT/blob/master/images/translation1.jpg?raw=true)

由于是简单模型，网络深度较浅，数据也较少，所以训练出的bleu比较低，在跑默认参数的情况下第一次训练出的bleu为4.8。

## 带有attention机制的模型训练

训练过程与之前基本相同，增加一个超参数`attention`用于训练带有注意力机制的翻译模型：

```bash
mkdir /tmp/nmt_attention_model
python -m nmt.nmt \
    --attention=scaled_luong \
    --src=vi --tgt=en \
    --vocab_prefix=/tmp/nmt_data/vocab  \
    --train_prefix=/tmp/nmt_data/train \
    --dev_prefix=/tmp/nmt_data/tst2012  \
    --test_prefix=/tmp/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_attention_model \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu
```

在网络的规模基本相同，数据集完全相同的情况下，注意力机制模型的bleu有16左右，可以说比简单的seq2seq模型提升了非常多。

这是两个模型对同一段语料的翻译结果，可以看出下面的翻译效果要明显的好出很多。

![翻译结果2](https://github.com/Lor-na/Train-Doc-For-NMT/blob/master/images/translation2.jpg?raw=true)



# 英德翻译模型及对抗

## 训练一个英德模型

在刚才的模型训练中我们看到，模型所需要的输入共有三部分：训练数据、词典和测试集。

在该项目中给出了一个完整的脚本，该脚本会下载`wmt16`的官方英德平行语料库，并对语料库进行预处理得到所需的词典和测试集。所以只需要运行一条命令即可完成训练英德模型的所有准备，超参数的设置同之前提到的脚本。

```bash
nmt/scripts/wmt16_en_de.sh /tmp/wmt16
```

数据准备好后就可以进行模型的训练了。英德数据集的规模相较之前的英越数据集要大很多，可以训练出一个更好的模型，但是要对模型的参数进行一定的调整，比如增加训练的epoch数量，对于网络的大小也要进行调整。在项目里也给出了一些官方建议的参考参数，其中就有英德模型训练的一套参数供使用，该参数的路径即超参数`hparams_path`所指向的路径。

```bash
python -m nmt.nmt \
    --src=de --tgt=en \
    --hparams_path=nmt/standard_hparams/wmt16_gnmt_4_layer.json \
    --out_dir=/tmp/deen_gnmt \
    --vocab_prefix=/tmp/wmt16/vocab.bpe.32000 \
    --train_prefix=/tmp/wmt16/train.tok.clean.bpe.32000 \
    --dev_prefix=/tmp/wmt16/newstest2013.tok.bpe.32000 \
    --test_prefix=/tmp/wmt16/newstest2015.tok.bpe.32000
```

运行这段命令，训练所需时长大概6-7天，训练结果`dev bleu = 26`，在`newstest2015`数据集上得到`bleu = 29`。

## trick——在不训练模型的时候测试bleu

在训练模型的时候，每隔固定的epoch模型就会输出一次测试得到的bleu值，但在训练结束后我希望对不同测试集进行bleu的测试该如何做？

其实在项目中提供了相应的bleu计算函数，我们只需要编写一个简单的python脚本调用这个函数即可。

```python
from .utils import evaluation_utils
import sys
ref = sys.argv[1]
hyp = sys.argv[2]
print(evaluation_utils.evaluate(ref, hyp, 'bleu'))
```

将该python脚本命名后（如eval_bleu.py）放入nmt目录下，调用时给出两个超参数——参考标准文件的路径和模型结果文件的路径，比如：

```bash
python -m nmt.eval_bleu ./origin.de ./output
```

## 加入噪声

To be continued...



# 中英翻译模型训练

## 数据集下载

本文档使用的是`AI Challenger Tranlation 2017`比赛数据集，在wmt17官网也可以下载到中英的平行语料库。

## 数据集预处理

由于原始数据集只有平行语料训练集，所以需要自己对数据进行处理并生成词典，这一步我们将分为中英两部分处理。

### 英文数据集预处理

为了得到一个词典我们首先要对原始语料进行分词，英文的分词其实比较容易，由于词与词之间有空格，所以可以通过空格来完成对一句话中词的分割。由于标点符号和单词之间有时没有空格，所以我们需要对语料库做一次tokenize，将标点符号与词语拆开来。

通过这样简单的对语料进行分割处理，我们会得到一个非常大的词典.，约6M左右。同时思考到模型所需要的词典其实只需要高频词，我们可以对刚生成的词典做一次频数筛选，假设词典中只包含频数大于3的词，则词典大小会降到2M左右，但这离模型的要求仍相距甚远。模型的decoder任务可以看作一个多分类任务，面对几万维甚至十几万维的向量模型很难收敛，根据论文词典大小在30K-80K左右会得到最好的效果。

所以不能通过频数筛选来简单的做一个词典，需要更加有效的算法对词典进行降维。在项目中，处理英德数据集的脚本使用了BPE算法，可以对词典做到很好的降维效果，通过编码，中英语料库可以生成一个约200K的词典。

BPE算法在github上有开源的项目，可以通过pip下载该库使用，[链接如下](https://github.com/rsennrich/subword-nmt)。

使用共分两步，首先通过语料库学习得到编码方式并生成词典：

```bash
subword-nmt learn-joint-bpe-and-vocab -i .\train.en -o .\code.file --write-vocabulary vocab.en
```

+ -i	输入文件名
+ -o   输出的code文件名（编码方式）
+ --write-vocabulary  输出词典

生成词典和编码方式后对英文数据集部分进行编码，包括训练集和测试集：

```bash
subword-nmt apply-bpe -i input.txt -c code.file -o output.txt
```

+ -i  输入文件名
+ -c  存储编码方式的文件
+ -o  输出文件名

这样就完成了对数据集的编码并得到了词典，英文数据集处理部分完成。

### 中文数据集预处理

中文相当于英文来说更加复杂，其基本单元从24个英文字母变成了数量级超越好几倍的中文单字，字与字之间的组合更是非常的多，当然中文也可以同样的使用bpe算法对分词后的结果进行降维压缩，但在这里我们使用一个更加直接简单的算法，即直接分字。

虽然中文的字较英文而言多出许多，但其也有好处——每个字都本身携带了一定的意义，而且这个意义往往在其组成词汇之后仍保留着。并且，进一步分析分词得到词典的意义，是将一个长句拆成多个包含基本语义的单位，即词。那么分字其实也能达到同样的效果，只是将一个长句拆成了更加细化的语义单位。

分字参考脚本路径：`/scripts/subword_chinese.py`。

若要直接使用该脚本，请将训练集放在脚本同级目录下，命名为`train.zh`，生成的词典结果也会保存在同级目录下名为`vocab.zh`

采用分字得到的语料库的中文词典降维效果十分明显，词典大小只有`33K`，完全符合训练需求。

## 数据集训练

训练方式与英德模型一致，请参照文档英德翻译模型部分。

模型训练约3天，测试集`bleu = 18`，能基本实现语义通顺。

![1576733428735](https://github.com/Lor-na/Train-Doc-For-NMT/blob/master/images/1576733428735.png?raw=true)

从结果中也可以看出，分字的做法有其优点也相应有很大的缺点。

优点显而易见，降维压缩的效果比较好，对于`decoder`的负担大大减小，模型的收敛速度比以前快了许多。

缺点也同样在于降维上，对于字来说，组合成词而附加的信息是中文词汇信息中非常重要的部分，占了很大的比重， 从句直接降维到字丢失的信息过于多了，对于表义丰富的字很难做到所有的语义能清晰的训练分辨出来，所以在翻译成句的时候会有语义误判的情况出现。

提示： 由于模型训练时是基于bpe编码后的英文训练集进行训练，所以测试时的英文原句也要同样经过编码后再用模型进行预测，编码格式与训练集编码时生成的编码文件一致即可。

## 模型结果演示

在训练好模型之后，写了一个简单的前端demo以方便对结果进行测试，demo的代码在项目`/translate_demo`目录下。

如要运行，需要配置`django`环境。配置好后将模型训练结果和模型训练源数据放在`manage.py`同级目录下，并对`/similar/translate_with_model.py`进行如下修改：

```python
# original code
val = os.system('python -m nmt.nmt \
                 --out_dir=./enzh_gnmt_singleword \
                 --inference_input_file=./input.txt \
                 --inference_output_file=./output.txt')
```

将上述代码中的`out_dir`修改为模型训练结果路径即可。

运行前端框架命令：

```bash
python manage.py runserver 8000
```

之后在浏览器端访问`xxx.xxx.xxx.xxx:8000/index`就可以看到前端页面了，xxx替换为服务器IP地址，例如本机就替换为`127.0.0.1:8000/index`。

在demo中可同时进行多句话的翻译，不同句之间用回车换行分隔。演示实例如下：

![1576734720035](https://github.com/Lor-na/Train-Doc-For-NMT/blob/master/images/1576734720035.png?raw=true)