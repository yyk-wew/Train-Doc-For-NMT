## char2char模型配置
---
paper中给出了噪声合成方式的代码,在代码的readme中提供了作者使用的char2char模型repo,为了复现论文,需要对模型进行配置.由于该模型使用的theano版本较低,在Windows上与CUDA驱动不兼容,请在Linux上配置

### 步骤
1. 克隆charNMT-noise的repo以及char2char模型代码
> git clone https://github.com/ybisk/charNMT-noise.git

> git clone https://github.com/nyu-dl/dl4mt-c2c.git

2. 安装conda
3. 如果安装的是conda的python3版本,新建一个python2.7的环境.
> conda create -n theano python=2.7

接下来的步骤在nvidia driver版本=430.50上测试成功,若测试时有问题,请自行尝试其它cuda版本

4. 在conda环境中安装theano
> conda install theano=1.0.4

模型代码基于的theano版本早于1.0.0, 所用API接口与1.0.0版本后并不一致,在之后需要对一些方法手动替换.不直接使用早期版本的原因是会出现各种花式bug.

5. 安装nltk
> conda install nltk

6. 安装 cudatoolkit 以及 cudnn

测试时使用以下版本可以正常使用GPU训练

> conda install cudatoolkit=8.0 cudnn=7.0.5

7. 复制一份cudnn的静态库文件到编译器的lib文件夹中.

如果之后训练模型出现了跟下面类似的问题

    ~/anaconda3/envs/test/bin/../lib/gcc/x86_64-conda_cos6-linux-gnu/7.3.0/../../../../x86_64-conda_cos6-linux-gnu/bin/ld: cannot find -lcudnn

这可能是因为在外部已有一个gcc的链接器ld,即使将lib文件添加到环境变量中,调用conda环境内的链接器时也并未获得此路径,因此最后将 libcudnn复制到编译器的默认lib文件夹即可.

> cp envs/$envname/lib/libcudnn.so envs/$envname/x86_64-conda_cos6-linux-gnu/lib/

注: 写文档时看到一篇博客,提到theano编译时用的编译器不固定(当环境中有多个gcc时),可以在.theanorc中增加以下配置解决

> cxx=COMPILER_PATH

但我并未进行测试,另外,可能还会缺少别的lib文件,因此复制所有的.o到默认lib文件夹下也是不错的选择.

1. 修改代码

theano 更新过后端代码,导致一些接口变化, 具体可以见以下链接:
> https://github.com/Theano/Theano/wiki/Converting-to-the-new-gpu-back-end%28gpuarray%29

将clone后的report内的代码检查一遍,将所有变化的API进行修改.

9. 测试

模型训练时需要手动设置环境变量,因此推荐写一个shell脚本,示例的代码如下:

```
export C_INCLUDE_PATH=~/anaconda3/envs/theano/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=~/anaconda3/envs/theano/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=$LIBRARY_PATH:~/anaconda3/envs/theano/lib
export THEANO_FLAGS=device=cuda0,floatX=float32
#use default config, models from paper, adam with lr = 1e-4, grad clip and with L2 normalization
python train_bi*.py -translate de_en -dispFreq 25 -sampleFreq 50  -saveFreq 10000 -batch_size 64  -learning_rate 0.0001

```

这一步安装后运行时仍提示找不到-lcudnn或其他与cuda相关的异常的话,可以降级cudnn版本试试

10. 关于数据集

在dl4mt-c2c的repo内,模型作者提供了预处理的数据集下载,因此不再赘述.

11. 关于训练参数

经过我的多次尝试,模型使用默认参数时没有训练效果,经检查发现默认参数即作者在paper中提供的参数,因此难以断定有什么问题.如果你也遇到一样的问题,可以使用作者提供的预训练模型(笑)

### 环境配置,有问题可以自行对照
```
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main    defaults
backports-shutil-get-terminal-size 1.0.0                    pypi_0    pypi
binutils_impl_linux-64    2.31.1               h6176602_1    defaults
binutils_linux-64         2.31.1               h6176602_8    defaults
ca-certificates           2019.8.28                     0    defaults
certifi                   2019.9.11                py27_0    defaults
cudatoolkit               8.0                           3    defaults
cudnn                     7.0.5                 cuda8.0_0    defaults
decorator                 4.4.0                    pypi_0    pypi
enum34                    1.1.6                    pypi_0    pypi
gcc_impl_linux-64         7.3.0                habb00fd_1    conda-forge
gcc_linux-64              7.3.0                h553295d_8    conda-forge
gxx_impl_linux-64         7.3.0                hdf63c60_1    conda-forge
gxx_linux-64              7.3.0                h553295d_8    conda-forge
intel-openmp              2019.4                      243    defaults
ipdb                      0.12.2                   pypi_0    pypi
ipython                   5.8.0                    pypi_0    pypi
ipython-genutils          0.2.0                    pypi_0    pypi
libblas                   3.8.0               12_openblas    conda-forge
libcblas                  3.8.0               12_openblas    conda-forge
libffi                    3.2.1             he1b5a44_1006    conda-forge
libgcc-ng                 9.1.0                hdf63c60_0    defaults
libgfortran-ng            7.3.0                hdf63c60_0    defaults
libgpuarray               0.7.6             h14c3975_1003    conda-forge
liblapack                 3.8.0               12_openblas    conda-forge
libopenblas               0.3.7                h6e990d7_1    conda-forge
libstdcxx-ng              9.1.0                hdf63c60_0    defaults
mako                      1.1.0                      py_0    conda-forge
markupsafe                1.1.1            py27h14c3975_0    conda-forge
mkl                       2019.4                      243    defaults
mkl-service               2.3.0            py27h516909a_0    conda-forge
ncurses                   6.1               hf484d3e_1002    conda-forge
nltk                      3.4.4                      py_0    conda-forge
nose                      1.3.7                 py27_1002    conda-forge
numpy                     1.16.4           py27h95a1406_0    conda-forge
openssl                   1.1.1d               h7b6447c_2    defaults
pathlib2                  2.3.5                    pypi_0    pypi
pexpect                   4.7.0                    pypi_0    pypi
pickleshare               0.7.5                    pypi_0    pypi
pip                       19.2.3                   py27_0    conda-forge
prompt-toolkit            1.0.16                   pypi_0    pypi
ptyprocess                0.6.0                    pypi_0    pypi
pygments                  2.4.2                    pypi_0    pypi
pygpu                     0.7.6           py27h3010b51_1000    conda-forge
python                    2.7.15            h5a48372_1009    conda-forge
readline                  8.0                  hf8c457e_0    conda-forge
scandir                   1.10.0                   pypi_0    pypi
scipy                     1.2.1            py27h921218d_2    conda-forge
setuptools                41.2.0                   py27_0    conda-forge
simplegeneric             0.8.1                    pypi_0    pypi
singledispatch            3.4.0.3               py27_1000    conda-forge
six                       1.12.0                py27_1000    conda-forge
sqlite                    3.29.0               hcee41ef_1    conda-forge
theano                    1.0.4           py27hf484d3e_1000    conda-forge
tk                        8.6.9             hed695b0_1002    conda-forge
traitlets                 4.3.2                    pypi_0    pypi
wcwidth                   0.1.7                    pypi_0    pypi
wheel                     0.33.6                   py27_0    conda-forge
zlib                      1.2.11            h516909a_1006    conda-forge

```
