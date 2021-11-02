---
title: "Snorkel学习笔记"
date: 2017-09-03
categories:
    - "NLP"
tags:
    - "NLP"
    - "文本抽取"
draft: false
---


## 1 简介

Snorkel是deepdive的后续项目，当前在github上很活跃。Snorkel将deepdive中的弱监督的思想进一步完善，使用纯python的形式构成了一套完整弱监督学习框架。

## 2 安装
由于Snorkel当前还处于开发状态，所以官方的安装教程不能保证在所有机器上都能顺利完成。
官方使用了anaconda作为python支持环境，而这个环境在个人看来存在不少问题（在单位这个win 7机器上异常的慢）。
我直接在一个ubuntu虚拟机内使用了virtualenv和原生python2.7对这套环境进行了安装。

**step 1: 部署python virtualenv**

为了不污染全局python环境，官方推荐使用conda进行虚拟环境管理，这里使用了virtualenv。
```sh
sudo apt install python-pip python-dev essential-utils
pip install python-virtualenv
cd snorkel
virtualenv snorkel_env
source snorkel_env/bin/activate
```

根据官方文档安装依赖，并启用jupyter对应的功能。
```sh
pip install numba
pip install --requirement python-package-requirement.txt
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

**step 2: 配置对虚拟机jupyter notebook的远程连接**

jupyter notebook规定远程连接jupyter需要密码。
使用以下命令生成密码的md5序列，放置到jupyter的配置文件中（也可以有其他生成密码的方式）。
```sh
jupyter notebook generate-config
python -c 'from notebook.auth import passwd; print passwd()'
vim ~/.jupyter/jupyter_notebook_config.py
```

修改生成的jupyter配置文件。
```sh
c.NotebookApp.ip = '*'
c.NotebookApp.password = u'sha1:bcd259ccf...your hashed password here'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
```

这里的虚拟机我使用了virtualbox，所以还需要对nat的端口进行映射，外部host才能访问到虚拟机中运行的jupyter。

**step 3: 启动项目**

按理来说直接执行./run.sh即可，但我在这里执行之后set_env.sh没有生效，并且没有错误提示。遇到该情况后，单独执行了以下命令，手工实现了run.sh的功能。
```sh
./install-parser.sh
unzip stanford-corenlp-full-2015-12-09.zip
mv stanford-corenlp-full-2015-12-09/stanford-corenlp-3.6.0.jar .
source set_env.sh
cd "$SNORKELHOME"
git submodule update --init --recursive
jupyter notebook
```

## 3 内部机制
与Deepdive的方式相同，一般使用一个外部库进行distance supervise，通过文本、文档结构等多种显性特征进行实体关系推断。
开发者可以人工将这些显性特征进行深度的加工和组合，定义出不同的LF（Labeling Function）辅助特征提取，这对应了Deepdive中的UDF（User Defined Function）。
还没有彻底弄清楚这里提到的GenerativeModel是否真的和传统意义上的生成模型是一回事？为什么感觉这里的Generative Model好像是一个以判别为目标的全新的模型？

用例：
- Weather sentiment。对于tweet中的言论，已有少量的trusted work标注的信息和大量crowdsource的标注信息。Snorkel在这里使用时，将每个人的标注准确度作为一个随机变量，通过估计这些随机变量的分布，修正最终的标签。
- Spouse。这个任务和deepdive中的例程是相同的，都是通过一部分人工规则和辅助库，估计当前规则中是否存在问题，最终生成大量的可用样本。

## 4 总结
Deepdive和Snorkel都是从扩充训练样本的角度出发，提供了一套完整的通过专家知识生成训练数据的方法。但在整体任务的角度，如果采用这些原始标签，仍然利用这些（用于生成训练样本的）专家知识（以冲突消解的形式）生成特征、之后再使用模型进行集成训练，在理论上很可能与Deepdive和Snorkel的方法是等效的，这些都还有待验证。不管怎么说，Deepdive和Snorkel对于知识抽取这样的单一任务而言都是不错的工具。
