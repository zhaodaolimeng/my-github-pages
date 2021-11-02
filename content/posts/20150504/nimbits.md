---
title: "Nimbits：一个IoT数据汇集平台"
date: 2015-05-04
categories:
    - "IoT"
tags:
    - "IoT"
    - "平台"
    - "配置"
draft: false
---

## 1 简介

[Nimbits](https://github.com/bsautner/com.nimbits)是一个专门针对物联网设计的数据汇集平台。该平台使用嵌入的hsql作为数据存储，使用Java作为编程语言，运行于Web环境中。相对于现有的Google/IBM 等公司的用于IoT的云平台，该平台简单、灵活、高度可定制，可为小型IoT业务提供了一套完整的数据解决方案。本文将介绍如何在本地搭建Appscale集群，并于之上部署Nimbits。

## 2 运行环境搭建

需要先后安装Virtual Box、Vagrant、Appscale。安装流程解释请参照
[fast-install](http://www.appscale.com/faststart)和[Appscale wiki](https://github.com/AppScale/appscale/wiki/AppScale-on-VirtualBox)。

### 2.1 安装VirtualBox和Vagrant

VirtualBox提供虚拟机运行环境，Vargrant提供虚拟机配置的复制和分离运行，类似于Docker的雏形。在安装Vargrant插件时，可能会出现删除VirtualBox的警告，可直接忽视。安装完成之后需要建立Vagrantfile，该文件用于指定虚拟机初始化内存、IP等配置信息。使用`vagrant up`命令启动vagrant虚拟机，使用`vagrant ssh`命令与vagrant虚拟机进行ssh通讯。

### 2.2 安装Appscale服务

Appscale是类似于Google App Engine的一个分布式应用运行平台。目标搭建的服务分别部署于本机和虚拟机集群。首先，下载基于Ubuntu 12.04的[Appscale镜像](http://download.appscale.com/apps/AppScale%201.12.0%20VirtualBox%20Image)，可使用wget和aria2c从备用地址加速镜像下载过程。该镜像已经包含集群环境。其次，在本地安装Appscale工具。使用命令`appscale init cluster`启动集群，使用命令`appscale up`启动控制服务。使用`appscale clean`和`appscale down`分别可清除部署和停止服务。启动服务时，可能出现如下问题：
```sh
stacktrace : Traceback (most recent call last):
File "/usr/local/appscale-tools/bin/appscale", line 57, in
appscale.up()
File "/usr/local/appscale-tools/bin/../lib/appscale.py", line 250, in up
AppScaleTools.run_instances(options)
File "/usr/local/appscale-tools/bin/../lib/appscale_tools.py", line 362, in run_instances
node_layout)
File "/usr/local/appscale-tools/bin/../lib/remote_helper.py", line 202, in start_head_node
raise AppControllerException(message)
...
```
根据这个讨论，关闭所有的Python、Java和Ruby程序可以修复该问题。

### 2.3 在集群环境下部署Nimbits

本地使用环境为Ubuntu 12.04、mysql 5.5、Java 7。启动之后可访问以下API进行测试，Nimbits会对应返回系统时间。
```
http://localhost:8080/nimbits/service/v2/time。
```

## 3. 实例：使用Nimbits当计数器

可使用js向服务器发送一个post命令，用来测试Nimbits功能。
```js
function updateCounter() {
    $.post(
        "http://cloud.nimbits.com//service/v2/value",
        {
            email: "youremail@gmail.com",
            key: "secret",
            id:  "youremail@gmail.com/counter",
            json:"{"d":1.0}"
        },
        function(data){  }
    );
}
```
需要注意的是当使用POST方法时，Nimbits端对应节点的设置需要是Read/Write To Single Point。每次POST请求后，对应的节点"d"的数据会加1。
