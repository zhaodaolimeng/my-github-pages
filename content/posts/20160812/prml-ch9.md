---
title: "PRML.Ch9读书笔记：EM方法"
date: 2016-08-12
categories:
    - "读书笔记"
tags:
    - "PRML"
    - "机器学习"
    - "读书笔记"
draft: false
---

## 1 引子

本文涉及EM的使用场景区别、理论推导。以实践的角度，EM方法是一个迭代优化以求解隐变量的方法。本文内容是PRML第九章的缩减。

## 2 EM用于GMM模型
### 2.1 极大似然尝试求解GMM参数

以GMM为例，这里先试图使用最大似然估计的方式求解参数：

$$p(x|\pi,\mu,\Sigma)=\sum_{k=1}^{K}\pi_{k}\mathcal{N}(x|\mu_{k},\Sigma_{k})$$

最终目标是求解两部分内容：未观测变量和模型参数。个人理解对于GMM，其未观测变量可明确地指定为\\(\pi_{k}\\)，而其模型参数确定为\\(\mu_k\\)和\\(\Sigma_k\\)。这里优化目标是当前的估计导致的损失，或者说对数似然函数：

$$\ln{p(X|\pi,\mu,\Sigma)}=\sum_{n=1}^{N}{\ln\sum_{k=1}^K{\pi_k\mathcal{N}(x_n|\mu_k,\Sigma_k)}}$$

以上问题由于隐变量的存在，同时由于参数在正态分布的积分中，一般来说是难解的。具体地，对\\(\ln{p(X|\pi,\mu,\Sigma)}\\)求导，并令导数为0可以看出隐变量和参数之间的关系：

$$\frac{\partial{\ln{p(X|\pi,\mu,\Sigma)}}}{\partial{\mu_k}}=-\sum_{n=1}^{N} \gamma(z_{nk})\Sigma_k(x_n-\mu_k)=0$$

$$\frac{\partial{\ln{p(X|\pi,\mu,\Sigma)}}}{\partial{\Sigma_k}}
=\sum_{n=1}^N \gamma(z_{nk}) {-\frac{N}{2}\Sigma^{-1}+\frac{N}{2}\Sigma^{-1}\sum_{d=1}^{D}(x_i-\mu)^T \Sigma^{-1}_k (x_i-\mu)\Sigma^{-1}}=0$$
其中，\\(\gamma(z_{nk})\\)的物理意义是第n个观测在第k簇的概率，形式为：

$$\gamma(z_{nk})=\frac{\pi_k\mathcal{N}(x_n|\mu_k,\Sigma_k)}{\sum_j{\pi_j\mathcal{N}(x_n|\mu_j,\Sigma_j)}}$$

具体的结果可参考PRML。使用以上两个等式，原则上可计算参数和未观测量的值，这里是为了展现：由于对数中本身有加和的形式，这种方式难以获得解析解。需要有一个更便捷的框架解决以上参数求解问题。

### 2.2 EM方法估计GMM参数

EM方法正是这样一个框架：套用以上的结果，使用迭代的方法通过不断修正找到一个函数\\(q(x)\\) ，使得\\(q(x)\\)与\\(p(x)\\)接近，那么即可使用\\(q(x)\\)对最终结果进行近似。具体的步骤如下：

1. 初始化参数\\(\mu_k\\)、\\(\Sigma_k\\)和未观测值\\(\pi_k\\)。一个可行的方式是，由于K-means迭代次数较快，可使用K-means对数据进行预处理，然后选择K-means的中心点作为\\(\mu_k\\)的初值。
2. E步，固定模型参数，优化未观测变量：
$$\gamma(z_{nk})=\frac{\pi_k\mathcal{N}(x_n|\mu_k,\Sigma_k)}{\sum_j{\pi_j\mathcal{N}(x_n|\mu_j,\Sigma_j)}}$$
3. M步，M步将固定未观测变量，优化模型参数：
$$\mu_k^{new}=\frac{1}{N_k}\sum_{n=1}^{N}\gamma(z_{nk})\bf{x}_n$$
$$\Sigma_k^{new}=\frac{1}{N_k}\sum_{n=1}^{N}\gamma(z_{nk})(\bf{x}_n-\mu_k^{new})(\bf{x}_n-\mu_k^{new})^T$$
1. 计算likehood，如果结果收敛则停止。

## 3 EM方法正确性
（待续）
