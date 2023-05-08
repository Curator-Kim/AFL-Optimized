# 基于深度学习的AFL优化探究

## 种子生成模型

数据集：Samples.zip 主要为经AFL变异后的elf文件

+ Seed_Optimized_DNN.py

  全连接网络作为生成器和判别器的架构

+ Seed_Optimized_LSTMpy

  生成器采用LSTM网络，判别器采用全连接网络

## 变异选择模型

指导AFL选择更佳变异位置

+ Mutation_Optimize_LSTM

## Others

+ Data_Processing.py

  一些数据预处理操作，以及断点重载

待优化...