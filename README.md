# TCMmodel 望庐模型
本模型名称望庐模型，是基于规则生成医案数据联合Transformer模拟方证到方药之间的非线性复杂映射的中医模型。

运行环境：建议使用python3.9，需安装cuda，依赖包参考源码

模型耗时参考：本人独立显卡RTX 2050 X1 ，使用data.xlsm规则数据训练耗时5天

使用简易教程：将Generator.py、Transformer.py、data.xlsm下载至同一个文件件，先运行Generator产生医案，后运行Transformer进行模型训练



（自定义规则教程待以后补充）
