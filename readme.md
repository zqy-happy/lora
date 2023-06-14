1.运行
conda env create -f environment.yml
会生成一个 pytorch的conda环境  

2.运行 conda activate pytorch   

3. 运行 python pre_dataset.py   
用来下载数据集，其中数据集将存储在 ./datasets/food101  
运行 python main.py -lr 0.005 >> main.log   

我想试一遍 lr 从0.005 0.001 0.0005 这三个值，你可以并行执行，也可以串行执行



