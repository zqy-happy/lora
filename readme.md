- please modify the last line of file 'environment.yml' with your own conda path.
- run "conda env create -f environment.yml"  
 it will create a conda environment called "pytorch"   

- run "conda activate pytorch"  

- run "python pre_dataset.py"    
this command is used to install datasets and store them in "./datasets/food101"

- run "python main.py -lr 0.005 >> main.log"  
if run from a checkpoint please add "-cp 1"  
python main.py -lr 0.005 -cp 1 >> main.log  




