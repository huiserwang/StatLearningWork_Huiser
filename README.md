## Create an env for this code

``` shell
conda create -n stat python=3.7
conda activate stat
pip install -r requirements.txt
```

## Run
``` shell
cd statLearningWork_SJTU
python run.py --method knn --out ./default_knn_test.csv
```

## Note!!
If you want to test convnet, you should install pytorch(1.9.0+cuda111) and torchvision(0.10.0+cuda111). Other versions maybe also OK.