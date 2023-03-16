## EEML Programming Assignment 5
## Author: B06702064 會計五 林聖硯 
## Date: 2021/12/28
---
- model path: https://drive.google.com/file/d/1h1LVKBVM7lZiSlxTaNmCuQDQsNEH_lV8/view?usp=sharing

- make sure the following files are stored in the same directory.
    - `Kaggle.ipynb` (Kaggle competition)
    - `Visualization.ipynb` (report - visualization)
    - `Eigenface.ipynb` (report - eigenface)
    - `Math.ipynb` (report - math)
    - `utils.py` (model class, training functions, ...)
    - `AE.pth` (my best training model)
    - `trainX.npy` (training data)
    - `visualization.npy` (for visualization)
    - `.\Aberdeen\*.jpg` (for report)

How to reproduce my kaggle results (use `Kaggle.ipynb`):
- you should modify the DATA_PATH and MODEL_PATH in this file if you store the data in another directory
- run ALL the cells in b06702064.ipynb before the **Training record** section
- A file called "prediction.csv" will be created for kaggle submission.

How to reproduce my **visualization results** (use `Visualization.ipynb`):
- you should modify the DATA_PATH, DATA_VIS_PATH and MODEL_PATH in this file if you store the data in another directory
- run ALL the cells in `Visualization.ipynb` before the **Evaluation** section
- The **Evaluation** section is for calculate accuracy of my AE and clustering result.

How to reproduce my **eigenface results** (use `Eigenface.ipynb`):
- Please make sure your kernel have enough CPU memory to prevent from OOM!!
- If your memory is not enough, please run **Problem1** section, restart the kernel, then run **Problem2** setion 
- you should modify the IMG_PATH in this file if you store the data in another directory
- run ALL the cells in `Eigenface.ipynb`

How to reproduce the results in **MATH Problem** (use `Math.ipynb`):
- run ALL the cells in `Math.ipynb`