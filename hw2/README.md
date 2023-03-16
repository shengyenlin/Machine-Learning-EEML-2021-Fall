## EEML Programming Assignment 2
## Author: B06702064 會計五 林聖硯 
## Date: 2021/10/29
---
How to run this program hw2_best.ipynb:
- **modify the data path in hw2_best.ipynb**
- **run ALL the cells in hw2_best.ipynb**
- the tuning result will show on your terminal and a file called "prediction.csv" will be created for kaggle submission.

How to run this program hw2_logistic.ipynb:
- **modify the data path in hw2_logistic.ipynb**
- **run ALL the cells in hw2_logistic.ipynb**
- the tuning result will show on your terminal and a file called "prediction_logistic.csv" will be created for kaggle submission.

How to run this program hw2_generative.ipynb:
- **modify the data path in hw2_generative.ipynb**
- **run ALL the cells in hw2_generative.ipynb**
- the tuning result will show on your terminal and a file called "prediction_generative.csv" will be created for kaggle submission.

How to submit prediction:
- pip install kaggle
- cd ~/.kaggle
- Create New API token in your kaggle account page
- mv ~/Downloads/kaggle.json ./
- chmod 600 ./kaggle.json
- kaggle competitions submit -c ml-2021fall-hw2 -f "prediction.csv" -m "Message"