## EEML Programming Assignment 1 
## Author: B06702064 會計五 林聖硯 
## Date: 2021/10/19
---
How to run this program:
- modify the data path in b06702064.ipynb
- **run ALL the cells in b06702064.ipynb**
- the tuning result will show on your terminal and a file called "prediction.csv" will be created for kaggle submission.

How to submit prediction:
- pip install kaggle
- cd ~/.kaggle
- Create New API token in your kaggle account page
- mv ~/Downloads/kaggle.json ./
- chmod 600 ./kaggle.json
- kaggle competitions submit -c ntueeml2021fallhw1 -f "prediction.csv" -m "Message"