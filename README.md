# Forest Cover Type Classification

This project shows how to create a pipeline which uses model deployed in google cloud. `forest_classification.py` can be used to call the model deployed. The forest cover type classification is a problem taken from https://www.kaggle.com/c/forest-cover-type-prediction/overview. 

`forest_cover` folder contains the code for creating custom prediction routing. At the moment it is not working since AI platform does not support this tensorflow version of > 2. 

Refer to `forest_classification.py` for creating the working pipeline which uses the model deployed on AI Platform. 
