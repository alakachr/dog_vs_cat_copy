### Cat vs Dog classification transfer web app 

This is a web app that displays the results of a binary classifier vision model. The model is trained on the *dogs vs cat* data.

The model is a mobilenetv2 finetuned on the the data, trained with pytorch. Our model achieves 80% accuracy with 5 epochs. We chose mobilenet because it is light weight, fast and easy to train.

Some images of cats and dogs have been provided to test the API. A subsample of the dataset is also provided at *model_training/train_dataset* to launch a quick training.



### To launch the app
-Run ```make build```
-Then ```make up```
-Then go to localhost:8501 in your browser

### To see the API doc
Once you launch the app, you can go to localhost:8080/docs to display the API generated documentation

### If you want to do some api unit testing
You can ssh into the backend container and then run *unit_test_api.py*

### To train the model
- Download the dataset from kaggle : https://www.kaggle.com/c/dogs-vs-cats/data
- Go to the *backend/model_training* folder
- Place the train images in the *train_dataset* folder
- To train in docker, Run ``` make build ``` then ``` make run ```