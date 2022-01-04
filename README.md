**Report**
- https://github.com/jessicaarrouasse/Object-Detection/blob/main/Report%20Chest%20X-ray%20Abnormalities%20Detection.pdf

**How to use this repo**

- Install the dependencies in requirement.txt
- Download the data at _https://www.kaggle.com/xhlulu/vinbigdata-chest-xray-resized-png-256x256_ and unzip into the `data/` folder. _The original data not resized can be dowloaded here : https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data_
- Run the `main.py` file with the `action` parameter in this way:
```python main.py <action>```
   
- The `action` parameter can be one of the following: 
  - `init` - Create the full metadata file needed to train the model 
  - `train` - Train the model with the parameters setted in 'params' in the main.py file 
  - `eval` - Compute metrics (APs and mAP) for each model saved in `checkpoints/` 
  - `loss` - Show the graph loss/epochs for each model saved in `checkpoints/` 
  - `predict` - Show examples of predictions using the model given as a param in the `main_predict` function 

**Result examples:**
![image](https://user-images.githubusercontent.com/44928800/129338345-4d6d12e7-1c22-4a0c-a698-e5d1f3ed6bb6.png)

