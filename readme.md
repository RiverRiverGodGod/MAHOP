1. This project mainly includes the implementation and testing of two models, MAHOP (Multi-Time Scale Aware Host Task Preferred Learning model) 
   and MAHOP/st, which have significant advantages in the field of multi-agent time series prediction. Through multi-task learning, 
   the similarity of different agent temporal features is fully extracted, and the learning effect of the main task is maximized through the help of auxiliary tasks. 
   The MAHOP model has been proven to have the ability to significantly improve the model's prediction performance in the field of waste household appliance recycling. 

2. Due to the privacy of the data collection volume, the data used in the project is desensitized and stored in ./data/cate_data.csv. 
   Data_Desensitization.py records the desensitization process, which is irreversible. 

3. Configure the project environment through pip install -r requirements.txt. 

4. Two newly proposed models (MAHOP and MAHOP/st) are stored in model.py, model_contrast.py, and model_improvement.py, respectively. 
   Start the project by running the main.py file. ./utils/config.py stores all the model hyperparameters, which can be modified as needed. 

5. engine.py is the model training file. Note that when predicting washing machines, refrigerators, and air conditioners and predicting TVs, 
   the calculation method of the loss needs to be adjusted according to specific requirements. 

6. The log folder will store the corresponding log files. Other comparative models, arima and MULAN, are stored in engine_arima.py and old.py.