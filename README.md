# LSTM-MultiStep-Forecasting
Implementation of Electric Load Forecasting Based on LSTM (BiLSTM). Including direct-multi-output forecasting, single-step-scrolling forecasting, multi-model-single-step forecasting, multi-model-scrolling forecasting, and seq2seq forecasting.

# Environment
pytorch==1.10.1+cu111

numpy==1.18.5

pandas==1.2.3

# Tree
```bash
.
│  args.py
│  data_process.py
│  models.py
│  model_test.py
│  model_train.py
│  README.md
│  tree.txt
<<<<<<< HEAD
│  model_train.py
=======
│  
├─.idea
│  │  LSTM-MultiStep-Forecasting.iml
│  │  misc.xml
│  │  modules.xml
│  │  other.xml
│  │  vcs.xml
│  │  workspace.xml
│  │  
│  └─inspectionProfiles
│          profiles_settings.xml
│          Project_Default.xml
>>>>>>> 3246c00c27084b2a0820426e2c07e4f68649f44c
│          
├─algorithms
│      multiple_outputs.py
│      multi_model_scrolling.py
│      multi_model_single_step.py
│      seq2seq.py
│      single_step_scrolling.py
│      
├─data
│      data.csv
│      
<<<<<<< HEAD
├─models
   │  multiple_outputs.pkl
   │  seq2seq.pkl
   │  single_step_scrolling.pkl
   │  
   ├─mms
   │      0.pkl
   │      1.pkl
   │      10.pkl
   │      11.pkl
   │      2.pkl
   │      3.pkl
   │      4.pkl
   │      5.pkl
   │      6.pkl
   │      7.pkl
   │      8.pkl
   │      9.pkl
   │      
   └─mmss
           0.pkl
           1.pkl
           10.pkl
           11.pkl
           2.pkl
           3.pkl
           4.pkl
           5.pkl
           6.pkl
           7.pkl
           8.pkl
           9.pkl

=======
└─models
    │  multiple_outputs.pkl
    │  seq2seq.pkl
    │  single_step_scrolling.pkl
    │  
    ├─mms
    │      0.pkl
    │      1.pkl
    │      10.pkl
    │      11.pkl
    │      2.pkl
    │      3.pkl
    │      4.pkl
    │      5.pkl
    │      6.pkl
    │      7.pkl
    │      8.pkl
    │      9.pkl
    │      
    └─mmss
            0.pkl
            1.pkl
            10.pkl
            11.pkl
            2.pkl
            3.pkl
            4.pkl
            5.pkl
            6.pkl
            7.pkl
            8.pkl
            9.pkl
>>>>>>> 3246c00c27084b2a0820426e2c07e4f68649f44c
```
1. **args.py** is a parameter configuration file, where you can set model parameters and training parameters.
2. **data_process.py** is the data processing file. If you need to use your own data, then you can modify the load_data function in data_process.py.
3. Three models are defined in **models.py**, including LSTM, bidirectional LSTM, and seq2seq.
4. **model_train.py** defines the training functions of the models in the five multi-step prediction methods.
5. **model_test.py** defines the testing functions of the models in the five multi-step prediction methods.
6. The trained model is saved in the **models** folder, which can be used directly for testing. The mms folder saves the model of multi-model-scrolling forecasting, and the mmss folder saves the model of multi-model-single-step forecasting.
7. Data files in csv format are saved under the **data** file.
# Usage
First switch the working path:
```bash
cd algorithms/
```
Then, execute in sequence:
```bash
python multi_model_scrolling.py --epochs 50 batch_size 30
python multi_model_single_step.py --epochs 50 batch_size 30
python multiple_outputs.py --epochs 50 batch_size 30
python seq2seq.py --epochs 50 batch_size 30
python single_step_scrolling.py --epochs 50 batch_size 30
```
If you need to change the parameters, please modify them manually in args.py.
# Result
Predict the next 12 steps, epochs=50, bacth_size=30, and the results of the 5 methods are shown in the following table:
| method| 1 | 2|3 |4|5 |
|--|--|--|--|--|--|
| MAPE/% | 9.33 |10.62 |9.94 | 22.45|9.09 |
