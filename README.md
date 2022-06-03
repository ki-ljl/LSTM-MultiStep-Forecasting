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
│  README.md
│  tree.txt
│  util.py
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
├─model
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

```
1. **args.py** is a parameter configuration file, where you can set model parameters and training parameters.
2. **data_process.py** is the data processing file. If you need to use your own data, then you can modify the load_data function in data_process.py.
3. Multiple models are defined in **models.py**, including LSTM, bidirectional LSTM, and seq2seq.
4. **util.py** defines the training and testing functions of the models in the five multi-step prediction methods.
5. The trained model is saved in the **model** folder, which can be used directly for testing. The mms folder saves the model of multi-model-scrolling forecasting, and the mmss folder saves the model of multi-model-single-step forecasting.
6. Data files in csv format are saved under the **data** file.
# Usage
First switch the working path:
```bash
cd LSTM-MultiStep-Forecasting
```
Then, execute in sequence:
```bash
python algorithms/multi_model_scrolling.py
python algorithms/multi_model_single_step.py
python algorithms/multiple_outputs.py
python algorithms/seq2seq.py
python algorithms/single_step_scrolling.py
```
If you need to change the parameters, please modify them manually in args.py.
# Result
Predict the next 12 steps, epochs=50, bacth_size=30, and the results of the 5 methods are shown in the following table:
| method| 1 | 2|3 |4|5 |
|--|--|--|--|--|--|
| MAPE/% | 9.33 |10.62 |9.94 | 22.45|9.09 |
