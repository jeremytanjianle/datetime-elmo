# Natural Date String Parser
A sequence-to-sequence model to parse natural date strings into machine parsable date strings.

| Input                | Output                    |
|----------------------|---------------------------|
| 1st of March 2019    | 01/03/2019 eos            |
| 15 Febroory 2008     | 15/02/2008 eos            |
| 19 Sept - 27 Sept 99 | 19/08/1999 - 27/08/99 eos |

### Usage
1. Swith to new environment and install requirements:  
`pip install -r requirements`
2. Generate synthetic dataset  
`python generate_trainset.py`
3. Train and save a model   
`python train.py`
4. Demo this in an app  
`streamlit run app.py` 
