# Natural Date String Parser
A sequence-to-sequence model to parse natural date strings into machine parsable date strings.
```
-
Input sentence: 28 September 97 - 21 May 98
Decoded sentence: 28/09/1997 - 21/05/1998

-
Input sentence: 03-07-2004
Decoded sentence: 07/03/2004

-
Input sentence: 03 September 13
Decoded sentence: 03/09/2013

-
Input sentence: 23 aug 18 - 18 jul 19
Decoded sentence: 23/08/2018 - 18/07/2019
```

### Usage
1. Swith to new environment and install requirements:  
`pip install -r requirements`
2. Generate synthetic dataset  
`python generate_trainset.py`
3. Train and save a model   
`python train.py`
4. Demo this in an app  
`streamlit run app.py` 
