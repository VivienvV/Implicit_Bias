# Debiasing Implicit Gender Bias

In this repository, we present the code we  used for our project `Towards Mitigating Implicit Gender Bias in Sentence Embeddings using
Self-Debias`, carried out for the course Advanced Topics in Computational Semantics at the University of Amsterdam By Ekaterina Shutova, Alina Leidinger and Rochelle Choenni.
In this project we evaluate whether we can use the Self-Debiasing technique by Schick et al (20..) to mitigate implicit gender agency bias from pre-trained language models. 

Our code is mostly based on the code by Meade et al. (20..). The folder `bias-bench` contains this code, with some small adaptations to our own setup. 

Our experiments are carried out in the following notebooks:
* `prompt_experiments.ipynb` contains our experiments with prompts
* `gender_proxies.ipynb` contains our experiments with various gender proxies 
* `racial_group_experiments.ipynb` for different racial groups
* `model_size_experiments.ipynb` for different model_sizes


To blank out gendered terms from the input text, we use code by Huang et al. which can be found here: https://github.com/tenghaohuang/uncover_implicit_bias. The two specific Python files we use can also be found in the folder `blank_gender`. The files can be run in the following order:

1. Classify sentences as having either a male, female, or unresolved gender protagonist
      ```sh
      python3 preprocess.py <story_filename.csv>
      ```
2. Blank out the gendered terms in male and female sentences
      
      ```sh
      python3 replaceGender.py 
      ```      

Furthermore, the file data_to_json.py can be used to complete the blanked out ROCStories data with gendered continuations in order to form gender pairs to evaluate. This can be ran as follows :       
```
python data_to_json.py '[SETTING]' [ID]        
```  
where `[SETTING]` is the name of the setting, which decides what gender proxy to use (e.g. 'tokenized' to only use singly tokenized names) and [ID] the id of the current dataset. The output will be a file: `gender_data_[SETTING]_[ID].json` in the same directory.
In our experiments, we generate 5 versions of each datasettint (ID 0-4), and the datafiles should be placed in the folder `bias_bench/stereoset/data`

to create the `namedict.pt` and `namedict_token.pt` files needed to run data `data_to_json.py`, `create_namedict` needs to be ran. to do this, you will need to 
* download `firstnames.xlsx` from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TYJKEZ and place it in this directory
* download `names.zip`, unzip the file, and place the resulting directory in this directory.
 



