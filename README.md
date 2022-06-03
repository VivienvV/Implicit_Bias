# Debiasing Implicit Gender Bias

In this repository, we present the code we  used for our project `Towards Mitigating Implicit Gender Bias in Sentence Embeddings using
Self-Debias`, carried out for the course Advanced Topics in Computational Semantics at the University of Amsterdam By Ekaterina Shutova, Alina Leidinger and Rochelle Choenni.
In this project we evaluate whether we can use the Self-Debiasing technique by Schick et al (20..) to mitigate implicit gender agency bias from pre-trained language models. 

Our code is mostly based on the code by Meade et al. (20..). The folder `bias-bench` contains this code, with some small adaptations to our own setup. 

Our experiments are carried out in the following notebooks:
* `prompt_experiments.ipnb` contains our experiments with prompts
* `gender_proxies.ipnb` contains our experiments with various gender proxies ...
* ...

To blank out gendered terms from the input text, we use code by Huang et al. which can be found here: https://github.com/tenghaohuang/uncover_implicit_bias.

The file `data_to_json.py` can be used to complete the blanked out ROCStories data with gendered continuations in order to form gender pairs to evaluate. This can be runned as follows :
`python data_to_json.py '[SETTING]' [ID]`
where `[SETTING]` is the name of the setting, which decides what gender proxy to use (e.g. `'tokenized'` to only use singly tokenized names) and `[ID]` the id of the current dataset. The output will be a file:
`gender_data_[SETTING]_[ID].json` in the same directory.

