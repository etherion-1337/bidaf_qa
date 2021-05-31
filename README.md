# Question Answering with BiDAF

## Introduction

The development of natural language processing (NLP) has come a long way. It started off humble and slowly progressed into sequence modelling like GRU and LSTM. In the landmark paper [Attention Is All You Need][1] the transformer architecture was introduced and soon after there is a significant paradigm shift with models like [BERT][2] and the rest is history. However the self-attention mechanism used in transformer is not a new idea. It has been discussed in numerous prior work such as the [BiDAF model][3].                 

We attempt to build a simple Question Answering (QA) engine with [AllenNLP][4] library.               

## Repository Structure

```
.
├── allennlp
│   ├── AllenNLP_qa.ipynb                        # nb for inference demo using the allennlp lib for bidaf
│   └── AllenNLP.md                              # short intro on allennlp
├── bidaf-model-2020.02.10-charpad               # bidaf model folder
│   ├── data                                     # train/test data
│   │   ├── excel_to_squad.py                    # util func to convert excel into SQuAD dataset format (needed for allennlp)
│   │   ├── train.json                           # train data in SQuAD format (JSON)
│   │   └── test.json                            # test data in SQuAD format
│   ├── glove                                    # glove embed for bidaf
│   │   └── glove.6B.100d.txt.gz                 
│   ├── vocabulary
│   │   ├── non_padded_namespaces.txt  
│   │   └── tokens.txt
│   ├── config.json                              # bidaf architecture and train/test file path (called during finetune)
│   ├── fine_tune.sh                             # shell for fine-tuning bidaf with custom dataset
│   ├── train.sh                                 # shell for training bidaf from scratch with custom dataset
│   └── weights.th                               # bidaf weights
├── src                             
│   ├── app_file.py                              # streamlit app (QA for files)
│   └── app.py                                   # streamlit app (QA for direct input)
├── weights                                      # reserved for fine-tune/trained weights
├── Dockerfile
├── requirements.txt
└── README.md
```
**Note that due to the file size limitation of Github, `bidaf-model-2020.02.10-charpad ` folder are not included, Contact author for more details**   


## Usage (After acquiring `weights` and `data`)               

1. Quick Start             

Navigate to the `src` folder and run the following in the Terminal to launch the App:

```
streamlit run app.py
```
You can input the context and question and see how the model performs.

2. Fine-tuning BiDAF         

Navigate to the `bidaf-model-2020.02.10-charpad` folder and run the following shell script in the Terminal:

```
bash fine_tune.sh
```
It will save the fine-tuned weights in the `weights` folder. Make sure `train_data_path` and `validation_data_path` is set properly in `config.json`.

3. Train BiDAF from scratch        

Navigate to the `bidaf-model-2020.02.10-charpad` folder and run the following shell script in the Terminal:

```
bash train.sh
```
It will save the trained weights in the `weights` folder. Make sure `train_data_path` and `validation_data_path` is set properly in `config.json`.

## Short Demo

https://user-images.githubusercontent.com/46531622/120223193-cbbb2800-c273-11eb-9d10-ff90d11ca956.mov


## TODO         
~~1. Re-organise folders~~                  
~~2. Redesign Front end~~                 
~~3. Dockerize the App~~                
4. Deploy on AWS EC2            

## Author

Zk Xav



[1]: https://arxiv.org/abs/1706.03762
[2]: https://arxiv.org/abs/1810.04805
[3]: https://arxiv.org/abs/1611.01603
[4]: https://allennlp.org/
