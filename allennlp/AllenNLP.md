# AllenNLP Library Overview

## Introduction
AllenNLP is a deep learning library that is developed by Allen Institute for AI that is specialised to handle NLP task. The package is installed simply via:
```
pip install allennlp
```

A great [tutorial][1] to get an overview of workflow can be found at their website. In essence, we are not expecting to write model and data pipeline from scratch with AllenNLP. This document will briefly walk through the important step in using the library.  


## General Workflow

A very (gentle) introduction can be found [here][2] where it walk you through the major steps in using the libaray in the context of sentiment analysis with IMDB dataset. There are three major components in using the libaray: DatasetReader, Model and config files.

### DatasetReader

The [DatasetReader][3] is the main class that is being used to prepare data. In short, this class will try to process data in the form of instances (i.e. each data entry/row is an instance) while the features are the attributes (they call these "fields") of the instance. There are a few methods in this class which can be modified (by inheriting the DatasetReader and create ur own reader) in order to suit your own needs, but most commonly the "read" and "text_to_instance" of the DatasetReader gets modified, as we can see from the example [here][4]. These two methods modifies the way the class tokenise and put each attribute into different fields of the instance.

When defining our own DatasetReader (or model) it is important to note that AllenNLP uses decorator to "register" the object. For example the new ``ImdbDatasetReader`` is assigned the name ``imdb``. This is important because it enables us to control it by config files by calling it.

```
@DatasetReader.register('imdb')
ImdbDatasetReader(DatasetReaer):
  def __init__(self, token_indexers, tokenizer):
    self._tokenizer = tokenizer
    self._token_indexers = token_indexers
```

The library also has a plethora of pre-defined DatasetReader (which also inherits from DatasetReader class), for example the [SquadReader][5] can directly process SQuAD data (JSON).

### Model

The details of the model implementation can be found in the [tutorial][2] above. In short the formulation follows largely from PyTorch. We should inherit the ``Model`` class to make our own, and them implement the three methods: ``__init__`` ,``forward`` and ``get_metrics``. 

### Config files

This is perhaps the most important step, especially for using pre-trianed model (architecture and weight). A very generic config file (json or jsonnet format) has the form of:

```
{
  "dataset_reader": {...},
  "model": {...},
  "trainer": {...}
}
```

The ``dataset_reader`` consist of paramters we wish for our pre-defined ``DatasetReader`` class (using the ``imdb`` name specified): 

```
"dataset_reader": {
  "type": "imdb",
  "token_indexers": {
    "tokens": {
      "type": "single_id"
    }
  },
  "tokenizer": {
    "type": "word"
  }
}
```

For models we can either define our own or use the pre-defined models in the library ([API][6] and [source][7]). 

## Training

To train a pre-defined model, simple use one of the pre-defined model structure (the config file) which specifiy the architecture and path for necessary embedding/train/dev data. A Bidaf model follwing [Seo et al. (2016)][8] is given below:

```
{
    "dataset_reader": {
        "type": "squad",
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            },
            "token_characters": {
                "type": "characters",
                "character_tokenizer": {
                    "byte_encoding": "utf-8",
                    "end_tokens": [
                        260
                    ],
                    "start_tokens": [
                        259
                    ]
                }
            },
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 40,
        "sorting_keys": [
            [
                "passage",
                "num_tokens"
            ],
            [
                "question",
                "num_tokens"
            ]
        ]
    },
    "model": {
        "type": "bidaf",
        "dropout": 0.2,
        "modeling_layer": {
            "type": "lstm",
            "bidirectional": true,
            "dropout": 0.2,
            "hidden_size": 100,
            "input_size": 800,
            "num_layers": 2
        },
        "num_highway_layers": 2,
        "phrase_layer": {
            "type": "lstm",
            "bidirectional": true,
            "dropout": 0.2,
            "hidden_size": 100,
            "input_size": 1224,
            "num_layers": 1
        },
        "similarity_function": {
            "type": "linear",
            "combination": "x,y,x*y",
            "tensor_1_dim": 200,
            "tensor_2_dim": 200
        },
        "span_end_encoder": {
            "type": "lstm",
            "bidirectional": true,
            "dropout": 0.2,
            "hidden_size": 100,
            "input_size": 1400,
            "num_layers": 1
        },
        "text_field_embedder": {
            "token_embedders": {
                "elmo": {
                    "type": "elmo_token_embedder",
                    "do_layer_norm": false,
                    "dropout": 0,
                    "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                    "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
                },
                "token_characters": {
                    "type": "character_encoding",
                    "dropout": 0.2,
                    "embedding": {
                        "embedding_dim": 16,
                        "num_embeddings": 262
                    },
                    "encoder": {
                        "type": "cnn",
                        "embedding_dim": 16,
                        "ngram_filter_sizes": [
                            5
                        ],
                        "num_filters": 100
                    }
                },
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 100,
                    "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
                    "trainable": false
                }
            }
        }
    },
    "train_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-train-v1.1.json",
    "validation_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-dev-v1.1.json",
    "trainer": {
        "cuda_device": 0,
        "grad_norm": 5,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 2
        },
        "num_epochs": 20,
        "optimizer": {
            "type": "adam",
            "betas": [
                0.9,
                0.9
            ]
        },
        "patience": 10,
        "validation_metric": "+em"
    }
}
```

The package that contains all the data necessary for training can be downloaded from this [link][9]. 

To train the model we simply run:

```
allennlp train config.json -s output_path
```

The output will consist of a folder where the trained weights for each epoch (and the best one), metrics, loggings and vocab can be found.

## Re-training

To retrain, or find-tune an existing model, we simply run:

```
allennlp fine-tune -m current_model_folder -c config.json -s output_path --extend-vocab
```

where ``current_model_folder`` contains the current model's config.json file, previosuly trained weights and vocab (token) from the previous dataset. The ``config.json`` in the command specifies the data path for the new data to be trained (the architecture in this config file will be ignored as the ``fine-tune`` command will use the architecture in the old model).

## Inference

The inference is demonstrated in the ``AllenNLP_qa.ipynb``


## Bonus: Excel to Squad v1.1 JSON format

The notebook ``AllenNLP_qa.ipynb`` also demonstrated a module that can convert a xlsx file into a format same as SQuAD v1.1 which can be used for ``SquadReader`` class in the library.

The xlsx has to have the format of :

| title | context | text | question |
| :----- | :------ | :----- | :------ |
| Super_Bowl_50 | Super Bowl 50 was an American football game .. | Denver Broncos | Which NFL team represented the AFC at Super Bowl 50? |

Each title (theme) can have multiple context (paragraph), each context can have multiple question, each question can have multiple answers.


## Author

Xavier

[1]: https://allennlp.org/tutorials
[2]: https://www.kdnuggets.com/2019/07/gentle-guide-starting-nlp-project-allennlp.html
[3]: http://docs.allennlp.org/0.9.0/api/allennlp.data.dataset_readers.dataset_reader.html
[4]: https://github.com/yasufumy/allennlp_imdb/blob/master/allennlp_imdb/data/dataset_readers/imdb.py#L22
[5]: http://docs.allennlp.org/0.9.0/api/allennlp.data.dataset_readers.reading_comprehension.html
[6]: http://docs.allennlp.org/0.9.0/api/allennlp.models.html
[7]: https://github.com/allenai/allennlp-models/tree/master/allennlp_models
[8]: https://arxiv.org/abs/1611.01603
[9]: https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2018.11.30-charpad.tar.gz