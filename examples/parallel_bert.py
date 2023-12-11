import os
import argparse
import functools
import pandas as pd
import pathlib
import numpy as np
import urllib.request
import zipfile

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import BackwardPrefetch

from gdeep.search.hpo import GiottoSummaryWriter
from gdeep.data.datasets import FromArray, DataLoaderBuilder
from gdeep.trainer.trainer import Trainer, Parallelism
import gdeep.utility_examples.args

from sklearn.model_selection import train_test_split
from transformers import (
                          BertTokenizer,
                          BertForSequenceClassification,
                          )
from transformers.models.bert.modeling_bert import BertLayer


def download_dataset():
    if not pathlib.Path("cola_public").exists():
        req = urllib.request.urlretrieve("https://nyu-mll.github.io/CoLA/cola_public_1.1.zip")
        with zipfile.ZipFile(req[0], "r") as zip_ref:
            zip_ref.extractall()


def main(args):
    n_sentences_to_consider=4000

    tmp_path=os.path.join('./cola_public','raw','in_domain_train.tsv')
    df = pd.read_csv(tmp_path, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    # Get the lists of sentences and their labels.
    sentences = df.sentence.values
    labels = df.label.values

    sentences=sentences[0:n_sentences_to_consider]
    labels=labels[0:n_sentences_to_consider]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []

    for sent in sentences:

        encoded_sent = tokenizer.encode( sent, add_special_tokens = True)

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)

    # Print sentence 0, now as a list of IDs.
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])

    print('Max length: ', max([len(sen) for sen in input_ids]))

    MAX_LEN = 64
    def pad_sequences(
        sequences,
        maxlen=None,
        dtype="int32",
        padding="pre",
        truncating="pre",
        value=0.0,
    ):
        if not hasattr(sequences, "__len__"):
            raise ValueError("`sequences` must be iterable.")
        num_samples = len(sequences)

        lengths = []
        sample_shape = ()
        flag = True
        for x in sequences:
            try:
                lengths.append(len(x))
                if flag and len(x):
                    sample_shape = np.asarray(x).shape[1:]
                    flag = False
            except TypeError as e:
                raise ValueError(
                    "`sequences` must be a list of iterables. "
                    f"Found non-iterable: {str(x)}"
                ) from e

        if maxlen is None:
            maxlen = np.max(lengths)

        is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(
            dtype, np.unicode_
        )
        if isinstance(value, str) and dtype != object and not is_dtype_str:
            raise ValueError(
                f"`dtype` {dtype} is not compatible with `value`'s type: "
                f"{type(value)}\nYou should set `dtype=object` for variable length "
                "strings."
            )

        x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
        for idx, s in enumerate(sequences):
            if not len(s):
                continue  # empty list/array was found
            if truncating == "pre":
                trunc = s[-maxlen:]
            elif truncating == "post":
                trunc = s[:maxlen]
            else:
                raise ValueError(f'Truncating type "{truncating}" not understood')

            # check `trunc` has expected shape
            trunc = np.asarray(trunc, dtype=dtype)
            if trunc.shape[1:] != sample_shape:
                raise ValueError(
                    f"Shape of sample {trunc.shape[1:]} of sequence at "
                    f"position {idx} is different from expected shape "
                    f"{sample_shape}"
                )

            if padding == "post":
                x[idx, : len(trunc)] = trunc
            elif padding == "pre":
                x[idx, -len(trunc) :] = trunc
            else:
                raise ValueError(f'Padding type "{padding}" not understood')
        return x


    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                            value=0, truncating="post", padding="post")

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=13, test_size=0.1)

    dl_builder = DataLoaderBuilder((FromArray(train_inputs, train_labels), \
                                FromArray(validation_inputs, validation_labels)))

    dl_tr, dl_val, _ = dl_builder.build(({"batch_size": args.batch_size}, {"batch_size": args.batch_size}))

    if args.big_model:
        model = BertForSequenceClassification.from_pretrained(
            "bert-large-uncased",
            num_labels = 2,
            output_attentions = True,
            output_hidden_states = False
            )
    else:
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels = 2,
            output_attentions = True,
            output_hidden_states = False
            )

    # Define the trainer

    writer = GiottoSummaryWriter()
    loss_function = nn.CrossEntropyLoss()
    trainer = Trainer(model, (dl_tr, dl_val), loss_function, writer)
    devices = list(range(torch.cuda.device_count()))
    config_fsdp = {
        "sharding_strategy": args.sharding.to_sharding_strategy(),
        "auto_wrap_policy": functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={BertLayer,}),
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE
        }
    parallel = Parallelism(args.parallel,
                           devices,
                           len(devices),
                           config_fsdp=config_fsdp,
                           pipeline_chunks=2)

    # train the model

    return trainer.train(Adam, args.n_epochs, parallel=parallel)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BERT Example')
    gdeep.utility_examples.args.add_default_arguments(parser)
    gdeep.utility_examples.args.add_big_model(parser)
    parser.add_argument("--download",
                        action="store_true",
                        help="Download dataset if it does not exist already")
    args = parser.parse_args()
    if args.download:
        download_dataset()
    main(args)
