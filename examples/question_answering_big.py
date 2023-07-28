import copy
import argparse

from torch.nn import Transformer
from torch.optim import Adam, SparseAdam, SGD
import numpy as np
from gtda.diagrams import BettiCurve
from gtda.plotting import plot_betti_surfaces
import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler

from gdeep.models import FFNet
from gdeep.visualization import persistence_diagrams_of_activations
from gdeep.data.datasets import DatasetBuilder
from gdeep.trainer import Trainer
from gdeep.models import ModelExtractor
from gdeep.utility import DEVICE
from gdeep.data import PreprocessingPipeline
from gdeep.data import TransformingDataset
from gdeep.data.preprocessors import Normalization, TokenizerQA
from gdeep.data.datasets import DataLoaderBuilder
from gdeep.visualization import Visualiser
from gdeep.search import GiottoSummaryWriter

from gdeep.trainer.trainer import Parallelism, ParallelismType

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Giotto FSDP Example')
    parser.add_argument('--fsdp', action='store_true', default=False,
                            help='Enable FSDP for training')
    parser.add_argument('--layer_cls', action='store_true', default=False,
                            help='Use layer class')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--n_epochs', type=int, default=1, metavar='N',
                        help='Number of epochs to train for (default: 1)')
    args = parser.parse_args()

    writer = GiottoSummaryWriter()

    bd = DatasetBuilder(name="SQuAD2", convert_to_map_dataset=False)
    ds_tr_str, ds_val_str, ds_ts_str = bd.build()

    #print("Before preprocessing: \n", ds_tr_str[0])


    tokenizer = TokenizerQA()

    # in case you need to combine multiple preprocessing:
    # ppp = PreprocessingPipeline(((PreprocessTextData(), IdentityTransform(), TextDataset),
    #                             (Normalisation(), IdentityTransform(), BasicDataset)))

    tokenizer.fit_to_dataset(ds_tr_str)
    transformed_textds = tokenizer.attach_transform_to_dataset(ds_tr_str)

    transformed_textts = tokenizer.attach_transform_to_dataset(
        ds_val_str
    )  # this has been fitted on the train set!

    #print("After the preprocessing: \n", transformed_textds[0])

    # the only part of the training/test set we are interested in
    train_indices = list(range(64 * 2))
    test_indices = list(range(64 * 1))
    #train_indices = list(range(int(len(transformed_textds)/10)))
    #test_indices = list(range(int(len(transformed_textts)/10)))

    dl_tr2, dl_ts2, _ = DataLoaderBuilder((transformed_textds, transformed_textts)).build(
        (
            {"batch_size": args.batch_size, "sampler": SubsetRandomSampler(train_indices)},
            {"batch_size": 16, "sampler": SubsetRandomSampler(test_indices)},
        )
    )

    # my simple transformer model
    class QATransformer(nn.Module):
        def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim):
            super(QATransformer, self).__init__()
            self.transformer = Transformer(
                d_model=embed_dim,
                nhead=8,
                num_encoder_layers=6,
                num_decoder_layers=6,
                dim_feedforward=512,
                dropout=0.1,
            )
            self.embedding_src = nn.Embedding(src_vocab_size, embed_dim, sparse=True)
            self.embedding_tgt = nn.Embedding(tgt_vocab_size, embed_dim, sparse=True)
            self.generator = nn.Linear(embed_dim, 2)

        def forward(self, ctx, qst):
            # print(src.shape, tgt.shape)
            ctx_emb = self.embedding_src(ctx).permute(1, 0, 2)
            qst_emb = self.embedding_tgt(qst).permute(1, 0, 2)
            # print(src_emb.shape, tgt_emb.shape)
            self.outs = self.transformer(qst_emb, ctx_emb).permute(1, 0, 2)
            # print(outs.shape)
            logits = self.generator(self.outs)
            return logits

        def __deepcopy__(self, memo):
            """this is needed to make sure that the 
            non-leaf nodes do not
            interfere with copy.deepcopy()
            """
            cls = self.__class__
            result = cls.__new__(cls)
            memo[id(self)] = result
            for k, v in self.__dict__.items():
                setattr(result, k, copy.deepcopy(v, memo))
            return result

        def encode(self, src, src_mask):
            """this method is used only at the inference step"""
            return self.transformer.encoder(self.embedding_src(src), src_mask)

        def decode(self, tgt, memory, tgt_mask):
            """this method is used only at the inference step"""
            return self.transformer.decoder(self.embedding_tgt(tgt), memory, tgt_mask)

    src_vocab_size = len(tokenizer.vocabulary)
    tgt_vocab_size = len(tokenizer.vocabulary)
    emb_size = 2048

    model = QATransformer(src_vocab_size, tgt_vocab_size, emb_size)
    print(model)

    def loss_fn(output_of_network, label_of_dataloader):
        # print(output_of_network.shape, label_of_dataloader.shape)
        tgt_out = label_of_dataloader
        logits = output_of_network
        cel = nn.CrossEntropyLoss()
        return cel(logits, tgt_out)

    # prepare a pipeline class with the model, dataloaders loss_fn and tensorboard writer
    pipe = Trainer(model, (dl_tr2, dl_ts2), loss_fn, writer)

    # train the model
    devices = list(range(torch.cuda.device_count()))
    parallel = Parallelism(ParallelismType.FSDP_ZERO2,
                        devices,
                        len(devices),
                        transformer_layer_class=nn.TransformerEncoder if args.layer_cls else None)
    pipe.train(SGD, args.n_epochs, False, {"lr": 0.01}, {"batch_size": 16}, parallel=parallel if args.fsdp else None)

    bb = next(iter(ds_val_str))
    bb[:2]

    # get vocabulary and tokenizer
    voc = tokenizer.vocabulary
    context = tokenizer.tokenizer(bb[0])
    question = tokenizer.tokenizer(bb[1])

    # get the indexes in the vocabulary of the tokens
    context_idx = torch.tensor(list(map(voc.__getitem__, context)))
    question_idx = torch.tensor(list(map(voc.__getitem__, question)))

    aa = next(iter(dl_tr2))
    pad_fn = lambda length_to_pad, item: torch.cat(
        [item, tokenizer.pad_item * torch.ones(length_to_pad - item.shape[0])]
    ).to(torch.long)

    # these tensors are ready to be fitted into the model
    length_to_pad = aa[0][0].shape[-1]  # context length
    context_ready_for_model = pad_fn(length_to_pad, context_idx)
    length_to_pad = aa[0][1].shape[-1]  # question length
    question_ready_for_model = pad_fn(length_to_pad, question_idx)

    input_list = [context_ready_for_model.view(1, -1).to(DEVICE), 
                question_ready_for_model.view(1,-1).to(DEVICE)]

    out = pipe.model(*input_list)

    answer_idx = torch.argmax(out, dim=1)

    # simple code to convert the model's answer into words
    try:
        if answer_idx[0][1] > answer_idx[0][0]:
            print(
                "The model proposes: '",
                " ".join(context[answer_idx[0][0] : answer_idx[0][1]]),
                "...'",
            )
        else:
            print("The model proposes: '", context[answer_idx[0][0]], "...'")
    except IndexError:
        print("The model was not able to find the answer.")
    print("The actual answer was: '" + bb[2][0] + "'")

    exit()