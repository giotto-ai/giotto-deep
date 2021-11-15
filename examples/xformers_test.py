# %%
! pip install xformers pytorch_lightning
# %%
import math
from enum import Enum

import pytorch_lightning as pl
import torch
from torch import nn

from xformers.factory import xFormer, xFormerConfig

from torch.utils.tensorboard import SummaryWriter  # type: ignore

from gdeep.data import OrbitsGenerator, DataLoaderKwargs
from gdeep.pipeline import Pipeline

# %%

class Classifier(str, Enum):
    GAP = "gap"
    TOKEN = "token"

class SetTransformer(pl.LightningModule):
    def __init__(
        self,
        steps,
        learning_rate=1e-2,
        weight_decay=0.0001,
        image_size=32,
        num_classes=10,
        patch_size=4,
        dim=256,
        n_layer=12,
        n_head=8,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        mlp_pdrop=0.1,
        attention="scaled_dot_product",
        hidden_layer_multiplier=4,
        linear_warmup_ratio=0.05,
        seq_len=1_000,
        classifier: Classifier = Classifier.GAP,
    ):

        super().__init__()

        # all the inputs are saved under self.hparams (hyperparams)
        self.save_hyperparameters()

        # A list of the encoder or decoder blocks which constitute the Transformer.
        xformer_config = [
            {
                "block_config": {
                    "block_type": "encoder",
                    "num_layers": n_layer,
                    "dim_model": dim,
                    "seq_len": seq_len,
                    "layer_norm_style": "pre",
                    "multi_head_config": {
                        "num_heads": n_head,
                        "residual_dropout": resid_pdrop,
                        "use_rotary_embeddings": False,
                        "attention": {
                            "name": attention,
                            "dropout": attn_pdrop,
                            "causal": False,
                        },
                    },
                    "feedforward_config": {
                        "name": "MLP",
                        "dropout": mlp_pdrop,
                        "activation": "gelu",
                        "hidden_layer_multiplier": hidden_layer_multiplier,
                    },
                }
            }
        ]

        config = xFormerConfig(xformer_config)
        self.transformer = xFormer.from_config(config)

        self.patch_emb = nn.Linear(2, dim)

        if classifier == Classifier.TOKEN:
            self.clf_token = nn.Parameter(torch.zeros(dim))

        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

    @staticmethod
    def linear_warmup_cosine_decay(warmup_steps, total_steps):
        """
        Linear warmup for warmup_steps, with cosine annealing to 0 at total_steps
        """

        def fn(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return fn

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay,
        )

        warmup_steps = int(self.hparams.linear_warmup_ratio * self.hparams.steps)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                self.linear_warmup_cosine_decay(warmup_steps, self.hparams.steps),
            ),
            "interval": "step",
        }

        return [optimizer], [scheduler]

    def forward(self, x):
        batch, *_ = x.shape  # BCHW

        x = self.patch_emb(x)

        # flatten patches into sequence
        #x = x.flatten(2, 3).transpose(1, 2).contiguous()  # B HW C

        if self.hparams.classifier == Classifier.TOKEN:
            # prepend classification token
            clf_token = (
                torch.ones(1, batch, self.hparams.dim, device=x.device) * self.clf_token
            )
            x = torch.cat([clf_token, x[:-1, :, :]], axis=0)

        x = self.transformer(x)
        x = self.ln(x)

        if self.hparams.classifier == Classifier.TOKEN:
            x = x[:, 0]
        elif self.hparams.classifier == Classifier.GAP:
            x = x.mean(dim=1)  # mean over sequence len

        x = self.head(x)
        return x
# %%
model = SetTransformer(steps=100,
                       num_classes=5,
                       dim=64,
                       n_layer=2,
                       n_head=4,
                       learning_rate=1e-5,
                       attention="scaled_dot_product").double()

# %%
homology_dimensions = (0, 1)

dataloaders_dicts = DataLoaderKwargs(train_kwargs = {"batch_size": 16},
                                     val_kwargs = {"batch_size": 4},
                                     test_kwargs = {"batch_size": 3})

og = OrbitsGenerator(num_orbits_per_class=1_000,
                     homology_dimensions = homology_dimensions,
                     validation_percentage=0.0,
                     test_percentage=0.0,
                     n_jobs=2
                     #dynamical_system = 'pp_convention'
                     )

dl_train, _, _ = og.get_dataloader_orbits(dataloaders_dicts)

# %%

loss_fn = model.criterion

# Initialize the Tensorflow writer
writer = SummaryWriter()

# initialise pipeline class
pipe = Pipeline(model, [dl_train, None], loss_fn, writer)
# %%
optimizer_list, scheduler_list = model.configure_optimizers()

# %%
warmup_steps = int(model.hparams.linear_warmup_ratio * model.hparams.steps)

print('warmup steps:', warmup_steps)

# train the model
pipe.train(torch.optim.SGD, model.hparams.steps, cross_validation=False,
        optimizers_param={"lr": model.hparams.learning_rate,
                          "momentum": 0.9,
                          "weight_decay":model.hparams.weight_decay,},
                          lr_scheduler=torch.optim.lr_scheduler.LambdaLR,
                          scheduler_params={'lr_lambda': model.linear_warmup_cosine_decay(warmup_steps, model.hparams.steps)}
                          )
# %%