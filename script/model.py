import argparse
import os

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import pandas as pd
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from kogpt2.utils import get_tokenizer
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from dataloader import ChatDataset

class KoGPT2Chat(pl.LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.hparams = hparams
        self.tok_path = get_tokenizer()
        self.neg = -1e18
        self.kogpt2, self.vocab = get_pytorch_kogpt2_model()
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',
                            type=int,
                            default=32,
                            help='max sentence length on input (default: 32)')

        parser.add_argument('--batch-size',
                            type=int,
                            default=96,
                            help='batch size for training (default: 96)')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        parser.add_argument('--train_file',
                            type=str,
                            default="ChatbotData.csv",
                            help="train file path")
        return parser

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output = self.kogpt2(inputs)[0]
        return output

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        tensorboard_logs = {'train_loss': loss_avg}
        return {'loss': loss_avg, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        data = pd.read_csv(self.hparams.train_file)
        self.train_set = ChatDataset(data, self.tok_path, self.vocab, max_len=self.hparams.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=5,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader

    def chat(self):
        tok = SentencepieceTokenizer(self.tok_path, num_best=0, alpha=0)
        with torch.no_grad():
            while 1:
                q = input('Q: ').strip()
                if q == 'quit':
                    break
                q_tok = tok(q)

                input_ids = torch.LongTensor(
                        [self.vocab['<usr>']] + self.vocab[q_tok] +
                        self.vocab['</s>', '<sys>']
                        ).unsqueeze(dim=0)
                
                gen = self.kogpt2.generate(input_ids,
                                           num_beams=5,
                                           max_length=self.hparams.max_len,
                                           no_repeat_ngram_size=2,
                                           bad_words_ids=[[47437]]
                                           )
                gen = self.vocab.to_tokens(gen.squeeze().tolist())
                
                answer = ''.join(g for g in gen)
                answer = answer[answer.find('<sys>')+5:]
                answer = answer[:answer.find('</s>')]
                answer = answer.replace('▁', ' ')

                print("A: {}".format(answer.strip()))