import os
import torchfly
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
torchfly.set_random_seed(123)

import time
import tqdm
import logging
from apex import amp
from allennlp.training.util import create_serialization_dir
from allennlp.common import Params
from allennlp.common.util import prepare_global_logging
from allennlp.training.checkpointer import Checkpointer
from pytorch_transformers import BertTokenizer, AdamW, WarmupLinearSchedule
from model import TransformerModel, TransformerLMHeadModel
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import argparse

from allennlp.nn.util import sequence_cross_entropy_with_logits
from torchfly.modules.losses import SequenceFocalLoss, SequenceCrossEntropyLoss

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--alpha", type=float, help="SIA alpha")
parser.add_argument("-b", "--beta",  type=float, help="SIA alpha")
parser.add_argument("-o", "--output_dir", help="SIA alpha", default="results")
args = parser.parse_args()


# put it at the start of the file
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

create_serialization_dir(Params({"seed": 123}), args.output_dir, recover=False, force=True)
stdout_handler = prepare_global_logging(serialization_dir=args.output_dir, file_friendly_logging=False)
checkpointer = Checkpointer(args.output_dir, keep_serialized_model_every_num_seconds=3600)

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


class SIAHeadlineDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.dataset = data
        self.CLS = [101]
        self.SEP = [102]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        summary, origin_title, target_title = self.dataset[idx]
        origin_title = tokenizer.convert_tokens_to_ids(origin_title)
        target_title = tokenizer.convert_tokens_to_ids(target_title)

        # source and token type
        if np.random.rand() < 0.9:
            max_len = 509 - len(origin_title)
            if len(summary) > max_len:
                summary = summary[:max_len]
            summary = tokenizer.convert_tokens_to_ids(summary)
            source = self.CLS + summary + self.SEP + origin_title + self.SEP
            source_type_ids = [0] * (len(summary) + 2) + [1] * (len(origin_title) + 1)
        else:
            max_len = 510
            if len(summary) > max_len:
                summary = summary[:max_len]
            summary = tokenizer.convert_tokens_to_ids(summary)
            source = self.CLS + summary + self.SEP
            source_type_ids = [0] * len(source)

        target = self.CLS + target_title + self.SEP

        # turn into tensors
        source = torch.LongTensor(source)
        source_mask = torch.ones(source.shape[0])
        target = torch.LongTensor(target)
        target_mask = torch.ones(target.shape[0])
        source_type_ids = torch.LongTensor(source_type_ids)

        return source, source_mask, source_type_ids, target, target_mask


def mod_collate(batch):
    source, source_mask, source_type_ids, target, target_mask = zip(*batch)

    # pad sequence
    source = pad_sequence(source, batch_first=True)
    source_mask = pad_sequence(source_mask, batch_first=True)
    source_type_ids = pad_sequence(source_type_ids, batch_first=True)

    target = pad_sequence(target, batch_first=True)
    target_mask = pad_sequence(target_mask, batch_first=True)

    return source, source_mask, source_type_ids, target, target_mask


def train_one_iter(batch, fp16=False):
    batch = [item.to(device) for item in batch]

    source, source_mask, source_type_ids, target, target_mask = batch

    _, past = encoder(source, source_mask, source_type_ids)

    mask = torch.cat([source_mask, target_mask], dim=1)
    logits, _ = decoder(target, mask, past=past, past_length=0)

    out = logits[:, :-1].contiguous()
    target = target[:, 1:].contiguous()
    target_mask = target_mask[:, 1:].contiguous()

    loss = criterion(out, target, target_mask, label_smoothing=0.02, reduce=True)
    loss /= num_gradients_accumulation

    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)

    record_loss = loss.item() * num_gradients_accumulation
    perplexity = np.exp(record_loss)

    return record_loss, perplexity


def validate(dataloader):
    with torch.no_grad():
        pb = tqdm.tqdm(dataloader)
        encoder.eval()
        decoder.eval()

        total_ppl = []

        for batch in pb:
            batch = [item.to(device) for item in batch]

            source, source_mask, source_type_ids, target, target_mask = batch

            _, past = encoder(source, source_mask, source_type_ids)

            mask = torch.cat([source_mask, target_mask], dim=1)
            logits, _ = decoder(target, mask, past=past, past_length=0)

            out = logits[:, :-1].contiguous()
            target = target[:, 1:].contiguous()
            target_mask = target_mask[:, 1:].contiguous()

            loss = eval_criterion(out, target, target_mask, label_smoothing=-1, reduce="sentence")

            ppl = torch.exp(loss)
            total_ppl.extend(ppl.tolist())

    return np.mean(total_ppl)


train_data = torch.load("data/train_tokenized.pkl")
val_data = torch.load("data/val_tokenized.pkl")

batch_size = 5
train_dataset = SIAHeadlineDataset(train_data)
val_dataset = SIAHeadlineDataset(val_data)

train_dataloader = DataLoader(dataset=train_dataset, shuffle=True,
                              batch_size=batch_size, collate_fn=mod_collate, num_workers=16)
val_dataloader = DataLoader(dataset=val_dataset, shuffle=False,
                            batch_size=batch_size, collate_fn=mod_collate)


# alpha_beta = [(0.0, 0.0),
#               (-1.0, 0.0),
#               (-0.5, 0.0),
#               (0.5, 0.0),
#               (1.0, 0.0),
#               (0.0, -1.0),
#               (0.0, 1.0),
#               (0.0, 2.0),
#               (0.0, 5.0),
#               ]
a = args.alpha
b = args.beta

encoder = TransformerModel(unidirectional=False)
decoder = TransformerLMHeadModel()


logger.info(f"Start training of alpha={a} beta={b}")

states = torch.load("../TSP/TSP-best.th")
encoder.load_state_dict(states["encoder"])
decoder.load_state_dict(states["decoder"])

device = torch.device("cuda")

encoder = encoder.to(device)
decoder = decoder.to(device)

num_epochs = 10
num_gradients_accumulation = 1
num_train_optimization_steps = num_train_optimization_steps = len(train_dataset) * num_epochs // batch_size // num_gradients_accumulation

param_optimizer = list(encoder.named_parameters()) + list(decoder.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters,
                    lr=1e-5,
                    correct_bias=False)

scheduler = WarmupLinearSchedule(optimizer,
                                    warmup_steps=int(num_train_optimization_steps*0.1),
                                    t_total=num_train_optimization_steps)

[encoder, decoder], optimizer = amp.initialize([encoder, decoder], optimizer, opt_level='O1')

criterion = SequenceFocalLoss(gamma=a, beta=b)
eval_criterion = SequenceCrossEntropyLoss()

update_count = 0
start = time.time()

for ep in range(5):
    "Training"
    pb = tqdm.tqdm(train_dataloader)
    encoder.train()
    decoder.train()

    for batch in pb:
        record_loss, perplexity = train_one_iter(batch, fp16=True)
        update_count += 1

        if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

            # speed measure
            end = time.time()
            speed = batch_size * num_gradients_accumulation / (end - start)
            start = end

            pb.set_postfix(loss=record_loss, perplexity=perplexity, speed=speed)
        

    "Evaluation"
    encoder.eval()
    decoder.eval()
    ppl = validate(val_dataloader)
    checkpointer.save_checkpoint(str(ep),
                                    {"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
                                    {"empty": None},
                                    is_best_so_far=True)
    

    logger.info(f"a={a} b={b} Epoch {ep} Validation perplexity: {ppl}")

logger.info(f"Finish training of alpha={a} beta={b}")