import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import logging
import torchfly
from torchfly.text.decode import top_k_top_p_filtering
from typing import List, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchfly.modules.transformers import CachedBertDecoderLM, ChineseBERTBaseConfig
from torchfly.text.tokenizers import BertTokenizer
from torchfly.utils import get_pretrained_states

from generate import TransformerDecodingHelper

# pylint: disable=no-member

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model_states = get_pretrained_states("chinese-gpt-bert-small")

model = CachedBertDecoderLM(ChineseBERTBaseConfig)
model.load_state_dict(model_states, strict=False)

device = torch.device("cuda")
model = model.to(device)
decoding_helper = TransformerDecodingHelper(model, device)
decoding_helper.config.batch_size = 2

start_input_ids = [torch.LongTensor(tokenizer.encode("中文")), torch.LongTensor(tokenizer.encode("最好的是什么"))]

# prepare the mask for the inputs of Transformer
states = {}
states["mask"] = [torch.ones(len(start_input_ids[0])), torch.ones(len(start_input_ids[1]))]
states["past"] = None
states["position_ids"] = [torch.arange(len(start_input_ids[0])), torch.arange(len(start_input_ids[1]))]

start_input_ids = pad_sequence(start_input_ids, batch_first=True).to(device)
states["mask"] = pad_sequence(states["mask"], batch_first=True).bool().to(device)
states["position_ids"] = pad_sequence(states["position_ids"], batch_first=True).to(device)

results = decoding_helper._generate_no_beam_search(start_input_ids, states)

breakpoint()
tokenizer.decode(results["tokens"][0])
pass