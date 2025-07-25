import os
import argparse
import re
import time
import json

from torch_xla import runtime as xr

xr.use_spmd()
assert xr.is_spmd() == True

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import functools

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.spmd as xs

from torch_xla.distributed.spmd import mark_sharding
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel as FSDPv2
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates 
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers import TextStreamer

from torch.utils.data import Dataset, DataLoader

import torch.distributed as dist

from tqdm import tqdm

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# os.environ["XLA_IR_DEBUG"] = "0"
# os.environ["XLA_HLO_DEBUG"] = "0"

# os.environ["XLA_SAVE_TENSORS_FMT"] = "hlo"
# os.environ["XLA_SAVE_TENSORS_FILE"] = "/tmp/save1.hlo"

# LlavaLlamaForCausalLM(
#   (model): LlavaLlamaModel(
#     (embed_tokens): Embedding(128261, 4096)
#     (layers): ModuleList(
#       (0-31): 32 x LlamaDecoderLayer(
#         (self_attn): LlamaAttention(
#           (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
#           (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
#           (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (rotary_emb): LlamaRotaryEmbedding()
#         )
#         (mlp): LlamaMLP(
#           (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
#           (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
#           (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
#           (act_fn): SiLU()
#         )
#         (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
#         (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
#       )
#     )
#     (norm): LlamaRMSNorm((4096,), eps=1e-05)
#     (rotary_emb): LlamaRotaryEmbedding()
#     (mm_projector): AttentionalPoolProjector(
#       (attn_pool): AttentionalPooler(
#         (ln_k): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#         (ln_q): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#         (to_q): Linear(in_features=512, out_features=512, bias=False)
#         (to_kv): Linear(in_features=512, out_features=1024, bias=False)
#         (to_out): Linear(in_features=512, out_features=512, bias=False)
#       )
#       (ln): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#       (proj): Sequential(
#         (0): Linear(in_features=512, out_features=4096, bias=True)
#         (1): GELU(approximate='none')
#         (2): Linear(in_features=4096, out_features=4096, bias=True)
#       )
#     )
#   )
#   (lm_head): Linear(in_features=4096, out_features=128261, bias=False)
# )

LLAVA_LLAMA_RULES = (
    # mp dim: 1 (no effect) - fsdp dim: divided to the 4 devices
    # (128261, 4096/8) = (128261, 512)
    ("model\\.embed_tokens", ("mp", "fsdp")),
    
    # fsdp dim: divided to the 4 devices - mp dim: 1 (no effect)
    # (4096/8, 4096) = (512, 4096)
    # minimizes communication during matrix multiplication
    ("self_attn\\.(q_proj|k_proj|v_proj)", ("fsdp", "mp")),
    
    # mp dim: 1 (no effect) - fsdp dim: divided to the 4 devices
    # (4096, 4096/8) = (4096, 512)
    # backward pass optimization
    ("self_attn\\.o_proj", ("mp", "fsdp")),
    
    # # fsdp dim: divided to the 4 devices - mp dim: 1 (no effect)
    # (4096/8, 14336) = (512, 14336)
    ("mlp\\.gate_proj", ("fsdp", "mp")),
    
    # mp dim: 1 (no effect) - fsdp dim: divided to the 4 devices
    # (4096, 4096/8) = (4096, 512)
    ("mlp\\.down_proj", ("mp", "fsdp")),
    
    # # fsdp dim: divided to the 4 devices - mp dim: 1 (no effect)
    # (14336, 4096/8) = (14336, 512)
    ("mlp\\.up_proj", ("fsdp", "mp")),
    
    # fsdp dim: divided to the 4 devices - mp dim: 1 (no effect)
    # (4096/8, 128261) = (512, 128261)
    ("lm_head", ("fsdp", "mp")),
    
    ("mm_projector\\.linear_1$", ("fsdp", "mp")),
    ("mm_projector\\.linear_2$", ("mp", "fsdp")),
)

def partition_module(model, mesh, device='xla', verbose=False):
    # partition_specs = find_rule(model)
    # rule = [(k, tuple([strkey2id.get(x) for x in v])) for k, v in partition_specs]
    partition_specs = LLAVA_LLAMA_RULES
        
    # print(rule)
    model.to(device)

    for name, module in (tqdm(model.named_modules(), desc="partitioning model", disable=not verbose, position=0)):
        if not hasattr(module, "weight") or not isinstance(module.weight, nn.Parameter):
            continue
        
        find = False
        # print(name, module.__class__.__name__)
        for rule_pattern, spec in partition_specs:
            if re.findall(rule_pattern, name):
                if verbose:
                    print("match", rule_pattern, name, spec)
                
                xs.mark_sharding(module.weight, mesh, spec)
                find = True
                break
            
        if not find:
            if verbose:
                print(f"no match {module}", name, module.weight.size(), module.weight.dim())
            xs.mark_sharding(module.weight, mesh, tuple([None] * module.weight.dim()))

class JsonDataset(Dataset):
    def __init__(self, json_path, tokenizer, image_processor, embeddings_path, max_length=512):
        with open(json_path, "r") as f:
            raw_data = json.load(f)
        # filter only report_generation
        self.entries = [e for e in raw_data
                        if e.get("conversations") and e["conversations"][0].get("type")=="report_generation"]
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.embeddings_path = embeddings_path
        self.max_length = max_length

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        # load npz embedding
        image_file = item['image'].replace('nii.gz', 'npz')
        npz_path = os.path.join(self.embeddings_path, image_file)
        arr = np.load(npz_path)['arr']
        image_tensor = torch.tensor(arr, dtype=torch.bfloat16)

        # build prompt
        conv = conv_templates['llama3'].copy()
        human_input = item['conversations'][0]['value']
        conv.append_message(conv.roles[0], human_input)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        return {
            'input_ids': input_ids,
            'image_tensor': image_tensor,
            'image_size': image_tensor.numel(),
            'image_file': image_file
        }
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/raspuntinov/gcs/ct_rate/models/ct_chat/llava-lora-llama_3.1_8b")
    parser.add_argument("--model_base", type=str, default="/home/raspuntinov/gcs/ct_rate/models/ct_chat/llama_3.1_8b_instrcut")
    parser.add_argument("--device", type=str, default="tpu")
    parser.add_argument("--conv_mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--drop_last", type=bool, default=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--auto_wrap_min_num_params", type=int, default=1e6)
    
    # Additional path parameters
    parser.add_argument("--output_file", type=str, 
                        default="/home/raspuntinov/gcs/ct_rate/outputs/output_validation_CTRATE_ChestReport_CT_CLIPllama_3_8b.json",
                        help="Output file path")
    parser.add_argument("--input_data", type=str, 
                        default="/home/raspuntinov/gcs/ct_rate/dataset/vqa/valid_vqa.json",
                        help="Input JSON file")
    parser.add_argument("--embeddings_path", type=str, default="/home/raspuntinov/gcs/ct_rate/dataset/embeddings")
    
    args = parser.parse_args()
    
    return args

# def shard_output(output, mesh):
#     xs.mark_sharding(output.logits, mesh, partition_spec=('fsdp', None, None))


def main(args):
    num_devices = xr.global_runtime_device_count()
    device_type = xr.device_type()
    device_ids = np.array(range(num_devices))
    
    print(f"{num_devices} {device_type} are using with device ids: {device_ids}")
    
    # dp (Data Parallel): 1 - No data parallelism
    # fsdp (Fully Sharded): 8 
    # mp (Model Parallel): 1 - No model parallelism
    # sp (Sequence Parallel): 1 - No sequence paralellism
    mesh_shape = (1, num_devices, 1)
    axis_names  = ("dp", "fsdp", "mp")
    mesh = xs.Mesh(device_ids, mesh_shape, axis_names)
    
    # args.batch_size = args.batch_size * num_devices
    device = torch_xla.device()
    
    model_name = get_model_name_from_path(args.model_path)
    
    print(f"Using model: {model_name}")
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, 
        args.model_base, 
        model_name, 
        args.load_8bit, 
        args.load_4bit, 
        device_map="auto", 
        device=str(device)
    )
    
    if xm.is_master_ordinal():
        print(model)
    
    dataset = JsonDataset(
        args.input_data,
        tokenizer,
        image_processor,
        args.embeddings_path,
        max_length=context_len
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=args.drop_last,
        num_workers=args.workers,
        collate_fn=lambda batch: {
            'input_ids': torch.nn.utils.rnn.pad_sequence(
                [b['input_ids'] for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id
            ),
            'image_tensor': torch.stack([b['image_tensor'] for b in batch]),
            'image_sizes': [b['image_size'] for b in batch],
            'image_files': [b['image_file'] for b in batch]
        }
    )
    device_loader = pl.MpDeviceLoader(
        loader,
        device,
        input_sharding={
            'input_ids': xs.ShardingSpec(mesh, ('fsdp', None)),
            'image_tensor': xs.ShardingSpec(mesh, ('fsdp', None, None, None))
        }
    )
    
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer
        },
    )
    
    # model = FSDPv2(model, mesh=mesh, shard_output=shard_output, auto_wrap_policy=auto_wrap_policy)
    partition_module(model, mesh, verbose=False)
    model = model.to(dtype=torch.bfloat16)
    
    model.eval()
    xm.master_print('Evaluation begin {}'.format(time.strftime('%l:%M%p %Z on %b %d, %Y')))
    
    
    results = []
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        images = batch['image_tensor'].to(device).squeeze(0)
        
        print(f"input_ids: {input_ids.shape}")
        print(f"images: {images.shape}")
        
        files = batch['image_files']
        sizes = batch['image_sizes']

        conv_mode = args.conv_mode or 'llama3'
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.no_grad(), torch.amp.autocast('xla', dtype=torch.bfloat16):
            outputs = model.generate(
                input_ids,
                images=images,
                image_sizes=sizes,
                do_sample=(args.temperature>0),
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True
            )
        xm.mark_step()

        for idx, seq in enumerate(outputs):
            txt = tokenizer.decode(seq, skip_special_tokens=True).strip()
            results.append({'image': files[idx], 'answer': txt})
            if args.debug: print(f"Image {files[idx]} -> {txt}\n")

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
    xm.master_print(f"Saved results to {args.output_file}")
        
    xm.master_print('Evaluation end {}'.format(time.strftime('%l:%M%p %Z on %b %d, %Y')))
    xm.wait_device_ops()


if __name__ == "__main__":
    args = parse_args()
    main(args)
