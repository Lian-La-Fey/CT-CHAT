import os
import argparse
import time
import itertools
import random
import json

import torch
import numpy as np
import pandas as pd
import functools

import torch_xla
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.spmd as xs

from torch_xla import runtime as xr
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel as FSDPv2
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
from torch_xla.distributed.fsdp.wrap import checkpoint_wrapper

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers import TextStreamer

from torch.utils.data import Dataset

class JsonDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as file:
            self.data = json.load(file)[:32]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/raspuntinov/gcs/ct_rate/models/ct_chat/llama_3.1_8b")
    parser.add_argument("--model_base", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--device", type=str, default="tpu")
    parser.add_argument("--conv_mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--drop_last", type=bool, default=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--auto_wrap_min_num_params", type=int, default=1e6)
    
    # Additional path parameters
    parser.add_argument("--output_file", type=str, 
                        default="./output/output_validation_CTRATE_ChestReport_CT_CLIPllama_3_8b.json",
                        help="Output file path")
    parser.add_argument("--input_data", type=str, 
                        default="/home/raspuntinov/gcs/ct_rate/dataset/vqa/valid_vqa.json",
                        help="Input JSON file")
    parser.add_argument("--embeddings_path", type=str, default="/home/raspuntinov/gcs/ct_rate/dataset/embeddings")
    
    args = parser.parse_args()
    
    return args

def main(args):
    num_devices = xr.global_runtime_device_count()
    device_type = xr.device_type()
    device_ids = np.array(range(num_devices))
    
    print(f"{num_devices} {device_type} are using with device ids: {device_ids}")
    
    mesh_shape = (num_devices, 1)
    mesh = xs.Mesh(device_ids, mesh_shape, ('fsdp', 'model'))
    xs.set_global_mesh(mesh)
    
    batch_size = args.batch_size * num_devices
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
    
    dataset = JsonDataset(args.input_data)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=args.drop_last,
        shuffle=False,
        num_workers=args.workers
    )
    
    device_loader = pl.MpDeviceLoader(
        loader,
        device,
        # Shard the input's batch dimension along the `fsdp` axis, no sharding along other dimensions
        input_sharding=xs.ShardingSpec(mesh, ('fsdp', None))
    )
    
    # print(f"Model: {model}")
    # print(model.model.layers[0])
    
    # auto_wrap_policy = functools.partial(
    #     transformer_auto_wrap_policy,
    #     transformer_layer_cls={
    #         LlamaDecoderLayer
    #     },
    # )
    
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=args.auto_wrap_min_num_params
    )
    
    model = FSDPv2(model, auto_wrap_policy=auto_wrap_policy)
    
    model.eval()
    xm.master_print('Evaluation begin {}'.format(time.strftime('%l:%M%p %Z on %b %d, %Y')))
    
    with open(args.input_data, 'r') as file:
        data_val = json.load(file)
        
    data_val = data_val[-32:]
    
    output_save = []
    for element in data_val:
        
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llama3"
        conv_mode = "llama3"
        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        image_file = element["image"].replace("nii.gz", "npz")
        image_path = f"{args.embeddings_path}/{image_file}"
        
        image = np.load(image_path)["arr"]
        image_size = image.size
        # Similar operation in model_worker.py
        #image_tensor = process_images([image], image_processor, model.config)
        
        image_tensor = torch.tensor(image).to(device, dtype=torch.bfloat16)  # BF16 for TPU
        
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        conversations_save = []
        for conversation in element["conversations"]:
            i = 0
            
            if conversation.get("type") != "report_generation":
                continue
            
            print(conversation)
            
            if conversation["from"] == "human":
                inp = conversation["value"]
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)

                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
                ).unsqueeze(0).to(model.device)

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                
                streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                
                
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids.to(device),
                        images=image_tensor,
                        image_sizes=[image_size],
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        streamer=streamer,
                        use_cache=False
                    )
                    
                
                print(f"Generated output for image {image_file}: {output_ids}")
                outputs = tokenizer.decode(output_ids[0]).strip()
                conv.messages[-1][-1] = outputs

                conversations_save.append({"question": inp, "answer":outputs})

            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        output_save.append({"image": image_file, "conversations_out": conversations_save})
    
    output_path = f"{args.output_file}.json"
    with open(output_path, "w") as json_file:
        json.dump(output_save, json_file, indent=4)
    
    xm.master_print('Evaluation end {}'.format(time.strftime('%l:%M%p %Z on %b %d, %Y')))
    xm.wait_device_ops()


if __name__ == "__main__":
    xr.use_spmd()
    assert xr.is_spmd() == True
    
    args = parse_args()
    main(args)
