import os
import gc
import argparse
import torch
import torch.nn as nn
import json

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import numpy as np
import tqdm

from torch.cpu.amp import autocast
from functools import partial

################ For TPU support ################

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch.distributed as dist

from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP, checkpoint_module
from torch_xla.distributed.fsdp.wrap import (size_based_auto_wrap_policy,
                                             transformer_auto_wrap_policy)

import torch_xla.distributed.fsdp as fsdp
import torch_xla.distributed.fsdp as xla_fsdp

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from llava.model.multimodal_projector.coca_attentional_pooler import AttentionalPoolProjector, AttentionalPooler


################ For TPU support ################

############ Bad Exhauste error #################

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

################################################

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def validation(rank, world_size, args):
    xm.master_print(f"World Size: {world_size}")
    
    device = xm.xla_device()
    # xm.set_rng_state(0, device=device)  # Determinizm
    
    args.load_8bit = False
    args.load_4bit = False

    model_name = get_model_name_from_path(args.model_path)
    
    print(f"Rank {rank} using model: {model_name}")
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, 
        args.model_base, 
        model_name, 
        args.load_8bit, 
        args.load_4bit, 
        device_map="auto", 
        device=str(device)  # TPU
    )
    
    if xm.is_master_ordinal():
        print(model)
    
    print("Before model fsdp_wrap:", model.model.mm_projector.attn_pool.query.shape)
    
    ####################### FSDP SETUP #######################   
    model = model.to(torch.float32)
        
    # auto_wrap_policy = partial(
    #     size_based_auto_wrap_policy,
    #     min_num_params=1e6,
    # )
    
    # auto_wrap_policy = partial(
    #     transformer_auto_wrap_policy,
    #     transformer_layer_cls={
    #         LlamaDecoderLayer
    #     },
    # )
    
    # auto_wrapper_callable = lambda m, *args, **kwargs: FSDP(checkpoint_module(m), *args, **kwargs)
    # grad_ckpt_wrap = checkpoint_module if args.use_gradient_checkpointing else (lambda x: x)
    
    # fsdp_wrap = lambda m: FSDP(
    #     m,
    #     auto_wrap_policy=None,
    #     auto_wrapper_callable=auto_wrapper_callable,
    #     reshard_after_forward=True,
    #     compute_dtype=torch.bfloat16,
    #     buffer_dtype=torch.bfloat16,
    #     pin_layout_in_collective_ops=True,
    # )
    
    ##################### NESTED FSDP #########################
    
    # wrap single LlamaDecoderLayers
    # model.model.layers = nn.ModuleList([
    #     fsdp_wrap(grad_ckpt_wrap(layer))
    #     for layer in model.model.layers
    # ])
    
    # wrap submodules
    # submodules_to_wrap = [
    #     "embed_tokens",
    #     "mm_projector",
    #     "norm",
    #     "lm_head"
    # ]
    
    # for name in submodules_to_wrap:
    #     submodule = getattr(model.model if hasattr(model.model, name) else model, name)
    #     if sum(p.numel() for p in submodule.parameters()) > 0:  # Skip if no params
    #         wrapped_module = fsdp_wrap(grad_ckpt_wrap(submodule))
    #         if hasattr(model.model, name):
    #             setattr(model.model, name, wrapped_module)
    #         else:
    #             setattr(model, name, wrapped_module)
    
    # # wrap the main language model container
    # model.model = fsdp_wrap(grad_ckpt_wrap(model.model))
    # model = fsdp_wrap(model)
    
    # -----------------------------------------------
    
    model = FSDP(model, reshard_after_forward=True, pin_layout_in_collective_ops=True, flatten_parameters=True, shard_param_on_dim_0=True)
    
    # -----------------------------------------------
    
        
    ####################### FSDP SETUP #######################
    
    # if xm.is_master_ordinal():
    #     print(model)
    
    model = model.to(torch.bfloat16)
    
    print("After model fsdp_wrap:", model.model.mm_projector.attn_pool.query.shape)
    
    model = model.to(device)
    model.eval()
    
    # Open and read the JSON file
    with open(args.input_data, 'r') as file:
        data_val = json.load(file)
        
    data_val = data_val[-32:]
        
    chunk_size = len(data_val) // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank < world_size - 1 else len(data_val)
    local_data = data_val[start_idx:end_idx]
    
    output_save = []
    for element in tqdm.tqdm(local_data, desc=f"Rank {rank}"):
        
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
        
        print(image_tensor.shape)
        
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
                    with torch.amp.autocast(device_type="xla", dtype=torch.bfloat16):
                        output_ids = model.generate(
                            input_ids.to(device),
                            images=image_tensor,
                            image_sizes=[image_size],
                            do_sample=True if args.temperature > 0 else False,
                            temperature=args.temperature,
                            max_new_tokens=args.max_new_tokens,
                            streamer=streamer,
                            use_cache=True
                        )
                        
                        print(output_ids)
                
                print(f"Rank {rank} generated output for image {image_file}: {output_ids}")
                outputs = tokenizer.decode(output_ids[0]).strip()
                conv.messages[-1][-1] = outputs

                conversations_save.append({"question": inp, "answer":outputs})

            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        output_save.append({"image": image_file, "conversations_out": conversations_save})
    
    output_path = f"{args.output_file}_rank{rank}.json"
    with open(output_path, "w") as json_file:
        json.dump(output_save, json_file, indent=4)

    xm.rendezvous("save_complete")
    
def _mp_fn(index, args):
    dist.init_process_group('xla', init_method='xla://')
    world_size = xr.world_size()
    validation(index, world_size, args)

if __name__ == "__main__":
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
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    
    # Additional path parameters
    parser.add_argument("--output_file", type=str, 
                        default="/home/raspuntinov/gcs/ct_rate/outputs/output_validation_CTRATE_ChestReport_CT_CLIPllama_3_8b.json",
                        help="Output file path")
    parser.add_argument("--input_data", type=str, 
                        default="/home/raspuntinov/gcs/ct_rate/dataset/vqa/valid_vqa.json",
                        help="Input JSON file")
    parser.add_argument("--embeddings_path", type=str, default="/home/raspuntinov/gcs/ct_rate/dataset/embeddings")
    
    args = parser.parse_args()
    
    torch_xla.launch(_mp_fn, args=(args,), debug_single_process=False)