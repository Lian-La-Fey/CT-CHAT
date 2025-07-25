import os
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

from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP, checkpoint_module
from torch_xla.distributed.fsdp.wrap import (size_based_auto_wrap_policy,
                                             transformer_auto_wrap_policy)

import torch_xla.distributed.fsdp as fsdp
import torch_xla.distributed.fsdp as xla_fsdp

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from llava.model.multimodal_projector.coca_attentional_pooler import AttentionalPoolProjector


################ For TPU support ################


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def validation(rank, world_size, args):
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
    
    
    
    ####################### FSDP SETUP #######################
    
    model = model.to(torch.float32)
    
    # if hasattr(model, 'tie_weights'):
    #     model.tie_weights()  # Ensure weights are tied initially
    # model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())
    
    # for i in range(len(model.model.layers)):
    #     model.model.layers[i] = checkpoint_module(model.model.layers[i])
    # model.model.mm_projector = checkpoint_module(model.model.mm_projector)
    
    # auto_wrap_policy = partial(
    #     size_based_auto_wrap_policy,
    #     min_num_params=1e4  # Adjust based on layer sizes
    # )
    
    # model = fsdp.XlaFullyShardedDataParallel(
    #     model,
    #     auto_wrap_policy=auto_wrap_policy,
    #     reshard_after_forward=True,  # Enable ZeRO-3
    #     flatten_parameters=True,
    # )
        
    # auto_wrap_policy = partial(
    #     size_based_auto_wrap_policy,
    #     min_num_params=1e2,
    # )
    
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer
        },
    )
    
    fsdp_wrap = lambda m: FSDP(
        m,
        # compute_dtype=torch.float32,
        fp32_reduce_scatter=False,
        flatten_parameters=False,
        shard_param_on_dim_0=False,
        pin_layout_in_collective_ops=True,
        auto_wrap_policy=auto_wrap_policy,
        auto_wrapper_callable=None,
        reshard_after_forward=True
    )
    
    # grad_ckpt_wrap = checkpoint_module if args.use_gradient_checkpointing else (lambda x: x)
    
    # for name, sub in model.model.named_children():
    #     print(name, sub)
    #     if sum(p.numel() for p in sub.parameters()) == 0:
    #         print(name, sub)
    #         continue
        
    #     if name == "mm_projector":
    #         continue
    
    #     m_fsdp = fsdp_wrap(grad_ckpt_wrap(getattr(model.model, name)))
    #     setattr(model, name, m_fsdp)
    
    # for name, sub_module in model.model.named_children():
    #     if sum(p.numel() for p in sub_module.parameters()) == 0:
    #         print(f"Skip empty / helper modules: {name, sub_module}")
    #         continue

    #     
    #     if name == "mm_projector":
    #         continue

    
    #     if name == "layers":
    #         print("→ Wrapping each LlamaDecoderLayer in 'layers'")
    #         for idx, layer in enumerate(sub_module):
    #             wrapped = fsdp_wrap(grad_ckpt_wrap(layer))
    #             sub_module[idx] = wrapped

    
    #         print(f"→ Wrapping model.model.{name}")
    #         wrapped = fsdp_wrap(grad_ckpt_wrap(sub_module))
    #         setattr(model.model, name, wrapped)

    
    # model.lm_head = fsdp_wrap(grad_ckpt_wrap(model.lm_head))
    
    model = fsdp_wrap(model)
    
    print("After model fsdp_wrap:", model.model.mm_projector.attn_pool.query.shape)
    
    # -----------------------------------------------
    
    # model = FSDP(model, reshard_after_forward=True)
    
    # -----------------------------------------------
    
    
    # llama_fsdp_policy = partial(
    #     transformer_auto_wrap_policy,
    #     transformer_layer_cls={LlamaDecoderLayer}
    # )

    # # 2. Wrap your model with XlaFullyShardedDataParallel
    # # The outer wrapper handles any parameters not in a LlamaDecoderLayer
    # # (like embeddings and the final lm_head).
    # model = xla_fsdp.XlaFullyShardedDataParallel(
    #     model,
    #     auto_wrap_policy=llama_fsdp_policy,
    #     # reshard_after_forward=True enables full ZeRO-3 parameter sharding
    #     reshard_after_forward=True
    # )
    
    # ---------------------------------------------------------
    
    # def llama_fsdp_policy(module, recurse, unwrapped_params):
    #     from llava.model.multimodal_projector.coca_attentional_pooler import AttentionalPoolProjector
    #     if isinstance(module, AttentionalPoolProjector):
    #         return False
    #     return transformer_auto_wrap_policy(
    #         module,
    #         recurse=recurse,
    #         unwrapped_params=unwrapped_params,
    #         transformer_layer_cls={LlamaDecoderLayer}
    #     )

    # wrapped_layers = nn.ModuleList()
    # for layer in model.model.layers:
    #     wrapped_layer = xla_fsdp.XlaFullyShardedDataParallel(
    #         checkpoint_module(layer),
    #         auto_wrap_policy=llama_fsdp_policy,
    #         reshard_after_forward=True
    #     )
    #     wrapped_layers.append(wrapped_layer)
    # model.model.layers = wrapped_layers

    # model = xla_fsdp.XlaFullyShardedDataParallel(
    #     model,
    #     auto_wrap_policy=llama_fsdp_policy,
    #     reshard_after_forward=True
    # )
    
    # -----------------------------------------
    
    # def my_wrap_policy(module, recurse, unwrapped_params):
    #     # skip projector
    #     if isinstance(module, AttentionalPoolProjector):
    #         return False
    #     # wrap all other submodules
    #     return True
    
    # fsdp_wrap = lambda m: FSDP(
    #     m,
    #     auto_wrap_policy=my_wrap_policy,
    #     reshard_after_forward=True,
    # )
    
    # wrapped_layers = nn.ModuleList([
    #     fsdp_wrap(checkpoint_module(layer))
    #     for layer in model.model.layers
    # ])
    # model.model.layers = wrapped_layers
    
    # for name, sub in model.model.named_children():
    #     if name == "layers" or name == "mm_projector":
    #         continue
    #     setattr(model.model, name, fsdp_wrap(sub))
        
    
    # model.lm_head = fsdp_wrap(model.lm_head)
    
    # model = fsdp_wrap(model)
    
        
    ####################### FSDP SETUP #######################
    
    # print(model)
    
    # model = model.to(device)
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
    for element in tqdm.tqdm(data_val, desc=f"Rank {rank}"):
        
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
    world_size = xr.world_size()
    validation(index, world_size, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/raspuntinov/gcs/ct_rate/models/ct_chat/llava-lora-llama_3.1_8b")
    parser.add_argument("--model_base", type=str, default="meta-llama/Llama-3.1-8B")
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
                        default="./output/output_validation_CTRATE_ChestReport_CT_CLIPllama_3_8b.json",
                        help="Output file path")
    parser.add_argument("--input_data", type=str, 
                        default="/home/raspuntinov/gcs/ct_rate/dataset/vqa/valid_vqa.json",
                        help="Input JSON file")
    parser.add_argument("--embeddings_path", type=str, default="/home/raspuntinov/gcs/ct_rate/dataset/embeddings")
    
    args = parser.parse_args()
    
    torch_xla.launch(_mp_fn, args=(args,), debug_single_process=False)