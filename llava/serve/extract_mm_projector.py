import torch

from llava.model.multimodal_projector.builder import build_vision_projector
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

# /home/raspuntinov/gcs/ct_rate/models/ct_chat/llava-lora-llama_3.1_8b
# /home/raspuntinov/gcs/ct_rate/models/ct_chat/llama_3.1_8b_instrcut

# 1) Load your merged checkpoint directory
MODEL_DIR = "/home/raspuntinov/gcs/ct_rate/models/ct_chat/llava-lora-llama_3.1_8b"
BASE_DIR  = "/home/raspuntinov/gcs/ct_rate/models/ct_chat/llama_3.1_8b_instrcut"  # same if fully merged

disable_torch_init()
tokenizer, model, _, _ = load_pretrained_model(
    MODEL_DIR, BASE_DIR, model_name="llava-lora-llama_3.1_8b",
    load_8bit=False, load_4bit=False,
    device_map="cpu", device="cpu"
)

# 2) Grab the config from model.config
config = model.config

# 3) Build a fresh projector module
projector = build_vision_projector(config)

# 4) Copy in the state dict from the loaded model’s projector
#    (this assumes the underlying loaded model did load the projector weights
#     into model.get_model().mm_projector)
full_model = model.get_model() if hasattr(model, "get_model") else model
projector.load_state_dict(full_model.mm_projector.state_dict(), strict=True)

# 5) Save it out
torch.save(projector.state_dict(), f"{MODEL_DIR}/mm_projector.bin")
print("Saved mm_projector.bin with keys:", projector.state_dict().keys())