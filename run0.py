
print("\n\nFinished Qwen2 tests, Beginning llava tests\n\n")
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import os
import sys
import warnings

warnings.filterwarnings("ignore")
ov_models = [
"llava-hf/llava-onevision-qwen2-0.5b-ov-hf",    # image-text-to-text
"llava-hf/llava-onevision-qwen2-7b-si-hf",      # image-text-to-text
#"llava-hf/llava-onevision-qwen2-72b-si-hf",    # image-text-to-text
#"llava-hf/llava-onevision-qwen2-72b-ov-hf",    # image-text-to-text
"llava-hf/llava-onevision-qwen2-0.5b-si-hf",    # image-text-to-text
"llava-hf/llava-onevision-qwen2-7b-ov-hf",      # image-text-to-text
"lmms-lab/llava-onevision-qwen2-0.5b-ov",       # text generation
"lmms-lab/llava-onevision-qwen2-7b-ov",         # text generation
#"lmms-lab/llava-onevision-qwen2-72b-si",      # text generation
"lmms-lab/llava-onevision-qwen2-0.5b-si",       # text generation
"lmms-lab/llava-onevision-qwen2-7b-si",         # text generation
#"lmms-lab/llava-onevision-qwen2-72b-ov-sft",   # text generation
#"lmms-lab/llava-onevision-qwen2-72b-ov-chat",  # text generation (Sept. 12 2024)
"lmms-lab/llava-onvevision-7b-ov-chat"  # llava-oncevision-7b-ov + DPO + human preference
]

load_args = {}
model_args = os.getenv("MODEL_ARGS", 'None')
if 'attn_implementation=None' in model_args.split(','):
    print("MODEL_ARGS --> turning OFF attn_implementation")
    load_args['attn_implementation'] = None

print(f" load_pretrained_model will get additional args\n {load_args=}\n", flush=True)

from transformers import AutoTokenizer, AutoProcessor
# Huh? from qwen_vl_utils import process_vision_info

def test_llava(pretrained: str):
    print(f'\n\nTrying to evaluate {pretrained = }" ...', flush=True)
    pretrained = ov_model
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, device_map=device_map,
        #attn_implementation=None,
        **load_args,
    )  # Add any other thing you want to pass in llava_model_args

    model.eval()

    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]


    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    print(text_outputs)

print(f"{ov_models = }", flush=True)
for ov_model in ov_models:
    try:
        test_llava(pretrained = ov_model)
    except Exception as e:
        print(f"\nERROR:\n{e}")

# ----------
print("\nFinished trying out llava-onevision models")
