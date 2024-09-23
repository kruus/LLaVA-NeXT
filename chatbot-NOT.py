from threading import Thread
from transformers import TextIteratorStreamer
import json

url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "qwen_1_5"
question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]

max_context_length = getattr(model.config, "max_position_embeddings", 2048)
num_image_tokens = question.count(DEFAULT_IMAGE_TOKEN) * model.get_vision_tower().num_patches

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

max_new_tokens = min(4096, max_context_length - input_ids.shape[-1] - num_image_tokens)

if max_new_tokens < 1:
    print(
        json.dumps(
            {
                "text": question + "Exceeds max token length. Please start a new conversation, thanks.",
                "error_code": 0,
            }
        )
    )
else:
    gen_kwargs = {
        "do_sample": False,
        "temperature": 0,
        "max_new_tokens": max_new_tokens,
        "images": image_tensor,
        "image_sizes": image_sizes,
    }

    thread = Thread(
        target=model.generate,
        kwargs=dict(
            inputs=input_ids,
            streamer=streamer,
            **gen_kwargs,
        ),
    )
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        print(generated_text, flush=True)
        # print(json.dumps({"text": generated_text, "error_code": 0}), flush=True)

    print("Final output:", generated_text)
