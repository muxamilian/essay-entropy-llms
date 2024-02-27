from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# nn = 'mistralai/Mistral-7B-v0.1'
# nn_dtype = torch.bfloat16

# nn = 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'
# nn_dtype = torch.float32

# nn = 'openai-community/gpt2'
# nn_dtype = torch.float32

nn = 'google/gemma-2b'
nn_dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(nn, device_map='cuda')

model = AutoModelForCausalLM.from_pretrained(nn, cache_dir='/data/dlvp/max/cache/huggingface/hub', torch_dtype=nn_dtype, device_map='cuda')

text = "Hello my name is"
inputs = tokenizer(text, return_tensors="pt").to('cuda')

outputs = model.generate(**inputs, max_new_tokens=20, return_dict_in_generate=True, output_scores=True, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True))
