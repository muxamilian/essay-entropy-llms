import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import os

max_len = 500
nn = 'mistralai/Mistral-7B-v0.1'
nn_dtype = torch.bfloat16

# nn = 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'
# nn_dtype = torch.float32

# nn = 'openai-community/gpt2'
# nn_dtype = torch.float32

# nn = 'openai-community/gpt2-large'
# nn_dtype = torch.float32

# nn = 'openai-community/gpt2-xl'
# nn_dtype = torch.float32

# nn = 'facebook/opt-125m'
# nn_dtype = torch.float16

# nn = 'facebook/opt-2.7b'
# nn_dtype = torch.float16 

# nn = 'google/gemma-2b'
# nn_dtype = torch.bfloat16

nn = 'google/gemma-7b'
nn_dtype = torch.bfloat16

output_path = 'final_output_bawe_gemma-7B.jsonl'
input_path = 'final_bawe.json'

tokenizer = AutoTokenizer.from_pretrained(nn, device_map='cpu')
model = AutoModelForCausalLM.from_pretrained(nn, cache_dir='/data/dlvp/max/cache/huggingface/hub', torch_dtype=nn_dtype, device_map='cuda')

# text = "Once upon a time there was a cow called Eva. She was undoubtedly a very kind cow and she produced more milk than most other cows at the farm."
with open(input_path) as f:
    all_texts = json.load(f)

with open('asap_descriptions.json') as f:
    essay_prompts = json.load(f)
    # print('essay_prompts', essay_prompts)

num_lines = 0
if os.path.isfile(output_path):
    with open(output_path) as f:
        num_lines = len([item.strip() for item in f.read().split('\n') if len(item.strip()) > 0])

print(f'{num_lines=}')

f_out = open(output_path, 'a')

for line_index, line in tqdm(list(enumerate(all_texts))):
    if line_index < num_lines:
        continue
    if "asap" in input_path:
        essay_set = line[1]
        written_text = line[2]
        essay = essay_prompts[int(essay_set)-1]
        written_text_tokens = tokenizer(written_text, return_tensors="pt", add_special_tokens=False)
        written_text_len = int(written_text_tokens['input_ids'].shape[-1])
        # print(f'{written_text_len=}')
        constant_part = f'{essay}\r\nAnswer:\r\n'
        constant_part_tokens = tokenizer(constant_part, return_tensors="pt")
        constant_part_len = int(constant_part_tokens['input_ids'].shape[-1])
        # print(f'{constant_part_len=}')
        text = constant_part + written_text
        print(f'{text=}')
        # print(f'{written_text[:max_len]}')
        inputs = copy.deepcopy(constant_part_tokens)
        inputs['input_ids'] = torch.cat([inputs['input_ids'], written_text_tokens['input_ids']], dim=1)
        inputs['attention_mask'] = torch.cat([inputs['attention_mask'], written_text_tokens['attention_mask']], dim=1)
        assert inputs.input_ids.shape[1] == inputs.attention_mask.shape[1]
    else:
        text = line['content']
        print(f'{text[:max_len]=}')
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        written_text_len = int(inputs['input_ids'].shape[-1])
        constant_part_len = 0

    sequence = []
    all_outputs = []
    for i in tqdm(range(min(written_text_len-1, max_len))):
        modified_inputs = copy.deepcopy(inputs)
        modified_inputs.data['input_ids'] = modified_inputs.data['input_ids'][:, :constant_part_len+i+1]
        modified_inputs.data['attention_mask'] = modified_inputs.data['attention_mask'][:, :constant_part_len+i+1]
        modified_inputs = modified_inputs.to('cuda')
        outputs = model.generate(**modified_inputs, min_new_tokens=1, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, pad_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True))
        all_outputs.append(outputs['scores'][0].to('cpu').squeeze())

    probabilities = [torch.nn.functional.softmax(item.float()) for item in all_outputs]
    entropies = [(torch.sum(torch.special.entr(item))).item() for item in probabilities]

    out_json = json.dumps(entropies)
    f_out.write(out_json+'\n')
    f_out.flush()
