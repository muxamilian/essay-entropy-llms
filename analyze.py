import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from collections import defaultdict
import statsmodels.api as sm
import argparse
import re
import copy

max_len = 500


parser = argparse.ArgumentParser()
# parser.add_argument('--raw_data_path', default="final_output_bawe_mistral.jsonl")
# parser.add_argument('--raw_input_path', default="final_bawe.json")
parser.add_argument('--raw_data_path', default="final_output_asap_mistral.jsonl")
parser.add_argument('--raw_input_path', default="final_asap_smaller.json")
args = parser.parse_args()

# raw_data_path = 'final_output_asap_mistral.jsonl'
# raw_input_path = 'final_asap_mistral.json'

# raw_data_path = 'output_asap_smaller.jsonl'
# raw_input_path = 'final_asap_smaller.json'

raw_data = open(args.raw_data_path).read()
lines = [item.strip() for item in raw_data.split('\n') if len(item.strip()) > 0]
jsons = [json.loads(item) for item in lines]
nps = [np.array(item, dtype=np.float64) for item in jsons]

raw_input = open(args.raw_input_path).read()
input_lines = json.loads(raw_input)
if 'asap' in args.raw_input_path:
    def compute_score(item):
        item[1] = int(item[1])
        item[6] = int(item[6])
        if item[1] == 2:
            item[9] = int(item[9])
        if item[1] == 1:
            return (item[6] - 2)/10
        elif item[1] == 2:
            return (item[6] + item[9] - 2)/8
        elif item[1] == 7:
            return item[6]/30
        elif item[1] == 8:
            return int(item[6]/60*30)/30
        else:
            raise ValueError(f'Got {item} with wrong essay id')
elif 'bawe':
    def compute_score(item):
        return 0. if item['grade'] == 'M' else 1.

scores = np.array([compute_score(item) for item in input_lines])
for score in scores:
    assert score >= 0
    assert score <= 1
scores = scores[:len(jsons)]

max_len = max([len(item) for item in jsons])
acc = np.zeros([max_len], dtype=np.float64)
nums = np.zeros([max_len], dtype=np.float64)

for array in nps:
    acc[:len(array)] += array
    nums[:len(array)] += 1

normalized = acc/nums

# plt.plot(range(len(normalized)), normalized)
# plt.show()

# normalized_nps = [item[:min(500, len(item))]/normalized[:min(500, len(item))] for item in nps]
normalized_nps = [item[:min(500, len(item))] for item in nps]
means = np.array([np.mean(item) for item in normalized_nps])

# inliers = (scores != 0) & (scores !=1)
# scores = scores[inliers]
# means = means[inliers]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
pattern = r"<\w+\s\d+>"
nn = 'mistralai/Mistral-7B-v0.1'
nn_dtype = torch.bfloat16
tokenizer = AutoTokenizer.from_pretrained(nn, device_map='cpu')
actual_texts = []
token_lens = []
for line in input_lines:
    if 'asap' in args.raw_input_path:
        written_text = line[2]
    else:
        written_text = line['content']
    written_text_tokens = tokenizer(written_text, return_tensors="pt", add_special_tokens=False)
    written_text_len = int(written_text_tokens['input_ids'].shape[-1])
    inputs = copy.deepcopy(written_text_tokens)
    # import pdb; pdb.set_trace()
    inputs['input_ids'] = written_text_tokens['input_ids'][:,:501]
    inputs['attention_mask'] = written_text_tokens['attention_mask'][:,:501]
    token_len = inputs['input_ids'].shape[-1]
    token_lens.append(token_len)
    actual_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    # print(actual_text+'\n')
    actual_texts.append(actual_text)
    assert inputs.input_ids.shape[1] == inputs.attention_mask.shape[1]
tokens_per_text = np.array([len(re.findall(pattern, item))/len(item) for item in actual_texts])

print("tokens and means")
pearsonr = scipy.stats.pearsonr(tokens_per_text, means)
print(f'{pearsonr=}')
spearmanr = scipy.stats.spearmanr(tokens_per_text, means)
print(f'{spearmanr=}')

# lens = np.array([len(item) for item in actual_texts])
print("mean lengths and means")
pearsonr = scipy.stats.pearsonr(token_lens, means)
print(f'{pearsonr=}')
spearmanr = scipy.stats.spearmanr(token_lens, means)
print(f'{spearmanr=}')

print("scores and means")
pearsonr = scipy.stats.pearsonr(scores, means)
print(f'{pearsonr=}')
spearmanr = scipy.stats.spearmanr(scores, means)
print(f'{spearmanr=}')

# # Aggregating y-values for each unique x-value
# xy_dict = defaultdict(list)
# for xi, yi in zip(scores, means):
#     xy_dict[xi].append(yi)

# # Averaging y-values
# x_unique = np.array(list(xy_dict.keys()))
# y_avg = np.array([np.mean(xy_dict[xi]) for xi in x_unique])

results = []
for i in [1]:
    coeffs = np.polyfit(scores, means, i)
    x_fit = np.linspace(min(scores), max(scores), 500)
    y_fit = np.polyval(coeffs, x_fit)
    min_x = None
    min_y = float('inf')
    for x_i, y_i in zip(x_fit, y_fit):
        if y_i < min_y:
            min_y = y_i
            min_x = x_i
    print(f'Min x: {min_x}, min y: {min_y}')
    results.append((x_fit, y_fit))
    # Calculate R-squared
    y_pred = np.polyval(coeffs, scores)  # Predicted y values for unique x
    ssr = np.sum((means - y_pred) ** 2)    # Sum of squares of the residuals
    sst = np.sum((means - np.mean(means)) ** 2)  # Total sum of squares
    r_squared = 1 - (ssr / sst)
    print(f'{i=}, {r_squared=}')

    x_poly = np.column_stack([scores**j for j in range(1, i + 1)])
    x_poly = sm.add_constant(x_poly)
    model = sm.OLS(means, x_poly).fit()
    print(model.summary())

    x_poly = np.column_stack([tokens_per_text] + [token_lens])
    x_poly = sm.add_constant(x_poly)
    model = sm.OLS(means, x_poly).fit()
    print(model.summary())

    x_poly = np.column_stack([scores**j for j in range(1, i + 1)] + [tokens_per_text] + [token_lens])
    x_poly = sm.add_constant(x_poly)
    model = sm.OLS(means, x_poly).fit()
    print(model.summary())

# plt.plot(scores, means, linestyle='None', marker='.', alpha=0.33, markeredgecolor='none')
# plt.plot(*(results[0]))
# # plt.plot(*(results[1]))
# plt.show()
    


# Define the number of bins

num_bins = min(len(np.unique(scores)), 5)
delta = 1 if num_bins == 2 else 0

x_values = scores
y_values = means
# Bin the x values
bins = np.linspace(0., 1., num_bins + (1 - delta))
indices = np.digitize(x_values, bins) - delta

# Calculate average y for each bin
bin_averages = [y_values[indices == i].mean() for i in range((1-delta), num_bins+(1-delta))]

# Plotting
plt.bar(range(num_bins), bin_averages, width=1, edgecolor="black", align='center')
if num_bins != 2:
    plt.xticks(range(num_bins), [f"{bins[i]:.1f} - {bins[i+1]:.1f}" for i in range(num_bins)], rotation=45)
else:
    plt.xticks(range(num_bins), [f"{i}" for i in range(num_bins)], rotation=45)
plt.xlabel('Bins')
plt.ylabel('Average Y')
plt.title('Average Y values in each X bin')
plt.tight_layout()
plt.show()
