import json
import re
import random

# Only take rather short prompts
# good_prompts = [0, 1, 6, 7]
good_prompts = [6, 7]
training_set_path = 'asap dataset/training_set_rel3.tsv'
acceptable_non_ascii_chars = {'°'}

def replace_anonymization(text):
    text = re.sub(r'@PERSON(\d+)', lambda m: f'<person {m.group(1)}>', text)
    text = re.sub(r'@ORGANIZATION(\d+)', lambda m: f'<organization {m.group(1)}>', text)
    text = re.sub(r'@LOCATION(\d+)', lambda m: f'<location {m.group(1)}>', text)
    text = re.sub(r'@DATE(\d+)', lambda m: f'<date {m.group(1)}>', text)
    text = re.sub(r'@TIME(\d+)', lambda m: f'<time {m.group(1)}>', text)
    text = re.sub(r'@PERSON(\d+)', lambda m: f'<person {m.group(1)}>', text)
    text = re.sub(r'@MONEY(\d+)', lambda m: f'<money {m.group(1)}>', text)
    text = re.sub(r'@EMAIL(\d+)', lambda m: f'"email<address {m.group(1)}>', text)
    text = re.sub(r'@NUM(\d+)', lambda m: f'<number {m.group(1)}>', text)
    text = re.sub(r'@CITY(\d+)', lambda m: f'<city {m.group(1)}>', text)
    text = re.sub(r'@STATE(\d+)', lambda m: f'<state {m.group(1)}>', text)
    text = re.sub(r'@PERCENT(\d+)', lambda m: f'<percentage {m.group(1)}>', text)
    text = re.sub(r'Dr. @DR(\d+)', lambda m: f'<doctor {m.group(1)}>', text)
    text = re.sub(r'Dr @DR(\d+)', lambda m: f'<doctor {m.group(1)}>', text)
    text = re.sub(r'@DR(\d+)', lambda m: f'<doctor {m.group(1)}>', text)
    text = re.sub(r'@CAPS(\d+)', lambda m: f'<placeholder {m.group(1)}>', text)
    # Also "may" is replaced unfortunately
    # text = re.sub(r'@MONTH(\d+)', '"month \1"', text)
    # When there are 2 or 3 or 4 etc. spaces after each other, replace them with only 1
    text = re.sub(r'\s{2,}', ' ', text)

    return text

def non_ascii_chars(s):
    return {char for char in s if ord(char) > 127}

def replace_weird_chars(text):
    text = text.replace('\x93', '"')
    text = text.replace('\x94', '"')
    text = text.replace('\x96', '-')
    text = text.replace('\x91', '\'')
    text = text.replace('\x92', '\'')
    text = text.replace('\x85', '...')
    text = text.replace('¼', '1/4')
    text = text.replace('½', '1/2')
    text = text.replace('""', '"')
    text = text.replace('®', '')
    text = text.replace('\x98', '')
    text = text.replace('\x99', '')
    # assert text[0] == '"', text
    # assert text[-1] == '"', text
    if text[0] == '"':
        text = text[1:]
    if text[-1] == '"':
        text = text[:-1]
    return text

with open(training_set_path, encoding='iso-8859-1') as f:
    all_lines = f.read()

all_lines = all_lines.split('\n')
all_lines = [item.strip() for item in all_lines if len(item.strip()) > 0]
all_lines = [line.split('\t') for line in all_lines]
mapping = dict([(field, i) for i, field in enumerate(all_lines[0])])
print('mapping', mapping)
all_lines = all_lines[1:]
random.seed(0)
random.shuffle(all_lines)
#  {'essay_id': 0, 'essay_set': 1, 'essay': 2, 'rater1_domain1': 3, 'rater2_domain1': 4, 'rater3_domain1': 5, 'domain1_score': 6, 'rater1_domain2': 7, 'rater2_domain2': 8, 'domain2_score': 9, 'rater1_trait1': 10, 'rater1_trait2': 11, 'rater1_trait3': 12, 'rater1_trait4': 13, 'rater1_trait5': 14, 'rater1_trait6': 15, 'rater2_trait1': 16, 'rater2_trait2': 17, 'rater2_trait3': 18, 'rater2_trait4': 19, 'rater2_trait5': 20, 'rater2_trait6': 21, 'rater3_trait1': 22, 'rater3_trait2': 23, 'rater3_trait3': 24, 'rater3_trait4': 25, 'rater3_trait5': 26, 'rater3_trait6': 27}
print(f'Got {len(all_lines)} essays')
all_lines = [item for item in all_lines if int(item[mapping['essay_set']])-1 in good_prompts]
print(f'After filtering only the essays with short prompts, there are {len(all_lines)}')
new_all_lines = []
for line in all_lines:
    line[mapping['essay']] = replace_weird_chars(line[mapping['essay']])
    remaining_weird_chars = non_ascii_chars(line[mapping['essay']])
    if len(remaining_weird_chars - acceptable_non_ascii_chars) == 0:
        new_all_lines.append(line)
all_lines = new_all_lines
print(f'After filtering only the essays with "acceptable" characters, there are {len(all_lines)}')
# contains_at_char = [item for item in all_lines if '@' in item[mapping['essay']]]
# print(f'{len(contains_at_char)} contain an @ char')
# all_non_ascii_chars = set()
# for line in all_lines:
#     all_non_ascii_chars = all_non_ascii_chars.union(non_ascii_chars(line[mapping['essay']]))
# print('These are all non-ascii chars:', all_non_ascii_chars)

for line in all_lines:
    line[mapping['essay']] = replace_anonymization(line[mapping['essay']])

essays_with_at_character = [item[mapping['essay']] for item in all_lines if '@' in item[mapping['essay']]]
pattern_no_numbers = r"@[A-Z]+[0-9]+"
all_terms = []
for essay in essays_with_at_character:
    # Finding all matches without the numbers
    matches_no_numbers = re.findall(pattern_no_numbers, essay)
    all_terms += matches_no_numbers
terms = set(all_terms)

print(f'Found this many essays with an "@" character: {len(essays_with_at_character)}, with these unique terms: {terms}')

new_all_lines = []
for line in all_lines:
    if '@MONTH' in line[mapping['essay']]:
        continue
    new_all_lines.append(line)
all_lines = new_all_lines

print(f'Final lines remaining: {len(all_lines)}')
with open('final_asap.json', 'w') as f:
    json.dump(all_lines, f)

# with open('final_asap_text_only.json', 'w') as f:
#     json.dump([item[mapping['essay']] for item in all_lines], f)

