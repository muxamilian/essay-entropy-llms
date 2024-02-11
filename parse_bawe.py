import os
import xml.etree.ElementTree as ET
import json
import re
import random

txt_path = 'bawe corpus/CORPUS_TXT'
xml_path = 'bawe corpus/CORPUS_UTF-8'
all_file_names = [item[:-4] for item in os.listdir(txt_path)]

unique_grades = {}
unique_dg = {}

all_essays = []
for name in all_file_names:
    new_item = {}
    with open(txt_path+'/'+name+'.txt') as f:
        content = f.read()
        # This filters out all texts containing tags
    # if re.search(r'<[^>]+>', content) is not None:
    #     continue
    # content = re.sub(r'<[^>]+>', ' ', content)
    # content = re.sub(r'\s{2,}', ' ', content)
    new_item['content'] = content

    # Load and parse the XML file
    tree = ET.parse(xml_path+'/'+name+'.xml')  # Replace with your actual file path
    root = tree.getroot()

    grade_element = root.find('.//p[@n="grade"]')
    assert grade_element is not None
    grade = grade_element.text
    if grade == 'unknown':
        continue
    new_item['grade'] = grade

    if grade not in unique_grades:
        unique_grades[grade] = 1
    else:
        unique_grades[grade] += 1

    disciplinary_group_element = root.find('.//p[@n="disciplinary group"]')
    assert disciplinary_group_element is not None
    disciplinary_group = disciplinary_group_element.text
    new_item['disciplinary_group'] = disciplinary_group

    if disciplinary_group not in unique_dg:
        unique_dg[disciplinary_group] = 1
    else:
        unique_dg[disciplinary_group] += 1

    all_essays.append(new_item)

print(f'Unique grades: {unique_grades}')
print(f'Unique disciplinaries: {unique_dg}')
print(f'Got {len(all_essays)} essays')
random.seed(0)
random.shuffle(all_essays)
with open('final_bawe.json', 'w') as f:
    json.dump(all_essays, f)
