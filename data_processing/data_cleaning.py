#!/usr/bin/env python
import numpy as np
import csv
import re
import gensim
from sklearn.metrics.pairwise import cosine_similarity


def string_cleaning(input_string):
    string_split = input_string.split(' ')
    result_array = []
    for sub_str in string_split:
        sub_str = sub_str.strip('/')
        sub_str = sub_str.strip('\'')
        sub_str = sub_str.strip('(')
        sub_str = sub_str.strip(')')
        sub_str = sub_str.strip('?')
        result_array.append(sub_str)
    temp_result = " ".join(result_array)
    return re.sub(' +', ' ', " ".join(temp_result.split('-')))


model = gensim.models.KeyedVectors.load_word2vec_format("glove_model.txt", binary=False)

csv_file = open("Abt.csv", "r", encoding='windows-1252')
reader = csv.reader(csv_file)
abt_result = {}
for item in reader:
    if reader.line_num == 1:
        continue
    abt_result[item[0]] = [string_cleaning(item[1]), string_cleaning(item[2])]
csv_file.close()

csv_file = open("Buy.csv", "r", encoding='windows-1252')
reader = csv.reader(csv_file)
buy_result = {}
for item in reader:
    if reader.line_num == 1:
        continue
    buy_result[item[0]] = [string_cleaning(item[1]), string_cleaning(item[2])]
csv_file.close()

abt_result_vec = {}
for key in abt_result:
    temp_name = abt_result[key][0]
    temp_name_vector = np.zeros(300)
    temp_description = abt_result[key][1]
    temp_description_vector = np.zeros(300)
    if len(temp_name) != 0:
        name_split = temp_name.split(' ')
        count = 0
        for word in name_split:
            if word in model:
                count += 1
                temp_name_vector = temp_name_vector + model[word]
        temp_name_vector = temp_name_vector / count
    if len(temp_description) != 0:
        des_split = temp_description.split(' ')
        for word in des_split:
            if word in model:
                temp_description_vector = temp_description_vector + model[word]
        temp_description_vector = temp_description_vector / len(des_split)
    result_vector = np.concatenate((temp_name_vector, temp_description_vector))
    abt_result_vec[key] = result_vector

buy_result_vec = {}
for key in buy_result:
    temp_name = buy_result[key][0]
    temp_name_vector = np.zeros(300)
    temp_description = buy_result[key][1]
    temp_description_vector = np.zeros(300)
    if len(temp_name) != 0:
        name_split = temp_name.split(' ')
        count = 0
        for word in name_split:
            if word in model:
                count += 1
                temp_name_vector = temp_name_vector + model[word]
        temp_name_vector = temp_name_vector / count
    if len(temp_description) != 0:
        des_split = temp_description.split(' ')
        for word in des_split:
            if word in model:
                temp_description_vector = temp_description_vector + model[word]
        temp_description_vector = temp_description_vector / len(des_split)
    result_vector = np.concatenate((temp_name_vector, temp_description_vector))
    buy_result_vec[key] = result_vector

abt_buy_map = {}
csv_file = open("abt_buy_perfectMapping.csv", "r", encoding='windows-1252')
reader = csv.reader(csv_file)
for item in reader:
    if reader.line_num == 1:
        continue
    abt_buy_map[item[0]] = item[1]
csv_file.close()

# L2 norm distance based matching
num_of_correct = 0
for abt in abt_buy_map:
    distance_list = []
    for key in buy_result_vec:
        temp_distance = np.linalg.norm(abt_result_vec[abt] - buy_result_vec[key])
        distance_list.append((temp_distance, key))
    distance_list.sort()
    if distance_list[0][1] == abt_buy_map[abt]:
        num_of_correct += 1
print(num_of_correct / len(abt_buy_map))

# Cosine distance based matching
num_of_correct = 0
for abt in abt_buy_map:
    distance_list = []
    for key in buy_result_vec:
        temp_vector = [abt_result_vec[abt], buy_result_vec[key]]
        cosine_distance = cosine_similarity(temp_vector)[0, 1]
        distance_list.append((cosine_distance, key))
    distance_list.sort(reverse=True)
    if distance_list[0][1] == abt_buy_map[abt]:
        num_of_correct += 1
print(num_of_correct / len(abt_buy_map))



