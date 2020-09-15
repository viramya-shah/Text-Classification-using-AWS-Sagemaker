import pickle
from random import shuffle
import multiprocessing
from multiprocessing import Pool
import csv
import nltk

def map_classes(file_path):
    index_to_label = {}
    with open(file_path) as f:
        for i, label in enumerate(f.readlines()):
            index_to_label[str(i+1)] = label.strip()

    pickle.dump(index_to_label, open("./dumps/index_to_label.pkl", 'wb'))
    return index_to_label

def transform_instance(row, index_to_label):
    cur_row = []
    label = "__label__" + index_to_label[row[0]]  
    cur_row.append(label)
    cur_row.extend(nltk.word_tokenize(row[1].lower()))
    cur_row.extend(nltk.word_tokenize(row[2].lower()))
    return cur_row

def preprocess(input_file, output_file, keep=1, index_to_label = None):
    index_to_label = pickle.load(open("./dumps/index_to_label.pkl", 'rb'))
    all_rows = []
    with open(input_file, 'r', encoding="utf-8") as csvinfile:
        csv_reader = csv.reader(csvinfile, delimiter=',')
        for row in csv_reader:
            all_rows.append(row)
    shuffle(all_rows)
    all_rows = all_rows[:int(keep*len(all_rows))]
    transformed_rows = []
    for i in all_rows:
        transformed_rows.append(transform_instance(i, index_to_label))

    with open(output_file, 'w', encoding="utf-8") as csvoutfile:
        csv_writer = csv.writer(csvoutfile, delimiter=' ', lineterminator='\n')
        csv_writer.writerows(transformed_rows)

