from data_processing_utils import map_classes, transform_instance, \
    preprocess

# indexing the labels
index_to_label = map_classes(file_path='./data/classes.txt')

# preprocess the data
preprocess('./data/train.csv', './data/dbpedia.train', keep=.2)        
preprocess('./data/test.csv', './data/dbpedia.validation')