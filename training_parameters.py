# machine params
train_instance_count = 1
train_instance_type = 'ml.c4.4xlarge'
train_volume_size = 30
train_max_run = 360000
input_mode = 'File'

# hyperparams
mode = "supervised"
epochs = 10
min_count = 2
learning_rate = 0.05
vector_dim = 10
early_stopping = True
patience = 4
min_epochs = 5
word_ngrams = 2
