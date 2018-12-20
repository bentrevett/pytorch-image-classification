import os
import shutil
import random

SEED = 1234

random.seed(SEED)

TRAIN_DIR = 'data/dogs-vs-cats/train/'
VALID_DIR = 'data/dogs-vs-cats/valid/'
TEST_DIR = 'data/dogs-vs-cats/test/'

assert os.path.exists(TRAIN_DIR)
assert os.path.exists(VALID_DIR)
assert os.path.exists(TEST_DIR)
assert os.path.exists(os.path.join(TRAIN_DIR, 'cat')), 'Run download-dogs-vs-cats.sh first!'
assert os.path.exists(os.path.join(TRAIN_DIR, 'dog')), 'Run download-dogs-vs-cats.sh first!'
assert os.path.exists(os.path.join(VALID_DIR, 'cat')), 'Run download-dogs-vs-cats.sh first!'
assert os.path.exists(os.path.join(VALID_DIR, 'dog')), 'Run download-dogs-vs-cats.sh first!'
assert os.path.exists(os.path.join(TEST_DIR, 'cat')), 'Run download-dogs-vs-cats.sh first!'
assert os.path.exists(os.path.join(TEST_DIR, 'dog')), 'Run download-dogs-vs-cats.sh first!'

all_images = os.listdir(TRAIN_DIR)

cats = [t for t in all_images if 'cat' in t and t.endswith('.jpg')]
dogs = [t for t in all_images if 'dog' in t and t.endswith('.jpg')]

random.shuffle(cats)
random.shuffle(dogs)

n_train_examples = int(len(all_images) * 0.8) // 2
n_valid_examples = int(len(all_images) * 0.1) // 2

train_cats = cats[:n_train_examples]
valid_cats = cats[n_train_examples:n_train_examples+n_valid_examples]
test_cats = cats[n_train_examples+n_valid_examples:]

train_dogs = dogs[:n_train_examples]
valid_dogs = dogs[n_train_examples:n_train_examples+n_valid_examples]
test_dogs = dogs[n_train_examples+n_valid_examples:]

for cat in train_cats:
    shutil.move(os.path.join(TRAIN_DIR, cat), os.path.join(TRAIN_DIR, 'cat', cat))

for cat in valid_cats:
    shutil.move(os.path.join(TRAIN_DIR, cat), os.path.join(VALID_DIR, 'cat', cat))
    
for cat in test_cats:
    shutil.move(os.path.join(TRAIN_DIR, cat), os.path.join(TEST_DIR, 'cat', cat))
    
for dog in train_dogs:
    shutil.move(os.path.join(TRAIN_DIR, dog), os.path.join(TRAIN_DIR, 'dog', dog))

for dog in valid_dogs:
    shutil.move(os.path.join(TRAIN_DIR, dog), os.path.join(VALID_DIR, 'dog', dog))
    
for dog in test_dogs:
    shutil.move(os.path.join(TRAIN_DIR, dog), os.path.join(TEST_DIR, 'dog', dog))