mkdir data
mkdir data/dogs-vs-cats
kaggle competitions download -c dogs-vs-cats
rm sampleSubmission.csv 
rm test1.zip
unzip train.zip
mv train data/dogs-vs-cats
rm train.zip
mkdir data/dogs-vs-cats/train/dog
mkdir data/dogs-vs-cats/train/cat 
mkdir data/dogs-vs-cats/valid
mkdir data/dogs-vs-cats/valid/dog
mkdir data/dogs-vs-cats/valid/cat
mkdir data/dogs-vs-cats/test
mkdir data/dogs-vs-cats/test/dog
mkdir data/dogs-vs-cats/test/cat
python process_dogs-vs-cats.py
