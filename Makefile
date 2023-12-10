
download_data:
	if [ ! -d "./data/raw" ]; then mkdir -p ./data/raw; fi
	curl -o ./data/raw/creditcard.csv https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

build_tfrecord:
	python example_gen.py --src ./data/raw/ --dst ./data/example_gen/ --train_fraction 0.8