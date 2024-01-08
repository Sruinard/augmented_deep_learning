
download_data:
	if [ ! -d "./data/raw" ]; then mkdir -p ./data/raw; fi
	curl -o ./data/raw/creditcard.csv https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

build_tfrecord: download_data
	python example_gen.py --config=configs/default.py

run_training:
	python main.py --config=configs/default.py

run_serving:
	python serving.py --config=configs/default.py