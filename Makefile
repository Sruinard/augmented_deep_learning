
download_data:
	if [ ! -d "./data" ]; then mkdir ./data; fi
	curl -o ./data/creditcard.csv https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv