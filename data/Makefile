all: extract_data extract_features

extract_data:
	cat data_part_* > data.tar.gz
	tar -xzf data.tar.gz 
	rm data.tar.gz

extract_features:
	unzip features_ring.zip
