#!/usr/bin/sh
cat data_part_* > data.tar.gz
tar -xzf data.tar.gz 
rm data.tar.gz
unzip features_ring.zip
