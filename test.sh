#!/bin/bash
source ../../../env.sh
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./comfortable_zone.py shot_logs.csv
