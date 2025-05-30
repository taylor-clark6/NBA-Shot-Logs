# NBA Comfortable Zone Analysis (PySpark)

This project uses PySpark and KMeans clustering to analyze NBA players' shooting data and identify their most "comfortable zones" on the court—zones where their shot success rate is the highest.

## Dataset
The [dataset](https://www.kaggle.com/datasets/dansbecker/nba-shot-logs) is expected to be a CSV file named `shot_logs.csv`, containing fields including:
- `PLAYER_NAME`
- `SHOT_DIST`
- `CLOSE_DEF_DIST`
- `SHOT_CLOCK`
- `SHOT_RESULT`

Ensure this file is uploaded to HDFS:
- hdfs dfs -put shot_logs.csv /user/root/

## Approach
1. Preprocess the dataset by cleaning nulls and casting key columns to DoubleType.

2. Use VectorAssembler to combine selected features.

3. Apply KMeans clustering to segment shots into zones.

4. Calculate each player's shooting hit rate per zone.

5. Identify the best zone (with highest hit rate) for a set of target players.

## Target Players
- James Harden
- Chris Paul
- Stephen Curry
- Lebron James

## Running the Script
Run the script using: bash ./test.sh

## Output
- A table showing each player's shooting hit rate across zones

- The best-performing zone per player

- The centroid values of each zone, representing average shot characteristics

## Requirements
- PySPark
- HDFS
- Spark
