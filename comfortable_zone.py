from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, expr
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql.types import DoubleType

def main():
    spark = SparkSession.builder \
        .appName("NBA Comfortable Zone Analysis") \
        .getOrCreate()

    # Load the data
    df = spark.read.option("header", True).option("inferSchema", True).csv("shot_logs.csv")

    # Drop rows with nulls in required feature columns and cast to DoubleType
    feature_cols = ['SHOT_DIST', 'CLOSE_DEF_DIST', 'SHOT_CLOCK']
    for col_name in feature_cols:
        df = df.withColumn(col_name, col(col_name).cast(DoubleType()))

    # Drop rows with nulls in any of the feature columns
    df_clean = df.dropna(subset=feature_cols)

    print("Total rows after dropping nulls: {}".format(df_clean.count()))

    # Assemble features using the clean data
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_features = assembler.transform(df_clean).select('PLAYER_NAME', 'SHOT_RESULT', 'features')

    # Apply KMeans clustering
    kmeans = KMeans(k=4, seed=1, featuresCol='features', predictionCol='zone')
    model = kmeans.fit(df_features)
    df_zones = model.transform(df_features)
    df_zones = df_zones.withColumn("zone", col("zone") + expr("1"))

    # Add binary column for made shots
    df_results = df_zones.withColumn("hit", when(col("SHOT_RESULT") == "made", 1).otherwise(0))

    # Compute zone hit rates per player
    zone_stats = df_results.groupBy("PLAYER_NAME", "zone") \
        .agg(
            count("*").alias("total_shots"),
            expr("sum(hit)").alias("made_shots")
        ) \
        .withColumn("hit_rate", col("made_shots") / col("total_shots"))

    # Focus on selected players
    target_players = ["james harden", "chris paul", "stephen curry", "lebron james"]
    best_zones = zone_stats.filter(col("PLAYER_NAME").isin(target_players)) \
        .orderBy("PLAYER_NAME", col("hit_rate").desc())

    # Show best zones
    best_zones.show(truncate=False)

    # Print centroids for interpretation
    print("\nZone Centroids:")
    centroids = model.clusterCenters()
    for i, c in enumerate(centroids):
        print("Zone {}: SHOT_DIST={:.2f}, CLOSE_DEF_DIST={:.2f}, SHOT_CLOCK={:.2f}".format(i + 1, c[0], c[1], c[2]))

    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number

    # Define window partitioned by player and ordered by hit_rate descending
    window_spec = Window.partitionBy("PLAYER_NAME").orderBy(col("hit_rate").desc())

    # Add row number within each player's partition
    ranked = best_zones.withColumn("rank", row_number().over(window_spec))

    # Filter for the top-ranked (highest hit_rate) zone for each player
    top_zones = ranked.filter(col("rank") == 1).drop("rank")

    # Rename 'zone' to 'best_zone'
    top_zones = top_zones.withColumnRenamed("zone", "best_zone")

    # Show the best zone for each player
    top_zones.show(truncate=False)


    spark.stop()

if __name__ == "__main__":
    main()

