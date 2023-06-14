from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, stddev, expr
from scipy.stats import kendalltau

# Crea una sessione Spark
spark = SparkSession.builder.appName("Query1").getOrCreate()

# Carico il file "ratings.csv" nel dataframe
ratings = spark.read.format("csv").option("header", "true").load("ratings.csv")

# Raccolgo i rating per "movieId", poi aggiungo due colonne alla tabella.
# La prima è la colonna contenente la deviazione standard per ogni film e
# viene rinominata "stddev", mentre la seconda contiene il conto dei rating
# per "movieId" e viene rinominata "count"
movies_stats = ratings.groupBy("movieId").agg(stddev(col("Rating")).alias("stddev"),count(col("Rating")).alias("count")   
)

# Calcolo la correlazione di Pearson tra la deviazione standard e il conteggio delle valutazioni
corr_matrix = movies_stats.select(expr("corr(stddev, count)").alias("correlation")).head()

# Correlazione di Pearson
pcorr = float(corr_matrix.correlation)

# La funzione stddev genera errori per valori nulli, per cui li rimuovo
movies_stats = movies_stats.na.drop(subset=["stddev", "count"])

# Raccolgo i dati dal dataframe e li inserisco in una lista di tuple
data = movies_stats.select("stddev", "count").rdd.map(lambda row: (float(row["stddev"]), int(row["count"]))).collect()

# Calcolo la correlazione di Kendall
kcorr, _ = kendalltau([x[0] for x in data], [x[1] for x in data])

print("Correlazione di Pearson:", pcorr)
print("Correlazione di Kendall:", kcorr)

# Chiudo la sessione
spark.stop()


