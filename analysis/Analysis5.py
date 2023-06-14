from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Creo una sessione Spark
spark = SparkSession.builder.appName("Analysis5").getOrCreate()

# Carico il file "ratings.csv" nel dataframe
ratings = spark.read.format("csv").option("header", "true").load("ratings.csv")

# Converti la colonna "rating" in float
ratings = ratings.withColumn("rating", col("rating").cast("float"))

# Calcola lo score medio per ogni film
average_score_per_movie = ratings.groupBy("movieId").avg("rating").orderBy("movieId").select(col("movieId"), col("avg(rating)").alias("Average Score"))

# Mostra la tabella con il numero di valutazioni e lo score medio per ogni film, ordinata per score medio.
# Per farlo devo raggruppare i rating per "movieId" e ottenere due colonne ulteriori.
# Nella prima devo inserire il conteggio di rating per film sotto il nome di "Number of Ratings",
# mentre nella seconda inserisco il rating medio.
# Poi di tutti questi film filtro quelli che abbiano un numero di rating >=20, li metto in
# ordine decrescente e prendo i primi 10
ratings_with_average_score = ratings.groupBy("movieId").count().join(average_score_per_movie, "movieId").withColumnRenamed("count", "Number of Ratings").filter(col("Number of Ratings") >= 20).orderBy(col("Average Score").desc()).limit(10)

# Mostro la tabella
ratings_with_average_score.show()

# Chiudo la sessione
spark.stop()

