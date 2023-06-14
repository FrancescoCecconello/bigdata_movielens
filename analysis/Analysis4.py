from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, format_number, floor, sum, count, avg
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Creo una sessione Spark
spark = SparkSession.builder.appName("Analysis3").getOrCreate()

# Carico il file "ratings.csv" nel dataframe
ratings = spark.read.format("csv").option("header", "true").load("ratings.csv")

# Converto la colonna "rating" in float
ratings = ratings.withColumn("rating", col("rating").cast("float"))

# Calcolo il rating medio per ogni utente. Per farlo raggruppo i rating per "userId"
# e poi applico la funzione avg, rinominando la colonna generata "AverageRating"
average_ratings_per_user = ratings.groupBy("userId").agg(avg("rating").alias("AverageRating"))

# Mostro la tabella con il rating medio per utente
average_ratings_per_user.show()

# Calcolo la media delle valutazioni per ogni utente e il numero totale di rating per ogni gruppo.
# Raggruppo i rating per "userId", poi genero due nuove colonne. Nella prima colonna "Average"
# inserisco il rating medio per utente, nella seconda colonna "NumRatings" inserisco il numero
# di rating per utente
average_ratings = ratings.groupBy("userId").agg(mean(col("Rating")).alias("Average"),count(col("Rating")).alias("NumRatings")
                                                 )

# Calcolo il punteggio medio arrotondato e mostro il numero totale di rating per ogni gruppo
average_ratings = average_ratings.withColumn("RoundedAverage", floor(col("Average") / 0.5) * 0.5).groupBy("RoundedAverage").agg(format_number("RoundedAverage", 1).alias("Average"),sum("NumRatings").alias("TotalRatings")).orderBy("RoundedAverage")

# Inserisco i dati in una lista
rounded_average = average_ratings.select("RoundedAverage").rdd.flatMap(lambda x: x).collect()
total_ratings = average_ratings.select("TotalRatings").rdd.flatMap(lambda x: x).collect()

# Interpolazione dei dati per rendere il grafico più smooth
spline = make_interp_spline(rounded_average, total_ratings)

# Generazione di punti intermedi per una rappresentazione più liscia
smooth_rounded_average = np.linspace(min(rounded_average), max(rounded_average), 500)
smooth_total_ratings = spline(smooth_rounded_average)

# Imposto un valore minimo per i total ratings, poiché con l'interpolazione
# alcuni valori diventano negativi ed è impossibile, visto che non possono
# esistere numeri negativi di rating
min_total_ratings = 0
smooth_total_ratings = np.clip(smooth_total_ratings, min_total_ratings, None)

# Creo il grafico
plt.plot(smooth_rounded_average, smooth_total_ratings)
plt.xlabel("Rating medio per utente")
plt.ylabel("Numero totale di rating")
plt.grid(True)

plt.show()

# Chiudo la sessione
spark.stop()

