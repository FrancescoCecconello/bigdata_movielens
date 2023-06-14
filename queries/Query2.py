from pyspark.sql import SparkSession
from pyspark.sql.functions import from_unixtime, year, count, expr, date_add, avg, col, stddev
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F

# Creo una sessione Spark
spark = SparkSession.builder.appName("Query2").getOrCreate()

# Carico il file "ratings.csv" nel dataframe
ratings = spark.read.format("csv").option("header", "true").load("ratings.csv")

# Converto la colonna timestamp da formato unix a formato data, rinominandola "date"
ratings = ratings.withColumn("date", from_unixtime(ratings.timestamp))

# Estraggo l'anno dalla data per ogni data e aggiungo la colonna "year" corrispondente
ratings = ratings.withColumn("year", year(ratings.date))

# Raccolgo i rating per anno, poi calcolo la media per ogni anno e
# la inserisco nella colonna "avg_rating"
avg_ratings = ratings.groupBy("year").agg(avg("rating").alias("avg_rating")).orderBy("year")

# Calcolo il numero di rating per ogni anno
count_ratings = ratings.groupBy("year").agg(count("rating").alias("rating_count")).orderBy("year")

# Per ogni anno estraggo il rating medio
years_avg = [row["year"] for row in avg_ratings.collect()]
avg_values = [row["avg_rating"] for row in avg_ratings.collect()]

# Per ogni anno estraggo il numero di rating
years_count = [row["year"] for row in count_ratings.collect()]
count_values = [row["rating_count"] for row in count_ratings.collect()]

# Interpolazione dei dati per un grafico più smooth
spline = make_interp_spline(years_avg, avg_values)
smooth_years_avg = np.linspace(min(years_avg), max(years_avg), 500)
smooth_avg_values = spline(smooth_years_avg)
X_Y_Spline_count = make_interp_spline(years_count, count_values)
smooth_years_count = np.linspace(min(years_count), max(years_count), 500)
smooth_count_values = X_Y_Spline_count(smooth_years_count)
labels_avg = [str(year) for year in years_avg]
labels_count = [str(year) for year in years_count]

# Creo due plot per i due grafi
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# GRAFO 1: Mostro l'evoluzione del rating medio nel tempo
ax1.plot(smooth_years_avg, smooth_avg_values)
ax1.set_title("Evoluzione del rating medio dei film")
ax1.set_xlabel("Anno")
ax1.set_ylabel("Rating medio")
ax1.set_xticks(years_avg)
ax1.set_xticklabels(labels_avg, rotation=45)

# GRAFO 2: Mostro l'evoluzione del numeri di rating nel tempo
ax2.plot(smooth_years_count, smooth_count_values)
ax2.set_title("Variazione del numero di rating ogni anno")
ax2.set_xlabel("Anno")
ax2.set_ylabel("Numero di Rating")
ax2.set_xticks(years_count)
ax2.set_xticklabels(labels_count, rotation=45)
plt.tight_layout()

# Creo i grafici
plt.show()

# Faccio il join del numero di rating per anno e del rating medio per anno
# sulla colonna "year"
combined_ratings = count_ratings.join(avg_ratings, "year")

# Calcolo la correlazione di Pearson fra la colonna "rating_count" e "avg_rating",
# cioè fra il numero di rating per anno e il rating medio per anno
pearson_corr = combined_ratings.stat.corr("rating_count", "avg_rating", method="pearson")
print("Correlazione di Pearson:", pearson_corr)

########################################################################################

# Creo la finestra di partizionamento per "movieId" ordinandola in ordine temporale
windowSpec = Window.partitionBy("movieId").orderBy("timestamp")

# Aggiungo la colonna "rank" alla tabella dei rating, dove il rank è un numero
# intero >=1 che indica l'ordine di inserimento del rating
ranked_ratings = ratings.withColumn("rank", F.row_number().over(windowSpec))

# Conversione della colonna "rating" in double
filtered_ratings = ranked_ratings.groupBy("movieId").agg(count("rating").alias("rating_count")).filter(col("rating_count") >= 10).join(ranked_ratings, "movieId").orderBy("timestamp")

# Filtraggio dei primi 20 rating per ogni film (dato desunto dall'analisi 1)
filtered_ratings = filtered_ratings.filter(filtered_ratings.rank <= 20).orderBy("timestamp")

# Calcolo la media dello score per ogni film
average_init_ratings = filtered_ratings.groupBy("movieId").agg(avg("rating").alias("average_rating"))

# Suddivido i film fra "initially high rated" e "initially low rated" con treshold 3.5 (desunta dall'analisi 3)
average_init_high_ratings = average_init_ratings.filter(col("average_rating") >= 3.5)

average_init_low_ratings = average_init_ratings.filter(col("average_rating") < 3.5)

init_high_rating_movies = average_init_high_ratings.select("movieId")

init_low_rating_movies = average_init_low_ratings.select("movieId")

average_final_high_ratings = ratings.groupBy("movieId").agg(avg("rating").alias("average_rating")).join(init_high_rating_movies, "movieId")

const_high_rated_movies = average_final_high_ratings.filter(col("average_rating") >= 3.5)

print("Rapporto tra film inizialmente con high rating e film con low rating: ", const_high_rated_movies.count() / init_high_rating_movies.count())


########################################################################################

# Prendo gli id dei film initially high rated
high_rated = ratings.join(init_high_rating_movies,"movieId")

# Prendo gli id dei film initially low rated
low_rated = ratings.join(init_low_rating_movies,"movieId")

# Calcolo il conto dei rating per anno per film high rated
high_rated_counts = high_rated.groupBy(year(from_unixtime("timestamp")).alias("year")).agg(count("*").alias("high_rated_count")).orderBy("year")

# Calcolo il conto dei rating per anno per film low rated
low_rated_counts = low_rated.groupBy(year(from_unixtime("timestamp")).alias("year")).agg(count("*").alias("low_rated_count")).orderBy("year")

# Estraggo i dati per il grafico
years = high_rated_counts.select("year").rdd.flatMap(lambda x: x).collect()
high_rated_count = high_rated_counts.select("high_rated_count").rdd.flatMap(lambda x: x).collect()
low_rated_count = low_rated_counts.select("low_rated_count").rdd.flatMap(lambda x: x).collect()

# Interpolazione dei dati per rendere il grafico più smooth
x_smooth = np.linspace(min(years), max(years), 300)
high_rated_count_smooth = make_interp_spline(years, high_rated_count)(x_smooth)
low_rated_count_smooth = make_interp_spline(years, low_rated_count)(x_smooth)

# Inserisco nello stesso grafico il count del rating per entrambi i gruppi
plt.figure(figsize=(10, 6))
plt.plot(x_smooth, high_rated_count_smooth, label="Film High Rated")
plt.plot(x_smooth, low_rated_count_smooth, label="Film Low Rated")
plt.xlabel("Anno")
plt.ylabel("Numero di rating")
plt.legend()
plt.show()

# Chiudo la sessione
spark.stop()

