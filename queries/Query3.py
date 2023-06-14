from pyspark.sql import SparkSession
from pyspark.sql.functions import from_unixtime, year, count, avg, col
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Creo una sessione Spark
spark = SparkSession.builder.appName("Query3").getOrCreate()

# Carico il file "ratings.csv" in un dataframe
ratings = spark.read.format("csv").option("header", "true").load("ratings.csv")

# Calcolo il numero di film valutati da ciascun utente. Raccolgo per "userId"
# e poi li conto
user_movie_counts = ratings.groupBy("userId").agg(count("movieId").alias("movie_count"))

# Definisco i limiti dei gruppi di utenti basati sul numero di film valutati
group_limits = [0, 20, 30, 50, 100, 200, 500, 1000]

# Calcolo l'andamento della valutazione media dei film per i diversi gruppi di utenti
average_ratings = []
x_labels = []

for i in range(1, len(group_limits)):
    # Filtro gli utenti che hanno valutato almeno il numero minimo di film
    filtered_users = user_movie_counts.filter(user_movie_counts.movie_count > group_limits[i - 1]).filter(
        user_movie_counts.movie_count <= group_limits[i])

    # Calcolo la valutazione media di ogni film per gli utenti filtrati
    average_rating = ratings.join(filtered_users, on="userId").groupBy("movieId").agg(
        avg("rating").alias("avg_rating"))

    # Calcolo la valutazione media complessiva di tutti i film
    overall_average = ratings.groupBy("movieId").agg(avg("rating").alias("overall_avg_rating"))

    # Calcolo la valutazione media dei film per il gruppo di utenti corrente
    result = average_rating.join(overall_average, on="movieId").select("movieId", "avg_rating", "overall_avg_rating")

    # Calcolo la valutazione media totale per il gruppo di utenti corrente
    total_avg_rating = result.select(avg("avg_rating")).first()[0]

    average_ratings.append(total_avg_rating)

    if i == 1:
        x_labels.append("<= {}".format(group_limits[i]))
    else:
        x_labels.append("{} < n <= {}".format(group_limits[i - 1], group_limits[i]))

# Interpolazione per rendere il grafico più smooth
spline = make_interp_spline(range(len(average_ratings)), average_ratings)
x_smooth = np.linspace(0, len(average_ratings) - 1, 1000)
y_smooth = spline(x_smooth)
plt.figure(figsize=(16, 9))
plt.plot(x_smooth, y_smooth)
plt.xlabel("Numero di Film Valutati dagli Utenti")
plt.ylabel("Valutazione Media")
plt.xticks(range(len(x_labels)), x_labels)
plt.grid(True)
plt.show()

# Chiudo la sessione
spark.stop()

