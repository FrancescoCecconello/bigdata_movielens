from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Creo una sessione Spark
spark = SparkSession.builder.appName("Analysis1").getOrCreate()

# Carico il file "ratings.csv" nel dataframe
ratings = spark.read.format("csv").option("header", "true").load("ratings.csv")

# Calcolo il numero di valutazioni per ogni film. Per farlo raggruppo i rating per movieId
# e li metto in ordine decrescente. Ritorno una tabella composta da due colonne, "movieId"
# e "count", rinominata "Number of Ratings"
ratings_per_movie = ratings.groupBy("movieId").count().orderBy("count", ascending=False).select("movieId", col("count").alias("Number of Ratings"))

# Mostro la tabella
ratings_per_movie.show()

# Calcolo il numero medio di rating per film applicando la funzione avg alla colonna "Number of Ratings"
mean_ratings = ratings_per_movie.select(avg("Number of Ratings").alias("Mean")).first()["Mean"]

# Stampo il valore medio del numero di rating per film
print("Valore medio del numero di rating per film:", mean_ratings)

# Inserisco i valori della colonna "Number of Ratings" in un array NumPy di interi per
# poterli usare con matplotlib
counts = np.array(ratings_per_movie.select("Number of Ratings").rdd.flatMap(lambda x: x).collect(), dtype=int)

# Calcolo la densità della distribuzione utilizzando KDE
kde = gaussian_kde(counts)
x = np.linspace(0, 200, 200)
y = kde(x)

# Moltiplico i valori di y per la scala appropriata per ottenere la densità
# Se non lo facessi otterrei dei valori difficilmente leggibili sull'asse y
# In questo modo ottengo il numero di rating
y_scaled = y * np.sum(counts)

plt.figure(figsize=(16, 9))

# Creo il grafico
plt.plot(x, y_scaled, color="blue", linewidth=2)

plt.xlim([0, 200])
plt.xlabel("Numero di rating per film")
plt.ylabel("Numero di film stimato")

plt.show()

# Chiudo la sessione
spark.stop()
