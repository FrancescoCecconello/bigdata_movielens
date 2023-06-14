from pyspark.sql import SparkSession
from pyspark.sql.functions import col, array
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd


# Creo una sessione Spark
spark = SparkSession.builder.appName("MovieClustering").getOrCreate()

# Carico il file "ratings.csv" in un dataframe
ratings = spark.read.format("csv").option("header", "true").load("ratings.csv")

# Carico il file "movies.csv" in un dataframe
movies = spark.read.format("csv").option("header", "true").load("movies.csv")

# Devo creare delle coppie [utente,rating] per poterle usare con kmeans
def create_movie_rating_couple(user, rating):
    return array(col("userId").cast("float"), col("rating").cast("float"))

# Il dataframe sarà consistente, per cui posso buttare via le colonne che non
# servono a kmeans, come timestamp, title, genre, userId e rating, visto che sono
# già contenute nelle coppie create con create_movie_rating_couple
grouped_ratings = ratings.join(movies, "movieId").drop("timestamp", "title", "genres").withColumn("movie_rating", create_movie_rating_couple(col("userId"), col("rating"))).drop("userId", "rating").orderBy("movieId").toPandas()

# Converto i dati per poterli usare in kmeans
movie_ratings_data = {}
grouped_movies = grouped_ratings.groupby("movieId")["movie_rating"].apply(list)
for movie_index, ratings_list in enumerate(grouped_movies, start=1):
    movie_ratings_data[movie_index] = {}
    for rating_tuple in ratings_list:
        user_id = int(rating_tuple[0])
        rating_value = int(rating_tuple[1])
        movie_ratings_data[movie_index][user_id] = rating_value

# Uso pandas per riempire i "buchi" con la media, in modo da non modificare
# la varianza usando PCA
ratings = pd.DataFrame(movie_ratings_data)
ratings.fillna(ratings.mean(), inplace=True)

# Uso kmeans con n=4 cluster da individuare
kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit_predict(ratings.transpose())

# Riduco le dimensioni del dataframe con PCA
dataframe_PCA = PCA(n_components=2)
pc = dataframe_PCA.fit_transform(ratings.transpose())

# Creo un nuovo dataframe che contenga le due colonne releative
# alle due principal components
principal_components = pd.DataFrame(
    data = pc,
    columns = ["PC1", "PC2"],
    index = ratings.columns.tolist()
)
principal_components["cluster"] = clusters

# Uso seaborn per la visualizzazione a punti dei cluster
cluster_colors = ["orange", "green", "red", "blue", "black"]
sns.scatterplot(
    data=principal_components,
    x="PC1",
    y="PC2",
    hue="cluster",
    palette=cluster_colors
)

# Mostro i cluster
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

