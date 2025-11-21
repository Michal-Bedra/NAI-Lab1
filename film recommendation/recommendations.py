import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import urllib.parse
import requests

"""
Program analizuje wystawione przez użytkowników oceny, które znajdują się w w bazie danych. Na podstawie tychże
ocenia podobieństwo użytkowników, aby znaleźć filmy które z dużą dozą prawodpodobieństwa spodobają się (lub nie)
wybranemu użytkownikowi. Dodakowo program pobiera informacje o filmach z bazy IMDB.
Przykład użycia i konkretnych wywołań widoczne na załączonych screenach.
https://scikit-learn.org/stable/getting_started.html
https://pandas.pydata.org/docs/user_guide/index.html
https://numpy.org/doc/stable/user/absolute_beginners.html
https://pypi.org/project/requests/
Filip Patuła s28615, Michał Bedra s28854
"""
#Movies database query
MOVIES_SEARCH_QUERY = 'https://imdb.iamidiotareyoutoo.com/search?q='
#Movie ratings data filename
FILE_NAME = "ratings_group.json"
#Mean reduction value for cluster fitting optimisation
high_mean_reduction = 2

def build_arg_parser():
    """ Creates arg parser for command line parameters for movies recommendation system
        Parameters:
        None
        Returns:
        ArgumentParser: argument parser with all command line parameters
    """
    parser = argparse.ArgumentParser(
        description='User-based recommendations from ratings_group.json'
    )
    parser.add_argument(
        '--username',
        required=False,
        type=str,
        default='',
        help='User for whom we want recommendations'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='How many recommendations / anti-recommendations to show (default: 5)'
    )
    parser.add_argument(
        '--recommend',
        default=True,
        action=argparse.BooleanOptionalAction,
        help='Should present movies recommendations/anti-recommendations (--no-recommend for anti-recommendations)'
    )
    return parser

def recommend_movies(movies_ratings_data, users_clusters):
    """ Recommend movies based on user clusters and movies ratings
        Parameters:
        movies_ratings_data: dataframe with movie ratings for each user
        users_clusters: array of users clusters
        Returns:
        movies: array with movie titles
    """
    cluster_data_frame = get_cluster_data_frame(movies_ratings_data, users_clusters)
    selected_user_cluster = get_user_cluster(cluster_data_frame)
    return get_movies_recommendation(cluster_data_frame, selected_user_cluster, movies_ratings_data)

def get_movies_recommendation(cluster_data_frame, selected_user_cluster, movies_ratings_data):
    """ Get movies recommendation based on user cluster (if exists) and movies ratings
        Parameters:
        movies_ratings_data: dataframe with movie ratings for each user
        selected_user_cluster: user cluster label
        cluster_data_frame: dataframe with user clusters
        Returns:
        movies: array with movie titles
    """
    if selected_user_cluster is not None:
        return get_movies_recommendations_for_cluster(cluster_data_frame, movies_ratings_data, selected_user_cluster)
    else:
        return get_random_movies_recommendations(movies_ratings_data)

def get_random_movies_recommendations(movies_ratings_data):
    """ Get random movies recommendation based on movies ratings
        Parameters:
        movies_ratings_data: dataframe with movie ratings for each user
        Returns:
        movies: array with movie titles
    """
    selected_movies = np.array([])
    movies_ranks_sorted = movies_ratings_data.mean(axis='columns').round().sort_values()
    selected_movies = np.append(selected_movies, movies_ranks_sorted.tail(movies_to_recommend).index if recommend else movies_ranks_sorted.head(movies_to_recommend).index)
    return selected_movies

def get_movies_recommendations_for_cluster(cluster_data_frame, movies_ratings_data, selected_user_cluster):
    """ Get movies recommendation based on user cluster and movies ratings, fills returned title list with data from other clusters if data for cluster is limited
        Parameters:
        movies_ratings_data: dataframe with movies ratings for each user
        selected_user_cluster: user cluster label
        cluster_data_frame: dataframe with user clusters
        Returns:
        movies: array with movie titles
    """
    selected_movies = np.array([])
    users_in_the_same_cluster = cluster_data_frame.loc[
        cluster_data_frame['cluster_fit'] == selected_user_cluster.values[0]].drop(index=selected_user_cluster.name)
    rating_data = movies_ratings_data.loc[:, users_in_the_same_cluster.index]
    user_movies = movies_ratings_data.loc[:, username].dropna()
    sorted_ranked_movies = rating_data.drop(index=user_movies.index).mean(axis='columns').dropna().round().sort_values()
    movies_to_recommend_left = movies_to_recommend
    while movies_to_recommend_left > 0:
        selected_rank = sorted_ranked_movies.iloc[-1 * recommend] if len(sorted_ranked_movies) != 0 else -1
        movies_to_add = sorted_ranked_movies.loc[sorted_ranked_movies == selected_rank]
        if len(movies_to_add) > movies_to_recommend_left:
            selected_movies = np.append(selected_movies, movies_to_add.sample(n=movies_to_recommend_left).index)
            movies_to_recommend_left = 0
        elif len(movies_to_add) == movies_to_recommend_left:
            selected_movies = np.append(selected_movies, movies_to_add.index)
            movies_to_recommend_left = 0
        elif len(movies_to_add) == 0:
            other_movies_ranks_sorted = movies_ratings_data.drop(columns=username).drop(
                index=user_movies.index).drop(index=selected_movies).mean(
                axis='columns').dropna().round().sort_values()
            selected_movies = np.append(selected_movies, other_movies_ranks_sorted.tail(
                movies_to_recommend_left).index if recommend else other_movies_ranks_sorted.head(
                movies_to_recommend_left).index)
            movies_to_recommend_left = 0
        else:
            movies_to_recommend_left -= len(movies_to_add)
            selected_movies = np.append(selected_movies, movies_to_add.index)
            sorted_ranked_movies = sorted_ranked_movies.drop(index=movies_to_add.index)
    return selected_movies


def get_cluster_data_frame(movies_ratings_data, users_cluster):
    """ Get cluster data from users clusters
        Parameters:
        movies_ratings_data: dataframe with movies ratings for each user
        users_clusters: array of users clusters
        Returns:
        cluster_data_frame: data frame with users clusters
    """
    return pd.DataFrame(data=users_cluster, index=movies_ratings_data.columns,
                        columns=["cluster_fit"])

def get_user_cluster(cluster_data_frame):
    """ Get cluster label for user from cluster data frame
        Parameters:
        cluster_data_frame: data frame with users clusters
        Returns:
        user cluster: user cluster label
    """
    try:
        user_cluster = cluster_data_frame.loc[username, :]
    except KeyError:
        user_cluster = None
    return user_cluster

def fit_model_and_get_users_clusters(movies_ratings_data):
    """ Fits clustering model and assign clusters to users
        Parameters:
        movies_ratings_data: dataframe with movies ratings for each user
        Returns:
        users_clusters: array of users clusters
    """
    movies_ratings_data_filled = movies_ratings_data.fillna(value=questionary_ratings_data.mean(axis='rows').round().sub(high_mean_reduction)).transpose()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(movies_ratings_data_filled)
    cluster = AgglomerativeClustering(n_clusters=4)
    users_cluster_fit = cluster.fit_predict(scaled_data)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled_data)
    df_plot = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": cluster.labels_
    })
    plt.scatter(df_plot["x"], df_plot["y"], c=df_plot["cluster"])
    plt.title("Users Clusters")
    plt.show()
    return users_cluster_fit


def show_movies_recommendation(recommended_movies):
    """ Display recommended movies
        Parameters:
        recommended_movies: array of movies recommended user
        Returns:
        None
    """
    print("Movies recommendation:")
    for movie in recommended_movies:
        encoded_movie = urllib.parse.quote(string = movie, encoding="utf-8")
        movie_response = requests.get(MOVIES_SEARCH_QUERY + encoded_movie)
        movie_data = movie_response.json()
        movie_details = movie_data['description'][0]
        print(f"Title: %r" % movie_details['#TITLE'])
        print(f"Year of production: %d" % movie_details['#YEAR'])
        print(f"Known actors: %r" % movie_details['#ACTORS'])

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    username = args.username
    movies_to_recommend = args.top_n
    recommend = args.recommend

    with open("%s" % FILE_NAME, "r", encoding="utf-8") as file:
        questionary_ratings_dict = json.load(file)

    questionary_ratings_data = pd.DataFrame.from_dict(questionary_ratings_dict)

    cluster_fit = fit_model_and_get_users_clusters(questionary_ratings_data)

    movies = recommend_movies(questionary_ratings_data, cluster_fit)

    show_movies_recommendation(movies)
