"""
DATA LOADING SCRIPT

==============

Script handles all new data digestion, preprocessing, and construction of the context vectors for the
MovieLens 10M dataset. Nothing algorithmic is executed here, just a procedure taking clean data in and
producing the clean feature matrices out. 

The script expects raw files from the MovieLens 10M dataset to take the following form:
    - ratings.dat  : UserID::MovieID::Rating::Timestamp
    - movies.dat   : MovieID::Title::Genres
    - tags.dat     : UserID::MovieID::Tag::Timestamp

Usage:
    from data_loader import load_and_prepare

    df, context_dim = load_and_prepare(
        ratings_path="data/ratings.dat",
        movies_path="data/movies.dat",
        tags_path="data/tags.dat",       # optional
        context_method="raw",            # or "pca"
        pca_dim=20,
        reward_threshold=4.0,
    )
"""

# Import dependencies
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import time
import sys


# ---------------------------------------------------------------------------
# LOAD RAW FILES
# ---------------------------------------------------------------------------
def load_ratings(path: str) -> pd.DataFrame:
    """
    Function to load raw ratings into a DataFrame. 

    Parameters
    -----------
    * path: file path to the ratings.dat file

    Returns
    ----------
    * DataFrame with columns user_id, movie_id, rating, timestamp, ordered by user_id and then
    by movie_id (as shipped by MovieLens). Ratings are on a 5-star scale with half-star increments.
    """
    # Read dat file
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="utf-8",
    )
    # Reformat DF columns
    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)
    return df


def load_movies(path: str) -> pd.DataFrame:
    """
    Function to load raw movies into a DataFrame.

    Parameters
    -----------
    * path: file path to the movies.dat file

    Returns
    ----------
    * DataFrame with columns movie_id, title, genres, where 'genres' is kept as the raw pipe-separated
    string here.
    """
    # Read dat file
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["movie_id", "title", "genres"],
        encoding="utf-8",
    )
    # Reformat movie_id column for easier sorting
    df["movie_id"] = df["movie_id"].astype(int)
    return df


def load_tags(path: str) -> pd.DataFrame:
    """
    Function to load raw tags into a DataFrame. Tags correspond to user-generated free-text metadata and
    are optional, in the sense that the file will only be used if tag-based features are requested.

    Parameters
    -----------
    * path: file path to the tags.dat file

    Returns
    ----------
    * DataFrame with columns user_id, movie_id, tag, timestamp.
    """
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "tag", "timestamp"],
        encoding="utf-8",
    )
    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)
    return df


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING SUPPORT
# ---------------------------------------------------------------------------
# List all 18 genres present in dataset
ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def encode_genre_features(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to expand the pipe-separated genres column into 18 binary indicator columns, one per genre
    (e.g. genre_Action, genre_Comedy, …).

    Parameters
    ----------
    * movies_df: a DataFrame containing the movies data

    Returns
    ----------
    * pd.DataFrame: a copy of the original movies DataFrame with the new genre columns appanded (the
    original is kept for reference).
    """
    # Make a copy to keep original
    movies_df = movies_df.copy()
    # Iterate through list of genres and create column for each
    for genre in ALL_GENRES:
        movies_df[_genre_col(genre)] = movies_df["genres"].str.contains(genre, regex=False).astype(float)
    # Return modified version 
    return movies_df


def compute_user_stats(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to compute per-user aggregate statistics from the ratings log. These statistics are from
    the *full ratings log* (i.e., before any train/test split) because they describe general user behavior
    and not held-out labels. 

    (Note: if we want a stricter setup, we can amend this function to compute the statistics only on the
    training rows and to pass the result when it is time to merge.)

    Parameters
    -----------
    * ratings_df: a DataFrame containing all ratings (before training/testing split)

    Returns
    -----------
    * pd.DataFrame: a DataFrame, organized by user_id, with the features user_mean_rating (mean rating
    given by a user), user_rating_count (total number of ratings given by the user – to determine the 
    user's activity level), and user_rating_std (standard deviation of this user's ratings – to indicate
    how discriminating the user is).
    """
    # Compute major statistics by grouping ratings by user
    stats = ratings_df.groupby("user_id")["rating"].agg(
        user_mean_rating="mean",
        user_rating_count="count",
        user_rating_std="std",
    ).reset_index()
    # Fill in std column with 0s
    stats["user_rating_std"] = stats["user_rating_std"].fillna(0.0)
    return stats


def compute_item_stats(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to compute per-item aggregate statistics from the ratings log. 

    Parameters
    -----------
    * ratings_df: a DataFrame containing all ratings. 

    Returns
    ----------
    * pd.DataFrame: a DataFrame, organized by movie_id, with the features item_mean_rating (mean rating
    by the movie), item_rating_count (total number of rating, indicating popularity), and item_rating_std
    (standard deviation of ratings, indicating polarization of the movie).
    """
    # Compute major statistics by grouping ratings by movie
    stats = ratings_df.groupby("movie_id")["rating"].agg(
        item_mean_rating="mean",
        item_rating_count="count",
        item_rating_std="std",
    ).reset_index()
    # Fill in std column with 0s
    stats["item_rating_std"] = stats["item_rating_std"].fillna(0.0)
    return stats


# ---------------------------------------------------------------------------
# PREPARE DATASET
# ---------------------------------------------------------------------------
def merge_dataset(ratings_df: pd.DataFrame, movies_df: pd.DataFrame,) -> pd.DataFrame:
    """
    Function to join ratings with movie features (genre indicators and item statistics) and user statistics
    into a single, interactive table. Each row in this joined table represents one logged interaction and
    contains:
        - user_id, movie_id, rating, timestamp
        - 18 binary genre indicator columns
        - user_mean_rating, user_rating_count, user_rating_std
        - item_mean_rating, item_rating_count, item_rating_std

    As a note, the MovieLens 10M file does not include any user demographic information (unlike its 1M
    version), so the user features are derived entirely from interaction history. 

    Parameters
    -----------------
    * ratings_df: a DataFrame containing the movie ratings 
    * movies_df: a DataFrame containing the movie data themselves

    Returns
    ----------------
    * pd.DataFrmae: a DataFrame containing all of the merged information together.
    """
    # Enrich movies with genre flags
    movies_enriched = encode_genre_features(movies_df)

    # Compute aggregate statistics
    user_stats = compute_user_stats(ratings_df)
    item_stats = compute_item_stats(ratings_df)

    # Merge everything onto the ratings table
    df = ratings_df.copy()
    df = df.merge(movies_enriched.drop(columns=["title", "genres"]), on="movie_id", how="left")
    df = df.merge(user_stats, on="user_id", how="left")
    df = df.merge(item_stats, on="movie_id", how="left")

    return df


# ---------------------------------------------------------------------------
# RECONSTRUCT REWARDS AS BINARY OUTCOMES
# ---------------------------------------------------------------------------
def binarize_rewards(df: pd.DataFrame, threshold: float = 4.0) -> pd.DataFrame:
    """
    Function to convert continuous rating column into binary reward signal. The reward is assigned a value
    of 1 if the rating is at least the provided threshold (i.e., corresponding to a 'liked' recommendation)
    and is assigned a value of 0 otherwise. 

    The default threshold of 4.0 follows common practice in bandit recommendation literature, as exhibited
    by Li et al. (2010) who used the same threshold for new clicks in their online recommendation context.

    Parameters
    ------------
    * df: a DataFrame containing the entire data
    * threshold: a double corresponding to the criterion for 'liked' recommendations

    Returns
    ----------
    * pd.DataFrame: a copy of the original DataFrame with a new 'reward' column added, corresponding to
    the binary, discretized reward value.
    """
    df = df.copy()
    df["reward"] = (df["rating"] >= threshold).astype(float)
    return df


# ---------------------------------------------------------------------------
# CONTEXT VECTOR CONSTRUCTION
# ---------------------------------------------------------------------------
def filter_sprarse_users(df: pd.DataFrame, min_ratings=20) -> pd.DataFrame: 
    """
    Function to filter out users who have fewer than (min_ratings) number of ratings associated with them.
    This is designed to help eliminate users whose interaction history might be uninformative or not
    sufficiently comprehensive for accurate recommendations to be made.

    Parameters
    -----------
    * df: a DataFrame containing all of the users
    * min_ratings: integer threshold for minimum number of ratings a user must have

    Returns
    ----------
    * pd.DataFrame: version of original DataFrame, having filter out largely inactive users.
    """
    df = df.copy()
    return df.groupby("user_id").filter(lambda x: len(x) >= min_ratings)

def filter_top_arms(df: pd.DataFrame, max_arms: int = 50) -> pd.DataFrame:
    """
    Restrict to the top-K most frequently rated movies.
    This directly increases offline replay match rate by concentrating
    the arm pool on items that appear often in the logged stream.
    """
    top_movies = (
        df["movie_id"]
        .value_counts()
        .head(max_arms)
        .index
    )
    return df[df["movie_id"].isin(top_movies)].reset_index(drop=True)



def _genre_col(genre: str) -> str:
    """
    Function to generate canonical column name for a genre indicator.

    Parameters
    -----------
    * genre: a string corresponding to the genre of the film

    Returns
    -------------
    * str: canonical column name for the genre listed. """
    return f"genre_{genre.replace(' ', '_').replace(chr(39), '')}"

# Columns that form the raw feature vector for each interaction
CONTEXT_FEATURE_COLS = (
    [_genre_col(g) for g in ALL_GENRES]
    + ["user_mean_rating", "user_rating_count", "user_rating_std"]
    + ["item_mean_rating", "item_rating_count", "item_rating_std"]
)




def build_context_vectors(df: pd.DataFrame,method: str = "raw", pca_dim: int = 20, scaler: StandardScaler = None,
    pca: PCA = None, fit: bool = True,) -> tuple[pd.DataFrame, int, StandardScaler, PCA | None]:
    """
    Function to construct and (optionally) reduce context feature vectors. 

    Parameters
    ----------
    * df: merged interaction DataFrame (output of merge_dataset)
    * method: a flag to determine whether to use all the context features as-is (after scaling) or to
      project down to number of pca dimensions after scaling
    * pca_dim: target dimensionality when method='pca'
    * scaler: a pre-fitted StandardScaler (pass when transforming test data)
    * pca: a pre-fitted PCA object (pass when transforming test data)
    * fit: flag to fit scaler (and PCA) on this data (if False, only transform)

    Returns
    -------
    * df_out: copy of df with a new 'context' column containing np.ndarray vectors
    * context_dim: integer dimension of each context vector
    * scaler: fitted StandardScaler (reuse on test data)
    * pca: fitted PCA object (or None if method='raw')
    """
    # Get context feature values
    X = df[CONTEXT_FEATURE_COLS].values.astype(float)
    # Fill any NaNs that slipped through (e.g. items with no prior ratings)
    X = np.nan_to_num(X, nan=0.0)

    # Standardise
    if scaler is None: scaler = StandardScaler()
    if fit:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    # Optional PCA reduction
    if method == "pca":
        if pca is None: pca = PCA(n_components=pca_dim, random_state=42)
        if fit:
            X = pca.fit_transform(X)
        else:
            X = pca.transform(X)
        context_dim = pca_dim
    else:
        pca = None
        context_dim = X.shape[1]

    df_out = df.copy()
    # Store each row's context as a 1-D numpy array inside a Python object column
    df_out["context"] = [X[i] for i in range(len(X))]

    return df_out, context_dim, scaler, pca


# ---------------------------------------------------------------------------
# GENERATE TRAIN/TEST SPLITS
# ---------------------------------------------------------------------------
def train_test_split_temporal(df: pd.DataFrame, test_fraction: float = 0.2,) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to split user interactions into training and testing sets by timestamp. The choice of timestamp
    as the splitting criterion is to avoid look-ahead bias in the various algorithms; an alternative choice
    could have been by user, so as not to memorize any particular user's preference. The (1 - test_fraction)
    earliest events, sorted by timestamp, become training data, while the remainder fill in the test set. 

    This function tries to imitate the operation of a deployed recommender: it is trained exclusively on
    historical interactions and evaluated on future ones.

    Parameters
    ----------
    * df: full merged interaction DataFrame
    * test_fraction : fraction of interactions (by time) to hold out for test

    Returns
    -------
    * pd.DataFrame: a DataFrame corresponding to the training data
    * pd.DataFrame: a DataFrame corresponding to the testing data
    """
    # Sort the data by timestamp and split according to splitting threshold
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_fraction))
    # Use sorted DF to get train/test split and return both
    train_df = df_sorted.iloc[:split_idx].reset_index(drop=True)
    test_df = df_sorted.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# PIPELINE FUNCTION PUTTING EVERYTHING TOGETHER
# ---------------------------------------------------------------------------
def load_and_prepare(ratings_path: str, movies_path: str, tags_path: str = None,       # reserved for future tag-based features
    context_method: str = "raw", pca_dim: int = 20, reward_threshold: float = 4.0, test_fraction: float = 0.2,
    max_arms: int=50, verbose: bool = True,) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Function to perform entire data loading pipeline. The function loads the raw files, merges each part
    together, reconstructs the reward to be binary, builds the context vectors, and temporally splits the
    data for training and testing purposes.

    Parameters
    ----------
    * ratings_path: path to ratings.dat
    * movies_path: path to movies.dat
    * tags_path: path to tags.dat (unused for now; reserved)
    * context_method: 'raw' or 'pca'
    * pca_dim: PCA target dimension (only used when context_method='pca')
    * reward_threshold: rating threshold for binary reward (default 4.0)
    * test_fraction: fraction of data (by time) held out for test
    * verbose: if True, print dataset statistics

    Returns
    -------
    * pd.DataFrame: a DataFrame containing training interactions, with 'context' and 'reward' columns
    * pd.DataFrame: a DataFrame containing testing interactions, with 'context' and 'reward' columns
    * int: an integer corresponding to the dimension of each context vector
    """
    if verbose: print("Loading raw files...")
    ratings = load_ratings(ratings_path)
    movies = load_movies(movies_path)

    if verbose:
        print(f"  Ratings: {len(ratings):,} rows | "
              f"Users: {ratings['user_id'].nunique():,} | "
              f"Movies: {ratings['movie_id'].nunique():,}")

    if verbose: print("\nMerging and filtering dataset, then engineering features...")
    df = merge_dataset(ratings, movies)
    df = binarize_rewards(df, threshold=reward_threshold)
    df = filter_sprarse_users(df)
    if max_arms is not None:
        if verbose:
            print(f"  Filtering to top {max_arms} arms by popularity...")
        df = filter_top_arms(df, max_arms=max_arms)
        if verbose:
            print(f"  Post-filter interactions: {len(df):,} | "
                f"Arms: {df['movie_id'].nunique()}")

    if verbose:
        pos_rate = df["reward"].mean()
        print(f"  Reward rate at threshold={reward_threshold}: {pos_rate:.3f}")

    if verbose: print(f"\nBuilding context vectors (method='{context_method}')...")
    df, context_dim, scaler, pca = build_context_vectors(
        df, method=context_method, pca_dim=pca_dim, fit=True
    )

    if verbose: print(f"  Context dimensionality: {context_dim}")

    if verbose: print("\nSplitting train / test by timestamp...")
    train_df, test_df = train_test_split_temporal(df, test_fraction=test_fraction)

    if verbose:
        print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")
        print("Done.\n")
    return train_df, test_df, context_dim


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()

    if len(sys.argv) < 3:
        print("Usage: python data_loader.py <ratings.dat> <movies.dat>")
        sys.exit(1)

    train_df, test_df, context_dim = load_and_prepare(
        ratings_path=sys.argv[1],
        movies_path=sys.argv[2],
        context_method="raw",
        verbose=True,
    )

    print("Sample training row:")
    row = train_df.iloc[0]
    print(f"  user={row['user_id']}  movie={row['movie_id']}  "
          f"reward={row['reward']}  context_shape={row['context'].shape}")
    
    end_time = time.time()
    print(f"\nData loading executed in {end_time - start_time:.3f} seconds ({(end_time - start_time)/60:.2f} minutes).")