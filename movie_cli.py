#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors


class MovieRecommender:
    def __init__(self, data_dir: str | Path = "data", verbose: bool = False):
        self.data_dir = Path(data_dir)
        self.verbose = verbose

        self.ratings: pd.DataFrame | None = None
        self.movies: pd.DataFrame | None = None
        self.movie_genres: pd.DataFrame | None = None

        self.M: np.ndarray | None = None
        self.M_train: np.ndarray | None = None
        self.test_df: pd.DataFrame | None = None

        self.n_users: int | None = None
        self.n_movies: int | None = None
        self.total_entries: int | None = None

        self.U: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.k: int | None = None
        self.alpha: float | None = None
        self.lambda_reg: float | None = None
        self.n_epochs: int | None = None

        self.load_data()

    def _resolve_file(self, *candidates: str) -> Path:
        for candidate in candidates:
            path = self.data_dir / candidate
            if path.exists():
                return path
        joined = ", ".join(str(self.data_dir / c) for c in candidates)
        raise FileNotFoundError(f"Could not find any of: {joined}")

    def load_data(self) -> None:
        ratings_path = self._resolve_file("u.data")
        movies_path = self._resolve_file("u.item")
        genres_path = self._resolve_file("u.genre")

        self.ratings = pd.read_csv(
            ratings_path,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
        )

        genre_names: list[str] = []
        with open(genres_path, "r", encoding="latin-1") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                genre_names.append(line.split("|")[0])

        usecols = [0, 1] + list(range(5, 5 + len(genre_names)))
        movie_cols = ["item_id", "title"] + genre_names
        self.movies = pd.read_csv(
            movies_path,
            sep="|",
            encoding="latin-1",
            header=None,
            usecols=usecols,
            names=movie_cols,
        )
        self.movie_genres = self.movies[["item_id", "title"] + genre_names].copy()

        if self.verbose:
            print(f"Loaded ratings: {len(self.ratings):,}")
            print(f"Loaded movies:  {len(self.movies):,}")
            print(f"Genres:         {', '.join(genre_names)}")

    def build_matrix(self) -> None:
        assert self.ratings is not None
        m_df = self.ratings.pivot_table(index="user_id", columns="item_id", values="rating")
        self.n_users, self.n_movies = m_df.shape
        n_ratings = len(self.ratings)
        self.total_entries = self.n_users * self.n_movies
        self.M = m_df.fillna(0).values

        if self.verbose:
            sparsity = 1 - (n_ratings / self.total_entries)
            print(f"Matrix: {self.n_users} users x {self.n_movies} movies")
            print(f"Known ratings: {n_ratings:,}")
            print(f"Sparsity: {sparsity:.1%}")

    def random_train_test(self, seed: int = 42, split: float = 0.8) -> None:
        assert self.ratings is not None
        assert self.n_users is not None and self.n_movies is not None
        all_ratings = self.ratings[["user_id", "item_id", "rating"]].copy()
        all_ratings["user_idx"] = all_ratings["user_id"] - 1
        all_ratings["item_idx"] = all_ratings["item_id"] - 1

        shuffled = all_ratings.sample(frac=1, random_state=seed).reset_index(drop=True)
        split_ix = int(split * len(shuffled))
        train_df = shuffled.iloc[:split_ix]
        self.test_df = shuffled.iloc[split_ix:]

        self.M_train = np.zeros((self.n_users, self.n_movies))
        for _, row in train_df.iterrows():
            self.M_train[int(row["user_idx"]), int(row["item_idx"])] = row["rating"]

    def init_factors(
        self,
        k: int = 20,
        alpha: float = 0.005,
        lambda_reg: float = 0.02,
        n_epochs: int = 30,
        seed: int = 42,
    ) -> None:
        assert self.n_users is not None and self.n_movies is not None
        self.k = k
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.n_epochs = n_epochs
        rng = np.random.default_rng(seed)
        self.U = rng.normal(scale=0.1, size=(self.n_users, k))
        self.V = rng.normal(scale=0.1, size=(self.n_movies, k))

    def predict(self, u: int, i: int) -> float:
        assert self.U is not None and self.V is not None
        return float(np.dot(self.U[u], self.V[i]))

    def predict_all(self) -> np.ndarray:
        assert self.U is not None and self.V is not None
        return self.U @ self.V.T

    def train(self) -> tuple[list[float], list[float]]:
        assert self.M_train is not None
        assert self.U is not None and self.V is not None
        assert self.test_df is not None
        assert self.alpha is not None and self.lambda_reg is not None and self.n_epochs is not None

        rows, cols = self.M_train.nonzero()
        train_samples = list(zip(rows, cols, self.M_train[rows, cols]))
        train_losses: list[float] = []
        test_rmses: list[float] = []

        for epoch in range(self.n_epochs):
            np.random.shuffle(train_samples)
            epoch_loss = 0.0

            for u, i, r in train_samples:
                r_hat = self.predict(u, i)
                e = r - r_hat
                u_old = self.U[u].copy()
                self.U[u] += self.alpha * (e * self.V[i] - self.lambda_reg * self.U[u])
                self.V[i] += self.alpha * (e * u_old - self.lambda_reg * self.V[i])
                epoch_loss += e**2 + self.lambda_reg * (np.sum(self.U[u] ** 2) + np.sum(self.V[i] ** 2))

            train_losses.append(epoch_loss / len(train_samples))

            preds = [
                np.clip(self.predict(int(row["user_idx"]), int(row["item_idx"])), 1, 5)
                for _, row in self.test_df.iterrows()
            ]
            rmse = float(np.sqrt(mean_squared_error(self.test_df["rating"], preds)))
            test_rmses.append(rmse)

            if self.verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch == self.n_epochs - 1):
                print(
                    f"Epoch {epoch + 1:>3}/{self.n_epochs} | "
                    f"Train loss: {train_losses[-1]:.4f} | Test RMSE: {rmse:.4f}"
                )

        return train_losses, test_rmses

    def get_user_history(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        assert self.ratings is not None and self.movies is not None
        history = self.ratings[self.ratings["user_id"] == user_id].merge(
            self.movies[["item_id", "title"]], on="item_id"
        )
        return history[["item_id", "title", "rating"]].sort_values(
            ["rating", "title"], ascending=[False, True]
        ).head(top_n)

    def get_recommendations(self, user_id: int, top_n: int = 10, genre: str | None = None) -> pd.DataFrame:
        assert self.M_train is not None and self.movies is not None
        assert self.U is not None and self.V is not None
        u = user_id - 1
        if u < 0 or u >= self.M_train.shape[0]:
            raise ValueError(f"user_id must be between 1 and {self.M_train.shape[0]}")

        pred_ratings = np.clip(self.U[u] @ self.V.T, 1, 5)
        already_rated = set(np.where(self.M_train[u] > 0)[0])
        candidates = self.movies.copy()
        candidates["predicted_rating"] = pred_ratings
        candidates["item_idx"] = candidates["item_id"] - 1
        candidates = candidates[~candidates["item_idx"].isin(already_rated)]

        if genre:
            if genre not in candidates.columns:
                valid = ", ".join(self.available_genres())
                raise ValueError(f"Unknown genre '{genre}'. Available genres: {valid}")
            candidates = candidates[candidates[genre] == 1]

        return candidates[["item_id", "title", "predicted_rating"]].sort_values(
            ["predicted_rating", "title"], ascending=[False, True]
        ).head(top_n).reset_index(drop=True)

    def available_genres(self) -> list[str]:
        assert self.movie_genres is not None
        return [col for col in self.movie_genres.columns if col not in {"item_id", "title"}]

    def nearest_genre_neighbors(self, title_query: str, n_neighbors: int = 5) -> tuple[pd.Series, pd.DataFrame]:
        assert self.movie_genres is not None
        genre_cols = self.available_genres()
        movies = self.movie_genres.copy()

        mask = movies["title"].str.contains(title_query, case=False, na=False)
        if not mask.any():
            raise ValueError(f"No movie title matched '{title_query}'")

        target = movies[mask].iloc[0]
        X = movies[genre_cols].to_numpy(dtype=int)
        nn = NearestNeighbors(metric="hamming")
        nn.fit(X)
        distances, indices = nn.kneighbors(
            [target[genre_cols].to_numpy(dtype=int)],
            n_neighbors=min(n_neighbors + 1, len(movies)),
        )

        rows = []
        for dist, idx in zip(distances[0], indices[0]):
            neighbor = movies.iloc[int(idx)]
            if int(neighbor["item_id"]) == int(target["item_id"]):
                continue
            rows.append({
                "item_id": int(neighbor["item_id"]),
                "title": neighbor["title"],
                "genre_distance": round(float(dist), 3),
            })
            if len(rows) >= n_neighbors:
                break

        return target[["item_id", "title"] + genre_cols], pd.DataFrame(rows)

    def search_movies(self, title_query: str, top_n: int = 10) -> pd.DataFrame:
        assert self.movies is not None
        matches = self.movies[self.movies["title"].str.contains(title_query, case=False, na=False)].copy()
        return matches[["item_id", "title"]].head(top_n).reset_index(drop=True)

    def resolve_movie(self, item_id: int | None = None, title_query: str | None = None) -> pd.Series:
        assert self.movies is not None
        if item_id is not None:
            matches = self.movies[self.movies["item_id"] == item_id]
            if matches.empty:
                raise ValueError(f"No movie with item_id={item_id}")
            return matches.iloc[0]
        if title_query:
            matches = self.movies[self.movies["title"].str.contains(title_query, case=False, na=False)]
            if matches.empty:
                raise ValueError(f"No movie title matched '{title_query}'")
            return matches.iloc[0]
        raise ValueError("Provide item_id or title_query")

    def fit_profile_vector(
        self,
        profile_ratings: dict[int, float],
        n_steps: int = 300,
        alpha: float | None = None,
        lambda_reg: float | None = None,
        seed: int = 42,
    ) -> np.ndarray:
        assert self.V is not None and self.k is not None
        if not profile_ratings:
            raise ValueError("Add at least one rating first.")

        lr = alpha if alpha is not None else (self.alpha if self.alpha is not None else 0.01)
        reg = lambda_reg if lambda_reg is not None else (self.lambda_reg if self.lambda_reg is not None else 0.02)
        rng = np.random.default_rng(seed)
        u_vec = rng.normal(scale=0.1, size=self.k)

        samples = []
        for item_id, rating in profile_ratings.items():
            item_idx = item_id - 1
            if item_idx < 0 or item_idx >= self.V.shape[0]:
                raise ValueError(f"Invalid item_id in profile: {item_id}")
            samples.append((item_idx, float(rating)))

        for _ in range(n_steps):
            rng.shuffle(samples)
            for item_idx, rating in samples:
                pred = float(np.dot(u_vec, self.V[item_idx]))
                err = rating - pred
                u_vec += lr * (err * self.V[item_idx] - reg * u_vec)

        return u_vec

    def get_profile_recommendations(
        self,
        profile_ratings: dict[int, float],
        top_n: int = 10,
        genre: str | None = None,
    ) -> pd.DataFrame:
        assert self.movies is not None and self.V is not None
        u_vec = self.fit_profile_vector(profile_ratings)
        pred_ratings = np.clip(u_vec @ self.V.T, 1, 5)

        candidates = self.movies.copy()
        candidates["predicted_rating"] = pred_ratings
        rated_item_ids = set(profile_ratings.keys())
        candidates = candidates[~candidates["item_id"].isin(rated_item_ids)]

        if genre:
            if genre not in candidates.columns:
                valid = ", ".join(self.available_genres())
                raise ValueError(f"Unknown genre '{genre}'. Available genres: {valid}")
            candidates = candidates[candidates[genre] == 1]

        return candidates[["item_id", "title", "predicted_rating"]].sort_values(
            ["predicted_rating", "title"], ascending=[False, True]
        ).head(top_n).reset_index(drop=True)


class RecommenderShell:
    def __init__(
        self,
        data_dir: str,
        k: int,
        alpha: float,
        lambda_reg: float,
        epochs: int,
        split: float,
        seed: int,
        verbose: bool,
    ):
        self.data_dir = data_dir
        self.k = k
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.epochs = epochs
        self.split = split
        self.seed = seed
        self.verbose = verbose
        self.model: MovieRecommender | None = None
        self.is_trained = False
        self.profile_ratings: dict[int, float] = {}

    def ensure_model_loaded(self) -> None:
        if self.model is None:
            self.model = MovieRecommender(data_dir=self.data_dir, verbose=self.verbose)

    def ensure_trained(self) -> None:
        if not self.is_trained:
            self.load_and_train()

    def load_and_train(self) -> None:
        self.ensure_model_loaded()
        assert self.model is not None
        self.model.build_matrix()
        self.model.random_train_test(seed=self.seed, split=self.split)
        self.model.init_factors(
            k=self.k,
            alpha=self.alpha,
            lambda_reg=self.lambda_reg,
            n_epochs=self.epochs,
            seed=self.seed,
        )
        _, rmses = self.model.train()
        self.is_trained = True
        print("Training complete.")
        print(f"Users: {self.model.n_users} | Movies: {self.model.n_movies} | k: {self.model.k}")
        if rmses:
            print(f"Final test RMSE: {rmses[-1]:.4f}")

    def print_help(self) -> None:
        print(
            "Commands:\n"
            "  help                                              Show this help\n"
            "  train                                             Train the matrix factorization model\n"
            "  recommend --user ID [--top-n N] [--genre NAME] [--show-history]\n"
            "  search --title TEXT [--top-n N]                   Find movies by title\n"
            "  rate (--item-id ID | --title TEXT) --rating R     Add or update your own rating (1-5)\n"
            "  unrate --item-id ID                               Remove one of your ratings\n"
            "  my-ratings                                        Show your current profile ratings\n"
            "  clear-ratings                                     Clear your profile ratings\n"
            "  recommend-me [--top-n N] [--genre NAME]           Recommend for your profile\n"
            "  genre-neighbors --title TEXT [--top-n N]\n"
            "  genres                                            List available genres\n"
            "  config                                            Show current shell settings\n"
            "  retrain                                           Retrain using current settings\n"
            "  exit / quit                                       Leave the shell\n"
        )

    def print_config(self) -> None:
        print("Current settings:")
        print(f"  data_dir          = {self.data_dir}")
        print(f"  k                 = {self.k}")
        print(f"  alpha             = {self.alpha}")
        print(f"  lambda_reg        = {self.lambda_reg}")
        print(f"  epochs            = {self.epochs}")
        print(f"  split             = {self.split}")
        print(f"  seed              = {self.seed}")
        print(f"  verbose           = {self.verbose}")
        print(f"  is_trained        = {self.is_trained}")
        print(f"  profile_ratings   = {len(self.profile_ratings)} movie(s)")

    def show_profile_ratings(self) -> None:
        self.ensure_model_loaded()
        assert self.model is not None
        if not self.profile_ratings:
            print("You have not rated any movies yet.")
            return
        rows = []
        for item_id, rating in self.profile_ratings.items():
            movie = self.model.resolve_movie(item_id=item_id)
            rows.append({"item_id": item_id, "title": movie["title"], "rating": rating})
        df = pd.DataFrame(rows).sort_values(["rating", "title"], ascending=[False, True]).reset_index(drop=True)
        print_table(df)

    def run(self) -> None:
        print("Movie recommender shell")
        print("Type 'help' for commands, 'exit' to quit.")
        while True:
            try:
                line = input("movie-cli> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not line:
                continue
            if line.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break
            if line.lower() == "help":
                self.print_help()
                continue
            if line.lower() == "config":
                self.print_config()
                continue
            if line.lower() in {"train", "retrain"}:
                try:
                    self.load_and_train()
                except Exception as exc:
                    print(f"Error: {exc}")
                continue
            if line.lower() == "genres":
                try:
                    self.ensure_model_loaded()
                    assert self.model is not None
                    print("Available genres:")
                    for genre in self.model.available_genres():
                        print(f"- {genre}")
                except Exception as exc:
                    print(f"Error: {exc}")
                continue
            if line.lower() == "my-ratings":
                try:
                    self.show_profile_ratings()
                except Exception as exc:
                    print(f"Error: {exc}")
                continue
            if line.lower() == "clear-ratings":
                self.profile_ratings.clear()
                print("Cleared your profile ratings.")
                continue
            if line.startswith("recommend ") or line == "recommend":
                try:
                    self.ensure_trained()
                    assert self.model is not None
                    cmd_parser = argparse.ArgumentParser(prog="recommend", add_help=False)
                    cmd_parser.add_argument("command")
                    cmd_parser.add_argument("--user", type=int, required=True)
                    cmd_parser.add_argument("--top-n", type=int, default=10)
                    cmd_parser.add_argument("--genre", type=str, default=None)
                    cmd_parser.add_argument("--show-history", action="store_true")
                    cmd_args = cmd_parser.parse_args(shlex.split(line))

                    if cmd_args.show_history:
                        print(f"\nUser {cmd_args.user} history")
                        print_table(self.model.get_user_history(cmd_args.user))
                        print()
                    print(f"Top {cmd_args.top_n} recommendations for user {cmd_args.user}")
                    print_table(
                        self.model.get_recommendations(
                            cmd_args.user,
                            top_n=cmd_args.top_n,
                            genre=cmd_args.genre,
                        )
                    )
                except SystemExit:
                    print("Usage: recommend --user ID [--top-n N] [--genre NAME] [--show-history]")
                except Exception as exc:
                    print(f"Error: {exc}")
                continue
            if line.startswith("recommend-me"):
                try:
                    self.ensure_trained()
                    assert self.model is not None
                    cmd_parser = argparse.ArgumentParser(prog="recommend-me", add_help=False)
                    cmd_parser.add_argument("command")
                    cmd_parser.add_argument("--top-n", type=int, default=10)
                    cmd_parser.add_argument("--genre", type=str, default=None)
                    cmd_args = cmd_parser.parse_args(shlex.split(line))
                    if not self.profile_ratings:
                        raise ValueError("Rate a few movies first with: rate --item-id ID --rating R")
                    print(f"Top {cmd_args.top_n} recommendations for your profile")
                    print_table(
                        self.model.get_profile_recommendations(
                            self.profile_ratings,
                            top_n=cmd_args.top_n,
                            genre=cmd_args.genre,
                        )
                    )
                except SystemExit:
                    print("Usage: recommend-me [--top-n N] [--genre NAME]")
                except Exception as exc:
                    print(f"Error: {exc}")
                continue
            if line.startswith("search"):
                try:
                    self.ensure_model_loaded()
                    assert self.model is not None
                    cmd_parser = argparse.ArgumentParser(prog="search", add_help=False)
                    cmd_parser.add_argument("command")
                    cmd_parser.add_argument("--title", required=True)
                    cmd_parser.add_argument("--top-n", type=int, default=10)
                    cmd_args = cmd_parser.parse_args(shlex.split(line))
                    print_table(self.model.search_movies(cmd_args.title, top_n=cmd_args.top_n))
                except SystemExit:
                    print("Usage: search --title TEXT [--top-n N]")
                except Exception as exc:
                    print(f"Error: {exc}")
                continue
            if line.startswith("rate"):
                try:
                    self.ensure_model_loaded()
                    assert self.model is not None
                    cmd_parser = argparse.ArgumentParser(prog="rate", add_help=False)
                    cmd_parser.add_argument("command")
                    cmd_parser.add_argument("--item-id", type=int, default=None)
                    cmd_parser.add_argument("--title", type=str, default=None)
                    cmd_parser.add_argument("--rating", type=float, required=True)
                    cmd_args = cmd_parser.parse_args(shlex.split(line))
                    if cmd_args.item_id is None and cmd_args.title is None:
                        raise ValueError("Provide --item-id or --title")
                    if not (1 <= cmd_args.rating <= 5):
                        raise ValueError("rating must be between 1 and 5")
                    movie = self.model.resolve_movie(item_id=cmd_args.item_id, title_query=cmd_args.title)
                    item_id = int(movie["item_id"])
                    self.profile_ratings[item_id] = float(cmd_args.rating)
                    print(f"Saved rating {cmd_args.rating:.1f} for: {movie['title']} (item_id={item_id})")
                except SystemExit:
                    print("Usage: rate (--item-id ID | --title TEXT) --rating R")
                except Exception as exc:
                    print(f"Error: {exc}")
                continue
            if line.startswith("unrate"):
                try:
                    cmd_parser = argparse.ArgumentParser(prog="unrate", add_help=False)
                    cmd_parser.add_argument("command")
                    cmd_parser.add_argument("--item-id", type=int, required=True)
                    cmd_args = cmd_parser.parse_args(shlex.split(line))
                    if cmd_args.item_id in self.profile_ratings:
                        del self.profile_ratings[cmd_args.item_id]
                        print(f"Removed rating for item_id={cmd_args.item_id}")
                    else:
                        print(f"No saved rating for item_id={cmd_args.item_id}")
                except SystemExit:
                    print("Usage: unrate --item-id ID")
                continue
            if line.startswith("genre-neighbors"):
                try:
                    self.ensure_model_loaded()
                    assert self.model is not None
                    cmd_parser = argparse.ArgumentParser(prog="genre-neighbors", add_help=False)
                    cmd_parser.add_argument("command")
                    cmd_parser.add_argument("--title", required=True)
                    cmd_parser.add_argument("--top-n", type=int, default=5)
                    cmd_args = cmd_parser.parse_args(shlex.split(line))

                    target, neighbors = self.model.nearest_genre_neighbors(
                        cmd_args.title, n_neighbors=cmd_args.top_n
                    )
                    genre_cols = [c for c in target.index if c not in {"item_id", "title"}]
                    active_genres = [g for g in genre_cols if int(target[g]) == 1]
                    print(f"Matched movie: {target['title']} (id={int(target['item_id'])})")
                    print(f"Genres: {', '.join(active_genres) if active_genres else 'None'}")
                    print()
                    print_table(neighbors)
                except SystemExit:
                    print("Usage: genre-neighbors --title TEXT [--top-n N]")
                except Exception as exc:
                    print(f"Error: {exc}")
                continue

            print("Unknown command. Type 'help' for a list of commands.")


def print_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("No results.")
    else:
        print(df.to_string(index=False))


def build_and_train(args: argparse.Namespace) -> MovieRecommender:
    model = MovieRecommender(data_dir=args.data_dir, verbose=args.verbose)
    model.build_matrix()
    model.random_train_test(seed=args.seed, split=args.split)
    model.init_factors(
        k=args.k,
        alpha=args.alpha,
        lambda_reg=args.lambda_reg,
        n_epochs=args.epochs,
        seed=args.seed,
    )
    _, rmses = model.train()
    if args.verbose:
        print(f"Final test RMSE: {rmses[-1]:.4f}")
    return model


def add_shared_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default="data", help="Directory containing u.data, u.item, and u.genre")
    parser.add_argument("--k", type=int, default=20, help="Latent factor dimension")
    parser.add_argument("--alpha", type=float, default=0.005, help="SGD learning rate")
    parser.add_argument("--lambda-reg", type=float, default=0.02, help="L2 regularization strength")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--split", type=float, default=0.8, help="Train split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print training progress")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Movie recommender CLI using matrix factorization and genre neighbors"
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    train_parser = subparsers.add_parser("train", help="Train the matrix factorization model and report RMSE")
    add_shared_training_args(train_parser)

    rec_parser = subparsers.add_parser("recommend", help="Recommend movies for a dataset user")
    add_shared_training_args(rec_parser)
    rec_parser.add_argument("--user", type=int, required=True, help="1-indexed user id")
    rec_parser.add_argument("--top-n", type=int, default=10, help="How many recommendations to show")
    rec_parser.add_argument("--genre", type=str, default=None, help="Optional genre filter, e.g. Comedy")
    rec_parser.add_argument("--show-history", action="store_true", help="Show the user's prior ratings first")

    genre_parser = subparsers.add_parser("genre-neighbors", help="Find movies with similar genre vectors")
    genre_parser.add_argument("--data-dir", default="data", help="Directory containing u.data, u.item, and u.genre")
    genre_parser.add_argument("--title", required=True, help="Movie title substring to search for")
    genre_parser.add_argument("--top-n", type=int, default=5, help="Number of neighbors to show")

    genres_parser = subparsers.add_parser("genres", help="List available genre names")
    genres_parser.add_argument("--data-dir", default="data", help="Directory containing u.data, u.item, and u.genre")

    shell_parser = subparsers.add_parser("shell", help="Start an interactive shell")
    add_shared_training_args(shell_parser)

    args = parser.parse_args()

    if args.command is None or args.command == "shell":
        data_dir = getattr(args, "data_dir", "data")
        k = getattr(args, "k", 20)
        alpha = getattr(args, "alpha", 0.005)
        lambda_reg = getattr(args, "lambda_reg", 0.02)
        epochs = getattr(args, "epochs", 30)
        split = getattr(args, "split", 0.8)
        seed = getattr(args, "seed", 42)
        verbose = getattr(args, "verbose", False)
        shell = RecommenderShell(
            data_dir=data_dir,
            k=k,
            alpha=alpha,
            lambda_reg=lambda_reg,
            epochs=epochs,
            split=split,
            seed=seed,
            verbose=verbose,
        )
        shell.run()
        return

    if args.command == "train":
        model = build_and_train(args)
        print("Training complete.")
        print(f"Users: {model.n_users} | Movies: {model.n_movies} | k: {model.k}")
        return

    if args.command == "recommend":
        model = build_and_train(args)
        if args.show_history:
            print(f"\nUser {args.user} history")
            print_table(model.get_user_history(args.user))
            print()
        print(f"Top {args.top_n} recommendations for user {args.user}")
        print_table(model.get_recommendations(args.user, top_n=args.top_n, genre=args.genre))
        return

    if args.command == "genre-neighbors":
        model = MovieRecommender(data_dir=args.data_dir)
        target, neighbors = model.nearest_genre_neighbors(args.title, n_neighbors=args.top_n)
        genre_cols = [c for c in target.index if c not in {"item_id", "title"}]
        active_genres = [g for g in genre_cols if int(target[g]) == 1]
        print(f"Matched movie: {target['title']} (id={int(target['item_id'])})")
        print(f"Genres: {', '.join(active_genres) if active_genres else 'None'}")
        print()
        print_table(neighbors)
        return

    if args.command == "genres":
        model = MovieRecommender(data_dir=args.data_dir)
        print("Available genres:")
        for genre in model.available_genres():
            print(f"- {genre}")
        return


if __name__ == "__main__":
    main()
