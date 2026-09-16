import re
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from rapidfuzz import fuzz, process


class MusicRecommender:
    def __init__(self, data_path: str = "data/songs.csv"):
        self.data_path = data_path
        self.feature_cols = [
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "valence",
            "tempo"
        ]
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.normalized_features = None
        self.track_names = []

    def load_and_prepare(self, limit: int = 50000):
        # 1. Fast loading with Polars
        raw_df = pl.read_csv(self.data_path)
        
        # 2. Clean up: empty values drop aur unique tracks filter
        self.df = (
            raw_df
            .filter(pl.col("track_name").is_not_null() & pl.col("artist_name").is_not_null())
            .unique(subset=["track_name", "artist_name"])
            .sort("popularity", descending=True)
            .head(limit)
        )

        # 3. Numeric audio features scale karna
        features = self.df.select(self.feature_cols).to_numpy()
        self.normalized_features = self.scaler.fit_transform(features)

        # 4. Nearest Neighbors model train karna
        self.model = NearestNeighbors(n_neighbors=11, metric="cosine", algorithm="brute")
        self.model.fit(self.normalized_features)

        # Track names cache for fast fuzzy lookup
        self.track_names = self.df.get_column("track_name").to_list()
        print(f"Recommender ready with {len(self.df)} tracks indexed!")

    def find_song_index(self, track_name: str, threshold: float = 65.0):
        if not track_name or not track_name.strip():
            return None

        # 1. Base query cleanup
        # Extra spaces hatao (left, right, aur beech ke 2-3 spaces -> 1 space)
        cleaned_query = re.sub(r"\s+", " ", track_name.strip()).lower()
        # Bilkul bina kisi space wala version (e.g. "star    boy" -> "starboy")
        collapsed_query = cleaned_query.replace(" ", "")

        # -------------------------------------------------------------
        # Tier 1: Direct Exact Match (Single pass, instant lookup)
        # -------------------------------------------------------------
        for idx, name in enumerate(self.track_names):
            name_clean = name.strip().lower()
            name_collapsed = name_clean.replace(" ", "")
            
            # Agar exact string match ho ya bina space ke exact match ho
            if name_clean == cleaned_query or name_collapsed == collapsed_query:
                return idx

        # -------------------------------------------------------------
        # Tier 2: Prefix Match (Features / Remastered versions ke liye)
        # e.g. "starboy" -> "Starboy (feat. Daft Punk)"
        # -------------------------------------------------------------
        for idx, name in enumerate(self.track_names):
            name_collapsed = name.strip().lower().replace(" ", "")
            if name_collapsed.startswith(collapsed_query):
                return idx

        # -------------------------------------------------------------
        # Tier 3: RapidFuzz Fallback (Spelling mistakes ke liye)
        # -------------------------------------------------------------
        match = process.extractOne(
            cleaned_query,
            self.track_names,
            scorer=fuzz.WRatio
        )

        if match:
            best_match, score, match_idx = match
            if score >= threshold:
                print(f"[FUZZY MATCH] '{track_name}' -> '{best_match}' (Score: {score:.1f})")
                return match_idx

        return None

    def recommend(self, track_name: str, top_k: int = 5):
        if self.df is None or self.model is None:
            raise ValueError("Engine load nahi hua hai. Pehle load_and_prepare() run karo.")

        # 1. Fuzzy search se direct row index lena
        song_idx = self.find_song_index(track_name)
        if song_idx is None:
            return None

        # 2. Target track fetch karna
        target_track = self.df.row(song_idx, named=True)

        # 3. Nearest neighbors nikalna
        distances, neighbor_indices = self.model.kneighbors(
            [self.normalized_features[song_idx]],
            n_neighbors=top_k + 1
        )

        # 4. Recommendations prepare karna
        recommended_records = []
        for idx, dist in zip(neighbor_indices[0][1:], distances[0][1:]):
            rec = self.df.row(idx, named=True)
            recommended_records.append({
                "track_name": rec["track_name"],
                "artist_name": rec["artist_name"],
                "genre": rec.get("genre", "Unknown"),
                "similarity_score": round(float(1.0 - dist), 4),
                "energy": rec["energy"],
                "valence": rec["valence"]
            })

        return {
            "query_song": {
                "track_name": target_track["track_name"],
                "artist_name": target_track["artist_name"],
                "genre": target_track.get("genre", "Unknown"),
                "similarity_score": None,
                "energy": target_track["energy"],
                "valence": target_track["valence"]
            },
            "recommendations": recommended_records
        }


recommender_engine = MusicRecommender()