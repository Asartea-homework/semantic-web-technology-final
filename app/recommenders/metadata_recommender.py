from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np


class MetadataRecommender:
    metadata_fields = {
        "multi_value": ["tag", "subgenre", "motif"],
        "single_value": [
            "decennium_group",
            "language",
        ],
    }

    def __init__(
        self,
        stories: dict[str, dict],
        weights: dict[str, float] | None = None,
        geo_decay_km: float = 30.0,
    ):
        self.stories = stories
        self.story_ids = list(stories.keys())
        self.geo_decay_km = geo_decay_km

        self.weights = weights or {
            "tag": 1.0,
            "subgenre": 0.5,  # subgenres are less important than tags, but still relevant
            "motif": 1.0,
            "decennium_group": 0.5,  # stories from the same decade are more relevant than those from different decades
            "language": 3.0,  # priotize same-language stories (e.g. Frisian <> Frisian)
            "geographic": 0.5,  # story being closer content wise is more important than being geographically close, but still relevant
        }

        if self.geo_decay_km <= 0:
            raise ValueError("geo_decay_km must be a positive number.")

        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("All weights must be non-negative.")
        if not any(weight > 0 for weight in self.weights.values()):
            raise ValueError("At least one weight must be positive.")

        self.matrices = {}
        self.vectorizers = {}

        for field in (
            self.metadata_fields["multi_value"] + self.metadata_fields["single_value"]
        ):
            self.matrices[field], self.vectorizers[field] = self.create_field_matrix(
                field
            )

        coordinates = [self.get_coordinates(story) for story in self.stories.values()]
        self.coordinates = np.array(coordinates, dtype=float)

    @staticmethod
    def get_values(story: dict[str, str | list[str]], field: str, multi_value: bool):
        if multi_value:
            values = story.get(field) or []
            if not isinstance(values, list):
                values = [values]
            return [str(value) for value in values if value is not None and value != ""]
        else:
            value = story.get(field)
            if isinstance(value, list):
                value = value[0] if value else None
            return [str(value)] if value is not None and value != "" else []

    @staticmethod
    def get_coordinates(story: dict[str, str | list[str]]):
        lat = story.get("latitude")
        lon = story.get("longitude")

        if isinstance(lat, list):
            lat = lat[0] if lat else None
        if isinstance(lon, list):
            lon = lon[0] if lon else None

        try:
            lat = float(lat) if lat is not None else None
            lon = float(lon) if lon is not None else None
        except (ValueError, TypeError):
            lat = None
            lon = None

        return lat, lon

    def create_field_matrix(self, field):
        multi_value = field in self.metadata_fields["multi_value"]
        feature_dicts = []

        for story in self.stories.values():
            values = self.get_values(story, field, multi_value)
            features = {value: 1 for value in values}
            feature_dicts.append(features)

        vectorizer = DictVectorizer(sparse=True)
        field_matrix = vectorizer.fit_transform(feature_dicts)

        field_matrix = normalize(field_matrix, norm="l2", axis=1)

        return field_matrix, vectorizer

    def haversine_distances(self, lat, lon):
        """Distances in kilometres from one point to all stories."""
        coords = self.coordinates
        valid = np.isfinite(coords).all(axis=1)

        distances = np.full(len(coords), np.nan)

        if not valid.any():
            return distances

        R = 6371.0

        lat1 = np.radians(lat)
        lon1 = np.radians(lon)

        lat2 = np.radians(coords[valid, 0])
        lon2 = np.radians(coords[valid, 1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

        distances[valid] = 2 * R * np.arcsin(np.sqrt(a))

        return distances

    def geographic_similarities(self, story_id):
        """Convert distances into similarities between zero and one."""
        lat, lon = self.coordinates[story_id]

        if not np.isfinite([lat, lon]).all():
            return np.zeros(len(self.story_ids))

        distances = self.haversine_distances(lat, lon)
        scores = np.zeros(len(self.story_ids))

        valid = np.isfinite(distances)
        scores[valid] = np.exp(-distances[valid] / self.geo_decay_km)

        return scores

    def recommend(self, story_id: str, k: int = 10) -> list[dict]:
        if story_id not in self.stories:
            raise ValueError(f"Story ID {story_id} not found in the dataset.")

        if k <= 0:
            return []

        story_index = self.story_ids.index(story_id)
        story = self.stories[story_id]

        active_fields = set()
        for field in (
            self.metadata_fields["multi_value"] + self.metadata_fields["single_value"]
        ):
            matrix = self.matrices[field]
            weight = self.weights.get(field, 0)

            if (
                weight > 0
                and matrix is not None
                and self.get_values(
                    story, field, field in self.metadata_fields["multi_value"]
                )
            ):
                active_fields.add(field)

        has_geo = (
            self.weights.get("geographic", 0) > 0
            and np.isfinite(self.coordinates[story_index]).all()
        )

        if not active_fields and not has_geo:
            return []

        total_scores = np.zeros(len(self.stories))
        total_weight = sum(self.weights[field] for field in active_fields)

        for field in active_fields:
            matrix = self.matrices[field]
            weight = self.weights[field]

            field_scores = cosine_similarity(matrix[story_index], matrix).flatten()

            total_scores += weight * field_scores

        if has_geo:
            geo_scores = self.geographic_similarities(story_index)
            total_scores += self.weights["geographic"] * geo_scores
            total_weight += self.weights["geographic"]

        similarities = total_scores / total_weight

        similarities[story_index] = -1

        top_indices = np.argsort(similarities)[::-1][:k]

        return [
            {
                "story_id": self.story_ids[index],
                "similarity": similarities[index],
                **self.stories[self.story_ids[index]],
            }
            for index in top_indices
        ]
