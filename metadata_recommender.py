from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np


class MetadataRecommender:
    metadata_fields = {
        "multi_value": ["tag", "subgenre", "motif"],
        "single_value": [
            "decennium_group",
            "language",
        ],
    }

    def __init__(self, stories: dict[str, dict]):
        self.stories = stories
        # self.metadata_matrix, self.vectorizer = self.create_metadata_matrix()
        self.metadata_matrix, self.vectorizer = self.tf_idf_matrix()

    def create_metadata_matrix(self):
        metadata_dicts = []

        for story in self.stories.values():
            features = {}

            # Multi-value fields: create one binary feature per value
            for field in self.metadata_fields["multi_value"]:
                values = story.get(field) or []

                for value in values:
                    if value is not None and value != "":
                        features[f"{field}={value}"] = 1

            for field in self.metadata_fields["single_value"]:
                value = story.get(field)

                if isinstance(value, list):
                    value = value[0] if value else None

                if value is not None and value != "":
                    features[field] = value

            metadata_dicts.append(features)

        vectorizer = DictVectorizer(sparse=True)
        metadata_matrix = vectorizer.fit_transform(metadata_dicts)

        metadata_matrix = normalize(metadata_matrix, norm="l2", axis=1)

        return metadata_matrix, vectorizer

    def old_recommend(self, story_id: str, k: int = 10) -> list[dict]:
        if story_id not in self.stories:
            raise ValueError(f"Story ID {story_id} not found in the dataset.")

        if k <= 0:
            return []

        story_ids = list(self.stories.keys())
        story_index = story_ids.index(story_id)
        story = self.stories[story_id]

        # Determine which metadata fields the query story actually has.
        active_fields = set()

        for field in self.metadata_fields["multi_value"]:
            values = story.get(field) or []
            if not isinstance(values, list):
                values = [values]

            if any(value is not None and value != "" for value in values):
                active_fields.add(field)

        for field in self.metadata_fields["single_value"]:
            value = story.get(field)
            if isinstance(value, list):
                value = value[0] if value else None

            if value is not None and value != "":
                active_fields.add(field)

        if not active_fields:
            return []

        feature_names = self.vectorizer.get_feature_names_out()

        feature_mask = np.array(
            [
                any(name.startswith(f"{field}=") for field in active_fields)
                for name in feature_names
            ]
        )

        query_vector = self.metadata_matrix[story_index].multiply(feature_mask)

        query_vector = normalize(query_vector, norm="l2")
        similarities = cosine_similarity(query_vector, self.metadata_matrix).flatten()

        # Exclude the query story itself.
        similarities[story_index] = -1

        k = min(k, len(similarities) - 1)
        if k <= 0:
            return []

        top_indices = similarities.argsort()[::-1][:k]

        return [
            {
                "id": story_ids[index],
                "similarity": float(similarities[index]),
                **self.stories[story_ids[index]],
            }
            for index in top_indices
        ]

    def recommend(self, story_id: str, k: int = 10) -> list[dict]:
        if story_id not in self.stories:
            raise ValueError(f"Story ID {story_id} not found in the dataset.")

        if k <= 0:
            return []

        story_ids = list(self.stories.keys())
        story_index = story_ids.index(story_id)

        query_vector = self.metadata_matrix[story_index]

        similarities = cosine_similarity(query_vector, self.metadata_matrix).flatten()

        # Exclude the query story itself.
        similarities[story_index] = -1

        k = min(k, len(similarities) - 1)
        if k <= 0:
            return []

        top_indices = similarities.argsort()[::-1][:k]

        return [
            {
                "id": story_ids[index],
                "similarity": float(similarities[index]),
                **self.stories[story_ids[index]],
            }
            for index in top_indices
        ]

    def tf_idf_matrix(self):
        # Collect all metadata as text documents
        documents = []

        for story in self.stories.values():
            # Combine all metadata into a single text
            parts = []

            # Multi-value fields
            for field in self.metadata_fields["multi_value"]:
                values = story.get(field) or []
                if isinstance(values, list):
                    parts.extend(values)
                elif values:
                    parts.append(str(values))

            # Single-value fields
            for field in self.metadata_fields["single_value"]:
                value = story.get(field)
                if isinstance(value, list):
                    value = value[0] if value else None
                if value:
                    parts.append(str(value))

            documents.append(" ".join(parts))

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer()

        # Fit and transform
        metadata_matrix = vectorizer.fit_transform(documents)

        # Normalize for cosine similarity
        metadata_matrix = normalize(metadata_matrix, norm="l2", axis=1)

        return metadata_matrix, vectorizer
