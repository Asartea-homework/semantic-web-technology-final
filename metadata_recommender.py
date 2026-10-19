from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


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
        self.metadata_matrix, self.vectorizer = self.create_metadata_matrix()

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

    def recommend(self, story_id: str, k: int = 10) -> list[dict]:
        if story_id not in self.stories:
            raise ValueError(f"Story ID {story_id} not found in the dataset.")

        story_ids = list(self.stories.keys())
        story_index = story_ids.index(story_id)

        story_vector = self.metadata_matrix[story_index]
        similarities = cosine_similarity(story_vector, self.metadata_matrix).flatten()

        similarities[story_index] = -1  # Exclude the story itself from recommendations

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
