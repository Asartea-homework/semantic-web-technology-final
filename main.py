from data import load_stories
from metadata_recommender import MetadataRecommender
from semantic_recommender import SemanticRecommender

import numpy as np
import random


class HybridRecommender:

    def __init__(self, stories):
        self.metadata_recommender = MetadataRecommender(stories)
        self.semantic_recommender = SemanticRecommender(stories)
        self.stories = stories

    def recommend(self, story_id: str, k: int = 10):
        metadata_recommendations = self.metadata_recommender.recommend(story_id, k * 2)
        semantic_recommendations = self.semantic_recommender.recommend(story_id, k * 2)

        weights = self.get_weights(story_id)

        combined_recommendations = {}

        for rec in metadata_recommendations:
            combined_recommendations[rec["id"]] = {
                "id": rec["id"],
                "similarity": rec["similarity"] * weights["metadata"],
                "title": rec.get("title", ""),
                "main_text": rec.get("main_text", ""),
            }

        for rec in semantic_recommendations:
            if rec["id"] in combined_recommendations:
                combined_recommendations[rec["id"]]["similarity"] += (
                    rec["similarity"] * weights["semantic"]
                )
            else:
                combined_recommendations[rec["id"]] = {
                    "id": rec["id"],
                    "similarity": rec["similarity"] * weights["semantic"],
                    "title": rec.get("title", ""),
                    "main_text": rec.get("main_text", ""),
                }

        # Sort the combined recommendations by similarity score in descending order
        sorted_recommendations = sorted(
            combined_recommendations.values(),
            key=lambda x: x["similarity"],
            reverse=True,
        )

        # Return the top k recommendations
        return sorted_recommendations[:k]

    def get_weights(self, story_id):
        story = self.stories[story_id]
        tag_count = len(story.get("tag", []))
        text_length = len(story.get("main_text")[0].split())

        metadata_weight = 0.3 + 0.3 * (1 - np.exp(-tag_count / 5))

        if text_length < 20:
            metadata_weight = metadata_weight + 0.2
        elif text_length > 100:
            metadata_weight = metadata_weight - 0.2

        return {"metadata": metadata_weight, "semantic": 1 - metadata_weight}


def print_recommendations(
    story_id: str, stories, recommender: MetadataRecommender, k: int = 10
):
    recommendations = recommender.recommend(story_id, k=k)
    print(
        f"Recommendations for story ID {story_id} with text {stories[story_id]['main_text']} and tags {stories[story_id]['tag']}:\n"
    )
    for rec in recommendations:
        print(
            f"ID: {rec['id']}, Similarity: {(rec['similarity'] * 100):.2f}, Title: {rec.get('title', '')}\n"
        )
        print(f"Text: {rec.get('main_text', '')}\n")


def main():
    stories = load_stories("stories.jsonl")

    # select the kind of recommender
    recommender = HybridRecommender(stories)
    ids = random.sample(list(stories.keys()), 10)
    for story_id in ids:
        print_recommendations(story_id, stories, recommender, k=5)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
