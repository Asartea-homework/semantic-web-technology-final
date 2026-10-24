from data import load_stories
from metadata_recommender import MetadataRecommender
from semantic_recommender import SemanticRecommender

import random


def print_recommendations(
    story_id: str, stories, recommender: MetadataRecommender, k: int = 10
):
    recommendations = recommender.recommend(story_id, k=k)
    print(
        f"Recommendations for story ID {story_id} with text {stories[story_id]['main_text']}:\n"
    )
    for rec in recommendations:
        print(
            f"ID: {rec['id']}, Similarity: {rec['similarity']:.4f}, Title: {rec.get('title', '')}\n"
        )
        print(f"Text: {rec.get('main_text', '')}\n")
    print("\n\n")


def main():
    stories = load_stories("stories.jsonl")

    # select the kind of recommender
    recommender = MetadataRecommender(stories)
    ids = random.sample(list(stories.keys()), 5)
    for story_id in ids:
        print_recommendations(story_id, stories, recommender, k=5)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
