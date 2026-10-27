import csv
import random

from app import recommender

stories = recommender.stories
semantic = recommender.semantic_recommender


def get_text(story):
    # main_text is a list with one string
    return story["main_text"][0].replace("\n", " ")


def short_id(story_id):
    # remove "Item_" so only the number is left
    return story_id.replace("Item_", "")


# only stories that have an embedding
story_ids = []
for story_id in stories:
    if story_id in semantic.id_to_idx:
        story_ids.append(story_id)
story_ids.sort()

# fixed seed, so the sample is the same every time
sample_ids = random.Random(42).sample(story_ids, 50)

systems = {
    "metadata": recommender.metadata_recommender,
    "semantic": semantic,
    "hybrid": recommender,
}

# get the recommendations once, so every file uses the same lists
all_recs = {}
for name in systems:
    all_recs[name] = {}
    for story_id in sample_ids:
        all_recs[name][story_id] = systems[name].recommend(story_id, k=5)

# one readable text file per system
for name in systems:
    with open("judging_" + name + ".txt", "w", encoding="utf-8") as f:
        for story_id in sample_ids:
            f.write("=" * 60 + "\n")
            f.write("STORY " + short_id(story_id) + "\n")
            f.write(get_text(stories[story_id]) + "\n\n")

            rank = 1
            for rec in all_recs[name][story_id]:
                f.write(str(rank) + ". " + short_id(rec["id"]) + "\n")
                f.write("   " + get_text(rec) + "\n")
                rank = rank + 1


# one csv with the recommended ids of each method
with open("recommended_ids.csv", "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["story_id", "metadata", "semantic", "hybrid"])

    for story_id in sample_ids:
        row = [short_id(story_id)]

        for name in systems:
            ids = []
            for rec in all_recs[name][story_id]:
                ids.append(short_id(rec["id"]))
            # all 5 ids in one cell, in rank order
            row.append(" ".join(ids))

        writer.writerow(row)