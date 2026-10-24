"""Semantic story recommender based on precomputed text embeddings and FAISS.

    1. In Google Colab (notebooks/), every story's `main_text` is turned into a
       vector with the BAAI/bge-m3 model.
       Stories with a similar meaning end up with vectors that point in a
       similar direction.
    2. The vectors are saved to `embeddings/embeddings.npy`, and the matching
       story IDs to `embeddings/story_ids.json`.
    3. This script loads those files, puts the vectors in a FAISS index, and
       recommends the stories whose vectors are closest to the query story.

cosine similarity: 1.0 means the same direction (very similar),
values near 0 mean unrelated.
"""

import json
from pathlib import Path

import faiss
import numpy as np


"""
Semantic story recommender based on precomputed text embeddings and FAISS.

How it works:
1. In Google Colab, each story's `main_text` is converted into a numerical
   vector (embedding) using the BAAI/bge-m3 language model.
2. Stories with similar meanings should have similar embedding vectors.
3. The vectors are saved to `embeddings/embeddings.npy`, while their
   corresponding story IDs are saved to `embeddings/story_ids.json`.
4. This script loads those files and creates a FAISS index to search for
   stories whose embeddings are most similar to a given story.

Cosine similarity:
- 1.0 means the vectors point in exactly the same direction.
- A value near 0 means the vectors are approximately unrelated in direction.
- A negative value means the vectors point in opposing directions.

Note: FAISS uses inner product in this implementation. Inner product is
equivalent to cosine similarity only when the embedding vectors are
normalized to have a length of 1.
"""

import json  # Provides functions for reading JSON files, such as story_ids.json.
from pathlib import Path  # Makes it easier to work with file and directory paths.

import faiss  # Provides efficient similarity search over numerical vectors.
import numpy as np  # Provides arrays and numerical operations for the embeddings.


class SemanticRecommender:
    """Recommends stories that are semantically similar to a given story."""

    def __init__(self, stories: dict[str, dict], embeddings_dir: str = "embeddings/"):
        """Loads the precomputed embeddings and builds the search index."""

        # Store the complete story dataset on the object.
        self.stories = stories

        # Convert the directory string into a Path object.
        embeddings_dir = Path(embeddings_dir)

        # Load the embedding vectors that are build with the notebook.
        # The embeddings were saved as float16 to reduce file size.
        # Convert them to float32 because FAISS expects float32 vectors.
        self.embeddings = np.load(
            embeddings_dir / "embeddings.npy"
        ).astype("float32")

        # Open the JSON file containing the story IDs in the same order as the rows in embeddings.npy.
        with open(embeddings_dir / "story_ids.json", "r") as f:
            # Convert the JSON array into a Python list of story IDs.
            self.story_ids = json.load(f)

        # Check that the loaded embeddings and story dataset are consistent.
        # This catches problems before the FAISS index is created.
        self.validate()

        # Build a dictionary that maps each story ID to its row number in the embeddings array.
        self.id_to_idx = {
            story_id: i
            for i, story_id in enumerate(self.story_ids)
        }

        # build the FAISS search index using all the embedding vectors.
        self.index = self.create_index()

    def validate(self):
        """Checks that the embeddings and the stories belong together."""

        # The number of story ids must match the number of embedding vectors.
        if len(self.story_ids) != self.embeddings.shape[0]:
            raise ValueError(
                "Number of story IDs does not match number of embeddings."
            )

        missing = [
            sid
            for sid in self.story_ids
            if sid not in self.stories
        ]

        # If any embedded stories are missing, stop and explain the problem.
        if missing:
            raise ValueError(
                f"{len(missing)} embedded stories are not in the dataset, "
                "re-run the notebook with the current stories.jsonl."
            )

    def create_index(self):
        """Builds a FAISS index over all story embeddings."""

        # Create a FAISS index
        index = faiss.IndexFlatIP(self.embeddings.shape[1])

        # Add every story embedding to the FAISS index.
        # FAISS stores these vectors so that they can be searched efficiently.
        index.add(self.embeddings)

        return index

    def scores_for(self, story_id: str) -> np.ndarray:
        """Returns the semantic similarity between one story and every story."""

        # Check whether the requested story has a corresponding embedding.
        if story_id not in self.id_to_idx:
            raise ValueError(
                f"Story ID {story_id} not found in the embeddings."
            )

        # Find the row number of the requested story's embedding.
        story_index = self.id_to_idx[story_id]

        # Get the total number of embedded stories
        n = len(self.story_ids)

        # Search the FAISS index using the selected story's embedding as the query.
        # n tells FAISS to return the n nearest vectors, meaning all stories.
        sims, idxs = self.index.search(
            self.embeddings[story_index : story_index + 1],
            n
        )

        # Create an array of zeros with one position for every story
        # use this array to put the scores back into the original
        # story order instead of leaving them sorted by similarity
        scores = np.zeros(n, dtype="float32")

        # Place each similarity score at the position of its corresponding
        # embedding row.
        scores[idxs[0]] = sims[0]

        # Return one score per story, in the original embedding order.
        return scores

    def recommend(self, story_id: str, k: int = 10) -> list[dict]:
        """Recommends the k stories most similar in meaning to a given story."""

        # Verify that the requested story has an embedding
        if story_id not in self.id_to_idx:
            raise ValueError(
                f"Story ID {story_id} not found in the embeddings."
            )

        # Find the row number of the selected story's embedding
        story_index = self.id_to_idx[story_id]

        # Limit k to the number of other available stories
        k = min(k, len(self.story_ids) - 1)

        # If k is zero or negative, there is nothing to recommend
        # Return an empty list rather than performing a search
        if k <= 0:
            return []

        # Search for k + 1 because the story itself is always its own best
        sims, idxs = self.index.search(
            self.embeddings[story_index : story_index + 1],
            k + 1
        )

        # Build a list of recommended stories
        return [
            {
                # The ID of the recommended story
                "id": self.story_ids[index],

                # The similarity score
                "similarity": float(sim),

                # Include all the other fields belonging to this story.
                **self.stories[self.story_ids[index]],
            }
            # Process each similarity score together with its matching row index.
            for sim, index in zip(sims[0], idxs[0])

            # Exclude the query story itself from the recommendations.
            if index != story_index
        ][:k] 