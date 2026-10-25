from flask import Flask
from app.data import load_stories
from app.recommenders.hybrid_recommender import HybridRecommender
from app.routes import routes
from app.errors import handle_404
import os
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
embeddings_path = os.path.join(project_root, "app/embeddings")
app = Flask(__name__)
print("Loading stories...")
stories = load_stories(Path("app/stories.jsonl"))
print(f"Loaded {len(stories)} stories.")
print("Initializing recommender...")
recommender = HybridRecommender(stories, embeddings_path=embeddings_path)
print("Recommender initialized.")
app.config["stories"] = stories
app.config["story_ids"] = list(stories.keys())
app.config["recommender"] = recommender
app.register_blueprint(routes)
app.register_error_handler(404, handle_404)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
