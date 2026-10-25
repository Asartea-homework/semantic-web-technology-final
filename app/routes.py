from flask.blueprints import Blueprint
from flask import render_template
from flask import current_app, redirect
import random

routes = Blueprint("routes", __name__)


@routes.route("/")
def index():
    return render_template("index.html")


@routes.route("/random")
def random_story():
    # get stories from app context
    stories = current_app.config["story_ids"]
    random_story_id = random.choice(stories)
    return redirect(f"/story/{random_story_id}")


@routes.route("/story/<story_id>")
def story(story_id):
    stories = current_app.config["stories"]
    recommender = current_app.config["recommender"]

    if story_id not in stories:
        return render_template("404.html"), 404

    story = stories[story_id]
    recommendations = recommender.recommend(story_id, k=5)

    story["title"] = (
        story.get("title", [[""]])[0]
        if isinstance(story.get("title"), list)
        else story.get("title", "")
    )
    story["text"] = (
        story.get("main_text", [[""]])[0]
        if isinstance(story.get("main_text"), list)
        else story.get("main_text", "")
    )
    story["tags"] = story.get("tag", []) if isinstance(story.get("tag"), list) else []

    # Normalize recommendations data for template
    for rec in recommendations:
        rec["title"] = (
            rec.get("title", [[""]])[0]
            if isinstance(rec.get("title"), list)
            else rec.get("title", "")
        )
        rec["text"] = (
            rec.get("main_text", [[""]])[0]
            if isinstance(rec.get("main_text"), list)
            else rec.get("main_text", "")
        )
        rec["tags"] = rec.get("tag", []) if isinstance(rec.get("tag"), list) else []

    return render_template("story.html", story=story, recommendations=recommendations)
