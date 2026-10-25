from flask import render_template


def handle_404(_):
    return render_template("404.html"), 404
