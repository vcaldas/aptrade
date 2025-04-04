import time

from flask import jsonify

from app.api import bp


@bp.route("/time", methods=["GET"])
def get_current_time():
    return jsonify({"time": time.time()})
