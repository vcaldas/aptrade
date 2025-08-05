import time

from flask import jsonify

from src.api import bp


@bp.route("/time", methods=["GET"])
def get_current_time():
    return jsonify({"time": time.time()})


@bp.route("/times", methods=["GET"])
def get_current_times():
    return jsonify({"time": time.time()})
