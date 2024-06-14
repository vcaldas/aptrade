"""Simple raw data for the cat example, as well as the conversion to a dictionary
of pydantic models of Cat
"""
from .models import Asset
import csv
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
_stocks = []

with open(os.path.join(dir_path, 'symbols.csv')) as f:
    for row in csv.reader(f):
        _stocks.append({
            'symbol': row[0],
            'name': row[1]
        })

DATA_STOCK = {i: Asset(asset_id=i, **x) for i, x in enumerate(_stocks)}
