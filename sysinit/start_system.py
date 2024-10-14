# This script initialized the system and updates the provided data.
# The system must be up and running -  docker-compose up -d
from syscore.genutils import new_removing_existing
from syscore.interactive.input import true_if_answer_is_yes
from sysdata.mongodb.mongo_spread_costs import mongoSpreadCostData
from sysdata.csv.csv_spread_costs import csvSpreadCostData
from sysdata.futures.spread_costs import spreadCostData
from sysdata.data_blob import dataBlob
import os
from sysproduction.data.prices import get_valid_instrument_code_from_user, diagPrices
import pandas as pd

# https://github.com/vcaldas/aptrade/blob/master/docs/data.md#instrument-configuration-and-spread-costs
from sysinit.futures.repocsv_spread_costs import copy_spread_costs_from_csv_to_mongo

myData = dataBlob()
instrument_code = "GOLD"
diag_prices = diagPrices()

parquet_futures_contract_price_data = diag_prices.db_futures_contract_price_data

copy_spread_costs_from_csv_to_mongo(myData)

# # Write the multiple contracts from CSV to the database - These are usually behind
# from sysinit.futures.multiple_and_adjusted_from_csv_to_arctic import init_arctic_with_csv_futures_contract_prices, init_arctic_with_csv_prices_for_code
# # init_arctic_with_csv_futures_contract_prices()
# init_arctic_with_csv_prices_for_code(instrument_code)

# # Then get the roll calendars from the multiple prices
# from sysinit.futures.rollcalendars_from_providedcsv_prices import generate_roll_calendars_from_provided_multiple_csv_prices
# generate_roll_calendars_from_provided_multiple_csv_prices(instrument_code=instrument_code)

# ## Now get the new contracts
# # Get data from IB
# from sysinit.futures.seed_price_data_from_IB import seed_price_data_from_IB
# seed_price_data_from_IB(instrument_code=instrument_code)

roll_calendars_from_db = os.path.join(
    os.sep, "home", "vcaldas", "data", "interim", "roll_calendars_from_contracts"
)
source_contracts = os.path.join(
    os.sep, "home", "vcaldas", "data", "parquet", "future_contract_prices"
)
if not os.path.exists(roll_calendars_from_db):
    os.makedirs(roll_calendars_from_db)

from sysinit.futures.rollcalendars_from_arcticprices_to_csv import (
    build_and_write_roll_calendar,
)

build_and_write_roll_calendar(
    instrument_code, output_datapath=roll_calendars_from_db, check_before_writing=False
)

## Generate Interim Prices
from sysinit.futures.multipleprices_from_db_prices_and_csv_calendars_to_db import (
    process_multiple_prices_single_instrument,
)
from syscore.constants import arg_not_supplied

csv_multiple_data_path = os.path.join(
    os.sep, "home", "vcaldas", "data", "interim", "multiple_prices_csv"
)
if not os.path.exists(csv_multiple_data_path):
    os.makedirs(csv_multiple_data_path)

multiple_prices_from_db = os.path.join(
    os.sep, "home", "vcaldas", "data", "interim", "multiple_from_db"
)
if not os.path.exists(multiple_prices_from_db):
    os.makedirs(multiple_prices_from_db)

# only change if you have written the files elsewhere
csv_roll_data_path = arg_not_supplied
process_multiple_prices_single_instrument(
    instrument_code=instrument_code,
    csv_multiple_data_path=csv_multiple_data_path,
    csv_roll_data_path=roll_calendars_from_db,
    ADD_TO_DB=False,
    ADD_TO_CSV=True,
)


## Get existent multiple prices
supplied_file = os.path.join(
    os.sep,
    "home",
    "vcaldas",
    "aptrade",
    "data",
    "futures",
    "multiple_prices_csv",
    instrument_code + ".csv",
)
generated_file = os.path.join(csv_multiple_data_path, instrument_code + ".csv")

supplied = pd.read_csv(supplied_file, index_col=0, parse_dates=True)
generated = pd.read_csv(generated_file, index_col=0, parse_dates=True)


last_supplied = supplied.index[-1]
print(last_supplied)
generated = generated.loc[last_supplied:]

first_generated = generated.index[0]
if first_generated == last_supplied:
    print("Skipping first row")
    generated = generated.iloc[1:]


try:
    assert (
        supplied.iloc[-1].PRICE_CONTRACT
        == generated.loc[last_supplied:].iloc[0].PRICE_CONTRACT
    )
    assert (
        supplied.iloc[-1].FORWARD_CONTRACT
        == generated.loc[last_supplied:].iloc[0].FORWARD_CONTRACT
    )
except AssertionError as e:
    print(supplied.tail())
    print(generated.head())
    raise e
print("Prices are consistent")


spliced = pd.concat([supplied, generated])
spliced_multiple_prices = os.path.join(
    os.sep, "home", "vcaldas", "data", "interim", "multiple_prices_csv_spliced"
)
if not os.path.exists(spliced_multiple_prices):
    os.makedirs(spliced_multiple_prices)

spliced.to_csv(os.path.join(spliced_multiple_prices, instrument_code + ".csv"))

db_multiple_prices = diag_prices.db_futures_multiple_prices_data
from sysdata.csv.csv_multiple_prices import csvFuturesMultiplePricesData

csv_multiple_prices = csvFuturesMultiplePricesData(spliced_multiple_prices)

print(csv_multiple_prices.datapath)
multiple_prices = csv_multiple_prices.get_multiple_prices(instrument_code)

db_multiple_prices.add_multiple_prices(
    instrument_code, multiple_prices, ignore_duplication=True
)

# Finally create the adjusted prices
from sysinit.futures.adjustedprices_from_db_multiple_to_db import (
    process_adjusted_prices_single_instrument,
)

csv_adj_data_path = os.path.join(
    os.sep, "home", "vcaldas", "data", "futures", "futures_adjusted_prices"
)
if not os.path.exists(csv_adj_data_path):
    os.makedirs(csv_adj_data_path)
process_adjusted_prices_single_instrument(
    instrument_code,
    csv_adj_data_path=csv_adj_data_path,
    ADD_TO_DB=True,
    ADD_TO_CSV=True,
)

# Update the roll calendars
from sysobjects.roll_calendars import rollCalendar
from sysdata.csv.csv_roll_calendars import csvRollCalendarData

spliced = pd.concat([supplied, generated])
roll_calendars_folder = os.path.join(
    os.sep, "home", "vcaldas", "data", "futures", "roll_calendars_csv"
)
if not os.path.exists(roll_calendars_folder):
    os.makedirs(roll_calendars_folder)

csv_roll_calendars = csvRollCalendarData(datapath=roll_calendars_folder)

roll_calendar = rollCalendar.back_out_from_multiple_prices(multiple_prices)
print("Calendar:")
print(roll_calendar)

# We ignore duplicates since this is run regularly
csv_roll_calendars.add_roll_calendar(
    instrument_code, roll_calendar, ignore_duplication=True
)
