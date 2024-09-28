from sysdata.csv.csv_futures_contract_prices import ConfigCsvFuturesPrices
from sysinit.futures.contract_prices_from_csv_to_arctic import (
    init_db_with_csv_futures_contract_prices_for_code
)
from sysinit.futures.contract_prices_from_split_freq_csv_to_db import (
    init_db_with_split_freq_csv_prices_for_code
)
from syscore.constants import arg_not_supplied

from sysdata.csv.csv_futures_contract_prices import ConfigCsvFuturesPrices
import os
from syscore.fileutils import (
    get_resolved_pathname,
    files_with_extension_in_resolved_pathname,
)
from sysdata.csv.csv_futures_contract_prices import csvFuturesContractPriceData

from syscore.dateutils import MIXED_FREQ, HOURLY_FREQ, DAILY_PRICE_FREQ

def strip_file_names(pathname):
    pathname = "/home/vcaldas/data/parquet/futures_contract_prices"
    # These won't have .csv attached
    resolved_pathname = get_resolved_pathname(pathname)
    print(resolved_pathname)
    file_names = files_with_extension_in_resolved_pathname(resolved_pathname,extension='.parquet')
    file_names = [f for f in file_names if f.startswith("Day")]
    for filename in file_names:
        new_file_name = filename.replace("Day@", "")
        new_full_name = os.path.join(resolved_pathname, new_file_name)
        old_full_name = os.path.join(resolved_pathname, filename + ".parquet")
        print("Rename %s to\n %s" % (old_full_name, new_full_name))

        # os.rename(old_full_name, new_full_name)
    return None
BARCHART_CONFIG = ConfigCsvFuturesPrices(
    input_date_index_name="Time",
    input_skiprows=0,
    input_skipfooter=0,
    input_date_format="%Y-%m-%dT%H:%M:%S%z",
    input_column_mapping=dict(
        OPEN="Open", HIGH="High", LOW="Low", FINAL="Close", VOLUME="Volume"
    ),
)


def init_db_with_csv_futures_contract_prices(
    datapath: str,
    csv_config=arg_not_supplied,
    frequency=MIXED_FREQ,
):
    csv_prices = csvFuturesContractPriceData(datapath)


    instrument_codes = (
        csv_prices.get_list_of_instrument_codes_with_price_data_at_frequency(HOURLY_FREQ)
    )
    instrument_codes.sort()
    print(instrument_codes)
    for instrument_code in instrument_codes:
        init_db_with_split_freq_csv_prices_for_code(
            instrument_code, datapath, csv_config=csv_config)
        

if __name__ == "__main__":
    datapath = "/home/vcaldas/data/barchart"

    target_datapath = "/home/vcaldas/data/parquet/futures_contract_prices"

    init_db_with_csv_futures_contract_prices(datapath=datapath, csv_config=BARCHART_CONFIG)

    # init_db_with_csv_futures_contract_prices(datapath, csv_config=BARCHART_CONFIG, frequency=HOURLY_FREQ)
    # init_db_with_csv_futures_contract_prices(datapath, csv_config=BARCHART_CONFIG, frequency=DAILY_PRICE_FREQ)
    # # print("Stripping file names")
    # strip_file_names(target_datapath)
