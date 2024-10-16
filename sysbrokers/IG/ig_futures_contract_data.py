from sysbrokers.broker_futures_contract_data import brokerFuturesContractData
from sysbrokers.broker_instrument_data import brokerFuturesInstrumentData
from sysdata.futures_spreadbet.market_info_data import marketInfoData
from sysdata.data_blob import dataBlob
from sysobjects.contract_dates_and_expiries import expiryDate
from sysobjects.contracts import futuresContract
from sysobjects.production.trading_hours.trading_hours import listOfTradingHours
from syslogging.logger import *
from sysbrokers.IG.ig_connection import IGConnection
from syscore.exceptions import missingContract, missingInstrument, missingData
from syscore.dateutils import contract_month_from_number
from sysbrokers.IG.ig_instruments_data import (
    get_instrument_object_from_config,
)
from sysdata.barchart.bc_connection import bcConnection


class IgFuturesContractData(brokerFuturesContractData):
    def __init__(
        self,
        broker_conn: IGConnection,
        data: dataBlob,
        log=get_logger("IgFuturesContractData"),
    ):
        super().__init__(log=log, data=data)
        self._broker_conn = broker_conn
        self._barchart = bcConnection()

    def __repr__(self):
        return f"IG FSB per contract data: {self._broker_conn}"

    @property
    def broker_conn(self):
        return self._broker_conn

    @property
    def barchart(self):
        return self._barchart

    @property
    def ig_instrument_data(self) -> brokerFuturesInstrumentData:
        return self.data.broker_futures_instrument

    @property
    def market_info_data(self) -> marketInfoData:
        return self.data.db_market_info

    def get_actual_expiry_date_for_single_contract(
        self, futures_contract: futuresContract
    ) -> expiryDate:
        """
        Get the actual expiry date of a contract

        :param futures_contract: type futuresContract
        :return: YYYYMMDD or None
        """

        if futures_contract.is_spread_contract():
            self.log.warning(
                "Can't find expiry for multiple leg contract here",
                **futures_contract.log_attributes(),
                method="temp",
            )
            raise missingContract

        if self._is_fsb(futures_contract):
            return self._get_fsb_expiry_date(futures_contract)
        else:
            return self._get_futures_expiry_date(futures_contract)

    def get_contract_object_with_config_data(
        self, futures_contract: futuresContract, requery_expiries: bool = True
    ) -> futuresContract:
        """
        Return contract_object with config data and correct expiry date added

        :param futures_contract:
        :return: modified contract_object
        :param requery_expiries:
        :type requery_expiries:
        """

        futures_contract_plus = self._get_contract_object_plus(futures_contract)

        if requery_expiries:
            futures_contract_plus = (
                futures_contract_plus.update_expiry_dates_one_at_a_time_with_method(
                    self._get_actual_expiry_date_given_single_contract_plus
                )
            )

        return futures_contract_plus

    def _is_fsb(self, contract_object: futuresContract) -> bool:
        return contract_object.instrument_code.endswith("_fsb")

    def _get_fsb_expiry_date(self, contract_object: futuresContract) -> expiryDate:
        contract_object_with_config_data = self.get_contract_object_with_config_data(
            contract_object
        )

        expiry_date = contract_object_with_config_data.expiry_date

        return expiry_date

    def _get_futures_expiry_date(self, contract_object: futuresContract) -> expiryDate:
        contract_object_with_config_data = self.get_contract_object_with_config_data(
            contract_object
        )

        barchart_id = self.get_barchart_id(contract_object_with_config_data)
        future_expiry = self.barchart.get_expiry_date_for_symbol(barchart_id)
        if future_expiry is None:
            self.log.warning(
                "Unable to get expiry from broker, calculating approx date"
            )
            exp_date = self.calc_approx_expiry(contract_object)
        else:
            exp_date = expiryDate.from_str(future_expiry, format="%m/%d/%y")
            exp_date.source = "B"

        return exp_date

    def _get_contract_object_plus(
        self, contract_object: futuresContract
    ) -> futuresContract:
        try:
            futures_contract_plus = (
                self.ig_instrument_data.get_futures_instrument_object_with_ig_data(
                    contract_object.instrument_code
                )
            )
        except missingInstrument as e:
            raise missingContract from e

        futures_contract_plus = (
            contract_object.new_contract_with_replaced_instrument_object(
                futures_contract_plus
            )
        )

        return futures_contract_plus

    def _get_actual_expiry_date_given_single_contract_plus(
        self, futures_contract_plus: futuresContract
    ) -> expiryDate:
        if futures_contract_plus.is_spread_contract():
            self.log.warning("Can't find expiry for multiple leg contract here")
            raise missingContract

        if self._is_fsb(futures_contract_plus):
            expiry_date = self._get_expiry_fsb(futures_contract_plus)
            date_format_str = "%Y-%m-%d %H:%M:%S"
        else:
            expiry_date = self._get_expiry_future(futures_contract_plus)
            date_format_str = "%m/%d/%y"

        if expiry_date is None:
            self.log.warning(
                f"Failed to get expiry for contract {futures_contract_plus}, returning approx date"
            )
            return self.calc_approx_expiry(futures_contract_plus)
        else:
            expiry_date = expiryDate.from_str(expiry_date, format=date_format_str)
            expiry_date.source = "B"

        return expiry_date

    def calc_approx_expiry(self, futures_contract_plus):
        datestring = futures_contract_plus.date_str
        if datestring[6:8] == "00":
            datestring = datestring[:6] + "28"
        expiry_date = expiryDate.from_str(datestring, format="%Y%m%d")
        expiry_date.source = "E"
        return expiry_date

    def _get_expiry_fsb(self, futures_contract_plus: futuresContract) -> str:
        if futures_contract_plus.key in self.market_info_data.expiry_dates:
            date_str = self.market_info_data.expiry_dates[futures_contract_plus.key]
        else:
            date_str = None

        return date_str

    def _get_expiry_future(self, contract_object: futuresContract) -> str:
        barchart_id = self.get_barchart_id(contract_object)
        date_str = self.barchart.get_expiry_date_for_symbol(barchart_id)
        return date_str

    def get_min_tick_size_for_contract(self, contract_object: futuresContract) -> float:
        return 0.01

    def is_contract_okay_to_trade(self, futures_contract: futuresContract) -> bool:
        try:
            epic = self.market_info_data.get_epic_for_contract(futures_contract)
        except missingData:
            self.log.warning(
                f"No epic found for '{futures_contract}' - perhaps it expired? "
                f"OK to trade now: False"
            )
            return False

        trading_hours = self.market_info_data.get_trading_hours_for_epic(epic)

        mkt_status = self.market_info_data.get_status_for_epic(epic)
        tradeable = mkt_status == "TRADEABLE"

        ok_to_trade = tradeable and trading_hours.okay_to_trade_now()
        self.log.info(
            f"Epic '{epic}' status '{mkt_status}', "
            f"in hours '{trading_hours.okay_to_trade_now()}', "
            f"ok to trade now: {ok_to_trade}"
        )

        return ok_to_trade

    def get_trading_hours_for_contract(
        self, futures_contract: futuresContract
    ) -> listOfTradingHours:
        try:
            epic = self.market_info_data.get_epic_for_contract(futures_contract)
            return self.market_info_data.get_trading_hours_for_epic(epic)
        except missingData:
            self.log.warning(
                f"No epic found for '{futures_contract}' - perhaps it expired? "
                f"Returning empty list of trading hours"
            )
            return listOfTradingHours([])

    def get_list_of_contract_dates_for_instrument_code(
        self, instrument_code: str, allow_expired: bool = False
    ):
        raise NotImplementedError("Consider implementing for consistent interface")

    def get_all_contract_objects_for_instrument_code(self, *args, **kwargs):
        raise NotImplementedError("Consider implementing for consistent interface")

    def _get_contract_data_without_checking(
        self, instrument_code: str, contract_date: str
    ) -> futuresContract:
        raise NotImplementedError("Consider implementing for consistent interface")

    def is_contract_in_data(self, *args, **kwargs):
        raise NotImplementedError("Consider implementing for consistent interface")

    def _delete_contract_data_without_any_warning_be_careful(
        self, instrument_code: str, contract_date: str
    ):
        raise NotImplementedError("Consider implementing for consistent interface")

    def _add_contract_object_without_checking_for_existing_entry(
        self, contract_object: futuresContract
    ):
        raise NotImplementedError("Consider implementing for consistent interface")

    def get_barchart_id(self, futures_contract: futuresContract) -> str:
        """
        Converts a contract identifier from internal format (GOLD/20200900), into Barchart format (GCM20)
        :param futures_contract: the internal format identifier
        :type futures_contract: futuresContract
        :return: Barchart format identifier
        :rtype: str
        """
        date_obj = datetime.datetime.strptime(
            futures_contract.contract_date.date_str, "%Y%m00"
        )

        config_data = get_instrument_object_from_config(
            futures_contract.instrument_code, config=self.ig_instrument_data.config
        )
        # bc_symbol = config_data.bc_code
        symbol = f"{config_data.bc_code}{contract_month_from_number(int(date_obj.strftime('%m')))}{date_obj.strftime('%y')}"
        return symbol


if __name__ == "__main__":
    from sysproduction.data.broker import dataBroker
    from sysobjects.contracts import futuresContract as fc

    data = dataBlob(
        csv_data_paths=dict(
            csvFuturesInstrumentData="fsb.csvconfig",
            csvRollParametersData="fsb.csvconfig",
        ),
    )
    broker = dataBroker(data)
    con = fc.from_two_strings("GOLD_fsb", "20231200")
    ok_to_trade = broker.broker_futures_contract_data.is_contract_okay_to_trade(con)
