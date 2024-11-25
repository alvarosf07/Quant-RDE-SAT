# Import utils:
import time
from datetime import datetime
from ib_insync import *

# Import other classes:
from ....config.config_data.api_credentials import *
from ...qrde1_data.api.api_baseclass import *

"""


# convert to pandas dataframea


market_data = ib.reqMktData(stock, '', False, False)

def onPendingTicker(ticker):
    print("pending ticker event received")
    print(ticker)

ib.pendingTickersEvent += onPendingTicker

ib.run()
"""


class InteractiveBrokersAPI(APIBaseClass):
    def __init__(self):
        super(InteractiveBrokersAPI, self).__init__()

        self.api_name = credentials_InteractiveBrokersAPI["api_name"]
        self.__clientId = credentials_InteractiveBrokersAPI["api_key"]
        self.__host = credentials_InteractiveBrokersAPI["host"]
        self.__port = credentials_InteractiveBrokersAPI["port"]

        self.function_HistoricalData = "reqHistoricalData"


    def __repr__(self):
        return self.api_name

    def __str__(self):
        return self.api_name

    def _api_authentication(self, api_source):
        if api_source == self.api_name:
            return True
        else:
            print(f"Wrong API call. {self.api_name} cannot resolve call intended for '{api_source}' \n")
            return False

    def _create_call(self, contract, sd, ed, frequency_interval=None, metric=None, extended_hours=None,
                     data_timezone=None, keep_updated=None, chart_options=None, timeout=None):
        """
            Args:
            contract: Contract of interest.
            endDateTime: Can be set to '' to indicate the current time,
                or it can be given as a datetime.date or datetime.datetime,
                or it can be given as a string in 'yyyyMMdd HH:mm:ss' format.
                If no timezone is given then the TWS login timezone is used.
            durationStr: Time span of all the bars. Examples:
                '60 S', '30 D', '13 W', '6 M', '10 Y'.
                *Input to assetclass will be sd and ed. From those, the durationStr will be calculated
            barSizeSetting: Time period of one bar. Must be one of:
                '1 secs', '5 secs', '10 secs' 15 secs', '30 secs',
                '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins',
                '20 mins', '30 mins',
                '1 hour', '2 hours', '3 hours', '4 hours', '8 hours',
                '1 day', '1 week', '1 month'.
            whatToShow: Specifies the source for constructing bars.
                Must be one of:
                'TRADES', 'MIDPOINT', 'BID', 'ASK', 'BID_ASK',
                'ADJUSTED_LAST', 'HISTORICAL_VOLATILITY',
                'OPTION_IMPLIED_VOLATILITY', 'REBATE_RATE', 'FEE_RATE',
                'YIELD_BID', 'YIELD_ASK', 'YIELD_BID_ASK', 'YIELD_LAST'.
                For 'SCHEDULE' use :meth:`.reqHistoricalSchedule`.
            useRTH: If True then only show data from within Regular
                Trading Hours, if False then show all data.
            formatDate: For an intraday request setting to 2 will cause
                the returned date fields to be timezone-aware
                datetime.datetime with UTC timezone, instead of local timezone
                as used by TWS.
            keepUpToDate: If True then a realtime subscription is started
                to keep the bars updated; ``endDateTime`` must be set
                empty ('') then.
            chartOptions: Unknown.
            timeout: Timeout in seconds after which to cancel the request
                and return an empty bar series. Set to ``0`` to wait
                indefinitely.

        """

        # Calculating durationStr
        duration_str = self._get_api_durationstr(sd, ed)

        # Converting date to correct format
        end_date = self._format_date(ed, input_format="%Y.%m.%d", output_format="YYYYMMDD HH:mm:ss")
        print(end_date)

        # Calculating barSizeSetting
        frequency_interval2 = self._get_api_interval(frequency_interval)

        call_params = {
            "contract": contract,
            "endDateTime": end_date,
            "durationStr": duration_str,
            "barSizeSetting": frequency_interval2,
            "whatToShow": metric,
            "useRTH": extended_hours == "true",
            "formatDate": data_timezone,
            "keepUpToDate": keep_updated,
            "chartOptions": chart_options,
            "timeout": timeout}

        return call_params

    def _make_request(self, call_endpoint=None, call_params=None, function=None):
        # 0) Removing null call parameters
        call_endpoint = call_endpoint or {"host": self.__host, "port": self.__port, "clientId": self.__clientId}
        call_params = {key: value for key, value in call_params.items() if value is not None}

        # I) Initialization of Interactive Brokers Class /Should I move this to the InteractiveBrokerAPI class creation??
        ib = IB()
        ib.connect(host=call_endpoint["host"], port=call_endpoint["port"], clientId=call_endpoint["clientId"])

        print(f"\nSourcing data from InteractiveBrokersAPI:\n --> ib.{function}{call_params} \n")
        data = getattr(ib, function)(**call_params)

        return data

    @staticmethod
    def _get_api_durationstr(sd, ed):
        sd = datetime.strptime(sd, "%Y.%m.%d")
        ed = datetime.strptime(ed, "%Y.%m.%d")
        delta = ed - sd
        duration_str = str(delta.days) + " D"

        return duration_str

    @staticmethod
    def _format_date(date, input_format="%Y.%m.%d", output_format="%Y%m%d %H:%M:%S"):
        """
            Consider moving to "datecalcs" page
        """
        d = datetime.strptime(date, input_format)
        d = d.strftime("%Y%m%d %H:%M:%S")

        return d

    @staticmethod
    def _get_api_interval(frequency_interval):

        freq_map = {
            "1s": "1 secs",
            "5s": "5 secs",
            "10s": "10 secs",
            "15s": "15 secs",
            "30s": "5 secs",
            "1min": "1 min",
            "2min": "2 mins",
            "3min": "3 mins",
            "5min": "5 mins",
            "10min": "10 mins",
            "15min": "15 mins",
            "20min": "20 mins",
            "30min": "30 mins",
            "60min": "1 hour",
            "1h": "1 hour",
            "2h": "2 hours",
            "3h": "3 hours",
            "4h": "4 hours",
            "8h": "8 hours",
            "daily": "1 day",
            "weekly": "1 week",
            "monthly": "1 month"}
        return freq_map[frequency_interval]

    def get_ohlcv_data_stock(self, symbol, sd, ed, frequency_interval, exchange="SMART", currency="USD", api_source="InteractiveBrokersAPI",
                             metric="MIDPOINT", data_adjusted="true", extended_hours="true", format_date=1, timeout=None, output_size="full"):

        # 0) Verification that this is the correct API to send the request
        if not self._api_authentication(api_source):
            return None

        # I) Data Cleaning

        # II) Creation of the call parameters
        stock = Stock(symbol=symbol, exchange=exchange, currency=currency)
        call_params = self._create_call(contract=stock, sd=sd, ed=ed, frequency_interval=frequency_interval, metric=metric,
                                        extended_hours=extended_hours, data_timezone=format_date, timeout=timeout)

        # III) Creation of the call endpoint
        call_endpoint = {"host": self.__host, "port": self.__port, "clientId": self.__clientId}

        # IV) Make API call request
        data = self._make_request(call_endpoint=call_endpoint, call_params=call_params, function=self.function_HistoricalData)

        # IV) Create data frame
        df = util.df(data)
        print(df)

        # V) Filter by correct date intervals
        try:
            #df_interval = df[(df['Date'] > sd) & (df['Date'] < ed)]
            return df
        except TypeError:
            return "No Output Provided"
