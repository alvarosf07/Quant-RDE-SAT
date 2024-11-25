# Import utils:
import requests
import pandas as pd
import dateutil.parser

# Import Internal Classes:
from ....config.config_research.default_parameters import *
from ....config.config_data.api_credentials import *
from ...qrde1_data.api.api_baseclass import *


class AlphaVantageAPI(APIBaseClass):
    def __init__(self):
        super(AlphaVantageAPI, self).__init__()

        self.api_name = credentials_AlphaVantageAPI["api_name"]
        self.api_key = credentials_AlphaVantageAPI["api_key"]
        self.server = credentials_AlphaVantageAPI["server"]
        self.endpoint = credentials_AlphaVantageAPI["endpoint"]

        self.function_stock_intraday = "TIME_SERIES_INTRADAY"
        self.function_stock_daily = "TIME_SERIES_DAILY"
        self.function_stock_daily_adjusted = "TIME_SERIES_DAILY_ADJUSTED"
        self.function_stock_weekly = "TIME_SERIES_WEEKLY"
        self.function_stock_weekly_adjusted = "TIME_SERIES_WEEKLY_ADJUSTED"
        self.function_stock_monthly = "TIME_SERIES_MONTHLY"
        self.function_stock_monthly_adjusted = "TIME_SERIES_MONTHLY_ADJUSTED"

        self.function_fx_exchange_rate = "CURRENCY_EXCHANGE_RATE"
        self.function_fx_intraday = "FX_INTRADAY"
        self.function_fx_daily = "FX_DAILY"
        self.function_fx_weekly = "FX_WEEKLY"
        self.function_fx_monthly = "FX_MONTHLY"
        self.function_crypto_intraday = "CRYPTO_INTRADAY"
        self.function_crypto_daily = "DIGITAL_CURRENCY_DAILY"
        self.function_crypto_weekly = "DIGITAL_CURRENCY_WEEKLY"
        self.function_crypto_monthly = "DIGITAL_CURRENCY_MONTHLY"

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

    def _create_call(self, function, symbol=None, frequency_interval=None, output_size=None, data_adjusted=None, extended_hours=None,
                     data_type="json", from_currency=None, to_currency=None, from_symbol=None, to_symbol=None, market=None):
        """
            Parameters Information:

            function --> name of the function to source data
            interval --> Time interval between two consecutive data points in the time series. The following values are supported:
                         1min, 5min, 15min, 30min, 60min, (daily, weekly, monthly)
                         *(daily, weekly, monthly) frequencies are inputs to asset class, but are sourced as functions instead of intervals
            output_size --> By default, outputsize=compact.
                                - "compact" returns only the latest 100 data points in the intraday time series;
                                - "full" returns trailing 30 days of the most recent intraday data if the month parameter (see above) is not specified,
                                  or the full intraday data for a specific month in history if the month parameter is specified.
            data_adjusted --> By default, adjusted=true and the output time series is adjusted by historical split and dividend events.
                              Set adjusted=false to query raw (as-traded) intraday values.
            extended_hours --> By default, extended_hours=true and the output time series will include both the regular trading hours and
                               the extended trading hours (4:00am to 8:00pm Eastern Time for the US market).
                               Set extended_hours=false to query regular trading hours (9:30am to 4:00pm US Eastern Time) only.

        """

        call_params = {
                        "function": function,
                        "symbol": symbol,
                        "from_currency": from_currency,
                        "to_currency": to_currency,
                        "from_symbol": from_symbol,
                        "to_symbol": to_symbol,
                        "market": market,
                        "interval": frequency_interval,
                        "outputsize": output_size,
                        "adjusted": data_adjusted,
                        "extended_hours": extended_hours,
                        "datatype": data_type,
                        "apikey": self.api_key}

        return call_params

    def _make_request(self, endpoint, call_params=None):
        # I) Creation of the url from the server plus the call parameters
        base_url = f"{self.server}{endpoint}?"
        params = "".join(["&" + key + "=" + value for key, value in call_params.items() if value])
        print(f"\nSourcing data from API:\n --> {base_url + params} \n")

        r = requests.get(base_url, params=params)
        data = r.json()

        return data

    def _get_api_function_interval_stock(self, frequency_interval, data_adjusted):
        function = None
        interval = None

        if frequency_interval in {"1min", "5min", "15min", "30min", "60min"}:
            function = self.function_stock_intraday
            interval = frequency_interval

        elif frequency_interval == "daily":
            if data_adjusted == "false":
                function = self.function_stock_daily
            if data_adjusted == "true":
                function = self.function_stock_daily_adjusted

        elif frequency_interval == "weekly":
            if data_adjusted == "false":
                function = self.function_stock_weekly
            if data_adjusted == "true":
                function = self.function_stock_weekly_adjusted

        elif frequency_interval == "monthly":
            if data_adjusted == "false":
                function = self.function_stock_monthly
            if data_adjusted == "true":
                function = self.function_stock_monthly_adjusted

        return function, interval

    def _get_api_function_interval_fx(self, frequency_interval):
        function = None
        interval = None

        if frequency_interval in {"1min", "5min", "15min", "30min", "60min"}:
            function = self.function_fx_intraday
            interval = frequency_interval

        elif frequency_interval == "daily":
            function = self.function_fx_daily

        elif frequency_interval == "weekly":
            function = self.function_fx_weekly

        elif frequency_interval == "monthly":
            function = self.function_fx_monthly

        return function, interval

    def _get_api_function_interval_crypto(self, frequency_interval):
        function = None
        interval = None

        if frequency_interval in {"1min", "5min", "15min", "30min", "60min"}:
            function = self.function_crypto_intraday
            interval = frequency_interval

        elif frequency_interval in {"1min", "5min", "15min", "30min", "1h"}:
            function = self.function_crypto_intraday
            interval = "60min"

        elif frequency_interval == "daily":
            function = self.function_crypto_daily

        elif frequency_interval == "weekly":
            function = self.function_crypto_weekly

        elif frequency_interval == "monthly":
            function = self.function_crypto_monthly

        return function, interval

    @staticmethod
    def _create_dataframe_uniform_headers(df):
        headers_list = df.columns.values.tolist()

        headers_dict = {}
        for e in headers_list:
            if e == "index":
                headers_dict[e] = "Date"
            elif "open" in e:
                headers_dict[e] = "Open"
            elif "high" in e:
                headers_dict[e] = "High"
            elif "low" in e:
                headers_dict[e] = "Low"
            elif ("close" in e) and ("adjusted" not in e):
                headers_dict[e] = "Close"
            elif "volume" in e:
                headers_dict[e] = "Volume"
            elif "adjusted close" in e:
                headers_dict[e] = "Adjusted Close"
            elif "dividend amount" in e:
                headers_dict[e] = "Dividend Amount"

        df.rename(columns=headers_dict, inplace=True)

        return df

    @staticmethod
    def _create_dataframe_uniform_headers_crypto(df):
        headers_list = df.columns.values.tolist()

        headers_dict = {}
        for e in headers_list:
            if e == "index":
                headers_dict[e] = "Date"
            elif "open" in e:
                headers_dict[e] = "Open" + e[8:]
            elif "high" in e:
                headers_dict[e] = "High" + e[8:]
            elif "low" in e:
                headers_dict[e] = "Low" + e[7:]
            elif "close" in e:
                headers_dict[e] = "Close" + e[9:]
            elif "volume" in e:
                headers_dict[e] = "Volume"
            elif "market cap" in e:
                headers_dict[e] = "Market Cap" + e[13:]

        df.rename(columns=headers_dict, inplace=True)

        return df

    @staticmethod
    def _create_dataframe_uniform_headers_data(df):
        headers_list = df.columns.values.tolist()

        headers_dict = {}
        for e in headers_list:
            if e == "date":
                headers_dict[e] = "Date"
            elif "value" in e:
                headers_dict[e] = "Value"

        df.rename(columns=headers_dict, inplace=True)

        return df

    def _create_dataframe_ts(self, data, data_type="default"):  # Create dataframe with time-series data
        """
            Consider Moving to lib > lib_data > dataframes
        """

        # 1) We convert data to dictionary of values
        try:
            data_dict = [data[k] for k in data.keys() if "Time Series" in k][0]
        except IndexError:
            print(f'INDEX ERROR. A problem occurred while trying to access the query provided: \n "{data.items}" \n')
            return data.items

        # 2) We create pandas dataframe and reset the index for the date column
        df = pd.DataFrame(data_dict).T
        df.reset_index(inplace=True)

        # 3) Convert the default headers to uniform headers
        if data_type == "crypto":
            df = self._create_dataframe_uniform_headers_crypto(df)
        else:
            df = self._create_dataframe_uniform_headers(df)

        return df

    @staticmethod
    def _create_dataframe_rt(self, data, data_type="default"):  # Create dataframe with real-time data
        """
            Consider Moving to lib > lib_data > dataframes
        """

        # 1) We convert data to dictionary of values
        try:
            data_dict = [data[k] for k in data.keys() if "Realtime" in k][0]
        except IndexError:
            print(data)
            return data.items

        # 2) We create pandas dataframe and reset the index for the date column
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        df.reset_index(inplace=True)

        # 3) Convert the default headers to uniform headers
        df.rename(columns={'index': 'key', 0: 'value'}, inplace=True)

        return df

    def _create_dataframe_data(self, data):  # Create dataframe with simple data
        """
            Consider Moving to lib > lib_data > dataframes
        """

        # 1) We convert data to dictionary of values
        try:
            data_dict = [data[k] for k in data.keys() if "data" in k][0]
        except IndexError:
            print(f'INDEX ERROR. A problem occurred while trying to access the query provided: \n "{data.items}" \n')
            return data.items

        # 2) We create pandas dataframe and reset the index for the date column
        df = pd.DataFrame(data_dict)

        # 3) Convert the default headers to uniform headers
        df = self._create_dataframe_uniform_headers_data(df)

        return df

    def get_ohlcv_data_stock(self, symbol, sd, ed, frequency_interval, exchange=None,  currency='USD', api_source='AlphaVantageAPI',
                             metric=None, data_adjusted='true', extended_hours='true', format_date=1, timeout=None, output_size='full',
                             data_type='json'):

        # 0) Verification that this is the correct API to send the request
        if not self._api_authentication(api_source):
            return None

        # I) Data Cleaning: Based on interval selected and whether data is adjusted, the correct AlphaVantage function is selected
        function, frequency_interval2 = self._get_api_function_interval_stock(frequency_interval, data_adjusted)

        # II) Creation of the call parameters
        call_params = self._create_call(function, symbol, frequency_interval2, output_size,
                                        data_adjusted, extended_hours, data_type)

        # III) Make API call request
        data = self._make_request(self.endpoint, call_params)

        # IV) Create data frame
        df = self._create_dataframe_ts(data)

        # V) Filter by correct date intervals
        #sd = dateutil.parser.parse(sd)
        #ed = dateutil.parser.parse(ed)
        try:
            df_interval = df[(df['Date'] > sd) & (df['Date'] < ed)]
            print(df_interval)
            return df_interval
        except TypeError:
            return "Type Error. No Output Provided"

    def save_ohlcv_data_stock(self, symbol, sd, ed, frequency_interval, exchange=None, currency="USD", api_source="AlphaVantageAPI",
                              metric=None, data_adjusted="true", extended_hours="true", format_date=1, timeout=None, output_size="full",
                              data_type="csv", file_name=default_csv_filename, save_path=default_save_path_asset):

        # I) Get the stock data
        df = self.get_ohlcv_data_stock(symbol, sd, ed, frequency_interval, api_source, data_adjusted,
                                       extended_hours, output_size)
        # 2) Save the data
        try:
            # print(f" save_path:{save_path}, \n file_name: {file_name}, \n data_type: {data_type}")
            filepath = save_path + file_name + '.' + data_type
            df.to_csv(filepath, index=False)
            print(f"Data saved to: {filepath}")
        except AttributeError:  # AttributeError: 'str' object has no attribute 'to_csv'
            print(f"\nError saving data to {data_type}. No Output Provided")
            return

        return

    def get_latest_exchange_rate(self, from_currency=None, to_currency="USD", api_source=default_api_name):
        # I) Verification that this is the correct API to send the request
        if not self._api_authentication(api_source):
            return None

        # II) Creation of the call parameters
        call_params = self._create_call(function=self.function_fx_exchange_rate, from_currency=from_currency, to_currency=to_currency)

        # III) Make API call request
        data = self._make_request(self.endpoint, call_params)

        # IV) Create data frame
        df = self._create_dataframe_rt(data)

        return df

    def get_latest_exchange_rate_int(self, from_currency=None, to_currency="USD", api_source=default_api_name):
        df = self.get_latest_exchange_rate(from_currency, to_currency, api_source)

        r = df.loc[df.key == '5. Exchange Rate', 'value'].item()

        return r

    def save_latest_exchange_rate(self, from_currency=None, to_currency="USD", api_source=default_api_name,
                                  data_type="csv", file_name=default_csv_filename, save_path=default_save_path_asset):
        # I) Get the stock data
        df = self.get_latest_exchange_rate(from_currency, to_currency, api_source)

        # 2) Save the data
        try:
            # print(f" save_path:{save_path}, \n file_name: {file_name}, \n data_type: {data_type}")
            filepath = save_path + file_name + '.' + data_type
            df.to_csv(filepath, index=False)
            print(f"Data saved to: {filepath}")
        except AttributeError:  # AttributeError: 'str' object has no attribute 'to_csv'
            print(f"\nError saving data to {data_type}. No Output Provided")
            return

        return

    def get_ohlcv_data_fx(self, sd, ed, frequency_interval, from_symbol, to_symbol, api_source, output_size="full", data_type="json"):
        # I) Verification that this is the correct API to send the request
        if not self._api_authentication(api_source):
            return None

        # II) Based on interval selected and whether data is adjusted, the correct AlphaVantage function is selected
        function, frequency_interval2 = self._get_api_function_interval_fx(frequency_interval)

        # III) Creation of the call parameters
        call_params = self._create_call(function=function, frequency_interval=frequency_interval2,
                                        output_size=output_size, data_type=data_type,
                                        from_symbol=from_symbol, to_symbol=to_symbol)

        # IV) Make API call request
        data = self._make_request(self.endpoint, call_params)

        # V) Create data frame
        df = self._create_dataframe_ts(data)

        # VI) Filter by correct date intervals
        sd = dateutil.parser.parse(sd)
        ed = dateutil.parser.parse(ed)
        try:
            df_interval = df[(df['Date'] > sd) & (df['Date'] < ed)]
            return df_interval
        except TypeError:
            return "No Output Provided"

    def save_ohlcv_data_fx(self, sd, ed, frequency_interval, from_symbol, to_symbol, api_source, output_size="full", data_type="json",
                           file_name=default_csv_filename, save_path=default_save_path_asset):
        # I) Get the stock data
        df = self.get_ohlcv_data_fx(sd, ed, frequency_interval, from_symbol, to_symbol, api_source, output_size)

        # 2) Save the data
        try:
            # print(f" save_path:{save_path}, \n file_name: {file_name}, \n data_type: {data_type}")
            filepath = save_path + file_name + '.' + data_type
            df.to_csv(filepath, index=False)
            print(f"Data saved to: {filepath}")
        except AttributeError:  # AttributeError: 'str' object has no attribute 'to_csv'
            print(f"\nError saving data to {data_type}. No Output Provided")
            return

        return

    def get_ohlcv_data_crypto(self, symbol, sd, ed, frequency_interval, exchange=None, currency='USD', api_source='AlphaVantageAPI', metric=None,
                              data_adjusted='true', extended_hours='true', format_date=1, timeout=None, output_size="full", data_type="json"):
        # I) Verification that this is the correct API to send the request
        if not self._api_authentication(api_source):
            return None

        # II) Based on interval selected and whether data is adjusted, the correct AlphaVantage function is selected
        function, frequency_interval2 = self._get_api_function_interval_crypto(frequency_interval)

        # III) Creation of the call parameters
        call_params = self._create_call(function=function, frequency_interval=frequency_interval2, output_size=output_size, data_type=data_type,
                                        symbol=symbol, market=currency)

        # IV) Make API call request
        data = self._make_request(self.endpoint, call_params)

        # V) Create data frame
        df = self._create_dataframe_ts(data, data_type="crypto")

        # VI) Filter by correct date intervals
        sd = dateutil.parser.parse(sd)
        ed = dateutil.parser.parse(ed)
        try:
            df_interval = df[(df['Date'] > sd) & (df['Date'] < ed)]
            return df_interval
        except TypeError:
            return "No Output Provided"

    def save_ohlcv_data_crypto(self, symbol, sd, ed, frequency_interval, exchange=None, currency='USD', api_source='AlphaVantageAPI', metric=None,
                               data_adjusted='true', extended_hours='true', format_date=1, timeout=None, output_size="full", data_type="json",
                               file_name=default_csv_filename, save_path=default_save_path_asset):
        # I) Get the stock data
        df = self.get_ohlcv_data_crypto(symbol=symbol, sd=sd, ed=ed, frequency_interval=frequency_interval, currency=currency,
                                        api_source=api_source, output_size=output_size)

        # 2) Save the data
        try:
            # print(f" save_path:{save_path}, \n file_name: {file_name}, \n data_type: {data_type}")
            filepath = save_path + file_name + '.' + data_type
            df.to_csv(filepath, index=False)
            print(f"Data saved to: {filepath}")
        except AttributeError:  # AttributeError: 'str' object has no attribute 'to_csv'
            print(f"\nError saving data to {data_type}. No Output Provided")
            return

        return

    def get_historical_price_data_commodities(self, function, sd, ed, frequency_interval, api_source, data_type="json"):
        # I) Verification that this is the correct API to send the request
        if not self._api_authentication(api_source):
            return None

        # II) Creation of the call parameters
        call_params = self._create_call(function=function, frequency_interval=frequency_interval, data_type=data_type)

        # III) Make API call request
        data = self._make_request(self.endpoint, call_params)

        # IV) Create data frame
        df = self._create_dataframe_data(data)

        # VI) Filter by correct date intervals
        sd = dateutil.parser.parse(sd)
        ed = dateutil.parser.parse(ed)
        try:
            df_interval = df[(df['Date'] > sd) & (df['Date'] < ed)]
            return df_interval
        except TypeError:
            return "No Output Provided"

    def save_historical_price_data_commodities(self, function, sd, ed, frequency_interval, api_source, data_type="json",
                                               file_name=default_csv_filename, save_path=default_save_path_asset):
        # I) Get the stock data
        df = self.get_historical_price_data_commodities(function, sd, ed, frequency_interval, api_source)

        # 2) Save the data
        try:
            # print(f" save_path:{save_path}, \n file_name: {file_name}, \n data_type: {data_type}")
            filepath = save_path + file_name + '.' + data_type
            df.to_csv(filepath, index=False)
            print(f"Data saved to: {filepath}")
        except AttributeError:  # AttributeError: 'str' object has no attribute 'to_csv'
            print(f"\nError saving data to {data_type}. No Output Provided")
            return

        return
