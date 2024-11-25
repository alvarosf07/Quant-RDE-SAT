# Import utils
import sys, os, glob
# import time
# from os.path import dirname, basename, isfile, join

# Import local modules
from ...config.config_data.api_credentials import *
from ...config.config_research.default_parameters import *
from ..qrde1_data.api.api_baseclass import *
from ..qrde1_data.api.alphavantage_api import *
from ..qrde1_data.api.interactivebrokers_api import *

#sys.path.append(module_config)
#sys.path.append(module_lib)
# modules = (glob.glob(join(dirname(module_config_research), "*.py"))
#            +glob.glob(join(dirname(module_APIs), "*.py"))
#            +glob.glob(join(dirname(__file__), "*.py")))
# __all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

# print(modules)
# print(__all__)


class Asset(object):
    """
    An asset is any sequence of observations that is timestamped,
    has several values like price and volume, and can be bought or sold.

    Assets can receive trading actions on them: buy, sell or hold.
    Assets can be performed raw data conversions: aggregations, differences and other transformations.
    Assets can be grouped in portfolios (see portfolio class).

    An asset can be initialized with:
    - symbol: trading ticker used by the asset

    """

    financial_class = "Asset"

    def __init__(self, symbol):
        self.symbol = symbol
        self.name = self.get_asset_name_from_symbol(self.symbol)
        self.time_creation = time.time()
        self.time_last_update = time.time()
        self.api_instance_dict = {}
        self.api_instance_base_class = APIBaseClass()
        self.api_instance_default = self._init_api_instance(default_api_name)

    def _init_api_instance(self, api_source):
        # 0) Return api_source if already initialized in the asset class dict
        if api_source in self.api_instance_dict:
            return self.api_instance_dict[api_source]

        # I) Listing subclasses of APIBaseClass (api_sources)
        api_sources = self.api_instance_base_class.get_api_sources()

        # II) Initializing the proper subclass of APIBaseClass (introduced by user in api_source)
        if api_source in api_sources:
            api_instance = getattr(sys.modules[__name__], api_source)()
            self.api_instance_dict[api_instance.api_name] = api_instance
            return api_instance

        elif api_source is None or api_source == "None" or api_source == "-":
            print("Data source not provided. API instance not created.\n")
            return None

        else:
            print(f"Invalid API key selected: '{api_source}' doesn't exist.")
            print("API instance not created.\n")
            return None

    @classmethod
    def list_assets_created(cls):
        assets_list = [a.__name__ for a in cls.__subclasses__()]
        return assets_list

    @staticmethod
    def get_asset_name_from_symbol(symbol):
        """
            TO DO: Implement method by creating a call to an online searcher
        """
        asset_name = symbol
        return asset_name


class EquityAsset (Asset):
    """
        Equity Assets represent ownership interests in a company.

        There are two subclasses of Equity Assets:
            - Stock (default)
            - Private Equity (must be specified). Private Equity Assets don't have publicly available data

    """

    asset_type = "Financial Asset"
    asset_subtype = "Securities"
    asset_superclass = "Equity"
    asset_class = "Equity"

    def __init__(self, symbol, asset_subclass="Stock"):
        super(EquityAsset, self).__init__(symbol)
        self.asset_subclass = asset_subclass

    def get_ohlcv_data(self, sd, ed, frequency_interval, exchange='SMART', currency='USD', api_source=default_api_name, metric = "MIDPOINT",
                       data_adjusted='false', extended_hours='false', output_size='full'):
        """
            sd --> starting data (yyyy.mm.dd)
            ed --> starting data (yyyy.mm.dd)
            frequency_interval --> time interval between two consecutive data points in the time series. Supported values:
                                    "1s": InteractiveBrokersAPI,
                                    "5s": InteractiveBrokersAPI,
                                    "10s": InteractiveBrokersAPI,
                                    "15s": InteractiveBrokersAPI,
                                    "30s": InteractiveBrokersAPI,
                                    "1min": InteractiveBrokersAPI, AlphaVantageAPI
                                    "2min": InteractiveBrokersAPI,
                                    "3min": InteractiveBrokersAPI,
                                    "5min": InteractiveBrokersAPI, AlphaVantageAPI
                                    "10min": InteractiveBrokersAPI,
                                    "15min": InteractiveBrokersAPI, AlphaVantageAPI
                                    "20min": InteractiveBrokersAPI,
                                    "30min": InteractiveBrokersAPI, AlphaVantageAPI
                                    "60min": InteractiveBrokersAPI, AlphaVantageAPI
                                    "1h": InteractiveBrokersAPI, AlphaVantageAPI
                                    "2h": InteractiveBrokersAPI,
                                    "3h": InteractiveBrokersAPI,
                                    "4h": InteractiveBrokersAPI,
                                    "8h": InteractiveBrokersAPI,
                                    "daily": InteractiveBrokersAPI, AlphaVantageAPI
                                    "weekly": InteractiveBrokersAPI, AlphaVantageAPI
                                    "monthly": InteractiveBrokersAPI, AlphaVantageAPI
        """
        api_instance = self._init_api_instance(api_source)
        output_data = api_instance.get_ohlcv_data_stock(symbol=self.symbol, sd=sd, ed=ed, frequency_interval=frequency_interval, exchange=exchange,
                                                        currency=currency, api_source=api_source, metric=metric, data_adjusted=data_adjusted,
                                                        extended_hours=extended_hours, output_size=output_size, format_date=1, timeout=None)
        return output_data

    def save_ohlcv_data(self, sd, ed, frequency_interval, api_source=default_api_name,
                        data_adjusted="true", extended_hours="true", output_size="full",
                        data_type="csv", file_name=default_csv_filename, save_path=default_save_path_asset):
        """
         Saves the data to specified data path. Returns None
        """
        api_instance = self._init_api_instance(api_source)
        api_instance.save_ohlcv_data_stock(self.symbol, sd, ed, frequency_interval, api_source,
                                           data_adjusted, extended_hours, output_size,
                                           data_type, file_name, save_path)
        return


class EquityPIVAsset (Asset):
    pass


class HybridAsset (Asset):
    pass


class DebtMacroAsset (Asset):
    pass


class DebtCreditAsset (Asset):
    pass


class DerivativeAsset (Asset):
    pass


class CurrencyFiatAsset (Asset):
    """
            Currency Fiat Assets represents investments in foreign currencies on the foreign exchange (forex) market.
    """

    asset_type = "Financial Asset"
    asset_subtype = "Fungibles"
    asset_superclass = "Currency"
    asset_class = "Fiat Currency"
    asset_subclass = "FX"

    def __init__(self, symbol):
        super(CurrencyFiatAsset, self).__init__(symbol)

    def get_latest_exchange_rate(self, from_currency=None, to_currency="USD", api_source=default_api_name):
        api_instance = self._init_api_instance(api_source)

        if from_currency is None:
            from_symbol = self.symbol

        output_data = api_instance.get_latest_exchange_rate(from_currency, to_currency, api_source)

        return output_data

    def get_latest_exchange_rate_int(self, from_currency=None, to_currency="USD", api_source=default_api_name):
        api_instance = self._init_api_instance(api_source)

        if from_currency is None:
            from_symbol = self.symbol

        output_data = api_instance.get_latest_exchange_rate_int(from_currency, to_currency, api_source)

        return output_data

    def get_ohlc_data_fx(self, sd, ed, frequency_interval, from_symbol=None, to_symbol="USD",
                         api_source=default_api_name, output_size="full"):
        api_instance = self._init_api_instance(api_source)

        if from_symbol is None:
            from_symbol = self.symbol

        output_data = api_instance.get_ohlcv_data_fx(sd, ed, frequency_interval, from_symbol, to_symbol,
                                                     api_source, output_size)
        return output_data

    def save_ohlc_data_fx(self, sd, ed, frequency_interval, from_symbol=None, to_symbol="USD", api_source=default_api_name, output_size="full",
                          data_type="csv", file_name=default_csv_filename, save_path=default_save_path_asset):
        """
         Saves the data to specified data path. Returns None
        """
        api_instance = self._init_api_instance(api_source)

        if from_symbol is None:
            from_symbol = self.symbol

        api_instance.save_ohlcv_data_fx(sd, ed, frequency_interval, from_symbol, to_symbol, api_source, output_size,
                                        data_type, file_name, save_path)
        return


class CurrencyDigitalAsset (Asset):
    """
            Currency Digital Assets represent investments in digital currencies.
    """

    asset_type = "Financial Asset"
    asset_subtype = "Fungibles"
    asset_superclass = "Currency"
    asset_class = "Digital Currency"
    asset_subclass = "Cryptocurrency"

    def __init__(self, symbol):
        super(CurrencyDigitalAsset, self).__init__(symbol)

    def get_latest_exchange_rate(self, from_currency=None, to_currency="USD", api_source=default_api_name):
        api_instance = self._init_api_instance(api_source)

        if from_currency is None:
            from_symbol = self.symbol

        output_data = api_instance.get_latest_exchange_rate(from_currency, to_currency, api_source)

        return output_data

    def get_latest_exchange_rate_int(self, from_currency=None, to_currency="USD", api_source=default_api_name):
        api_instance = self._init_api_instance(api_source)

        if from_currency is None:
            from_symbol = self.symbol

        output_data = api_instance.get_latest_exchange_rate_int(from_currency, to_currency, api_source)

        return output_data

    def get_ohlc_data_crypto(self, sd, ed, frequency_interval, exchange='SMART', from_symbol=None, to_symbol="USD", currency='USD',
                             api_source=default_api_name, metric='MIDPOINT', data_adjusted='false', extended_hours='false', output_size="full"):
        api_instance = self._init_api_instance(api_source)

        if from_symbol is None:
            from_symbol = self.symbol

        output_data = api_instance.get_ohlcv_data_crypto(symbol=self.symbol, sd=sd, ed=ed, frequency_interval=frequency_interval, exchange=exchange,
                                                         currency=currency, from_symbol=from_symbol, to_symbol=to_symbol, api_source=api_source,
                                                         metric=metric, data_adjusted=data_adjusted, extended_hours=extended_hours,
                                                         output_size=output_size, format_date=1, timeout=None)

        output_data = api_instance.get_ohlcv_data_crypto(sd, ed, frequency_interval, from_symbol, to_symbol,
                                                         api_source, output_size)
        return output_data

    def save_ohlc_data_crypto(self, sd, ed, frequency_interval, exchange='SMART', from_symbol=None, to_symbol="USD", currency='USD',
                              api_source=default_api_name, metric='MIDPOINT', data_adjusted='false', extended_hours='false', output_size="full",
                              data_type="csv", file_name=default_csv_filename, save_path=default_save_path_asset):
        """
         Saves the data to specified data path. Returns None
        """
        api_instance = self._init_api_instance(api_source)

        if from_symbol is None:
            from_symbol = self.symbol

        api_instance.save_ohlcv_data_crypto(symbol=self.symbol, sd=sd, ed=ed, frequency_interval=frequency_interval, exchange=exchange,
                                            currency=currency, from_symbol=from_symbol, to_symbol=to_symbol, api_source=api_source,
                                            metric=metric, data_adjusted=data_adjusted, extended_hours=extended_hours,output_size=output_size,
                                            format_date=1, timeout=None, data_type=data_type, file_name=file_name, save_path=save_path)
        return


class CurrencyDigitalNFTAsset (Asset):
    pass


class CommodityAsset (Asset):
    """
        Commodity Assets represent investments in commodities.
    """

    asset_type = "Real Assets"
    asset_subtype = "Fungibles"
    asset_superclass = "Commodities"
    asset_class = "Commodities"
    asset_subclass = "Commodities"

    def __init__(self, symbol):
        super(CommodityAsset, self).__init__(symbol)

    def get_historical_price_data(self, sd, ed, frequency_interval, api_source=default_api_name):
        api_instance = self._init_api_instance(api_source)

        output_data = api_instance.get_historical_price_data_commodities(self.symbol, sd, ed, frequency_interval, api_source)

        return output_data

    def save_historical_price_data(self, sd, ed, frequency_interval, api_source=default_api_name,
                                   data_type="csv", file_name=default_csv_filename, save_path=default_save_path_asset):
        """
            Saves the data to specified data path. Returns None
        """
        api_instance = self._init_api_instance(api_source)

        api_instance.save_historical_price_data_commodities(self.symbol, sd, ed, frequency_interval, api_source, data_type, file_name, save_path)

        return


class CommodityCollectibleAsset (Asset):
    pass


class RealEstateAsset (Asset):
    pass




# Improvements:
#    - Init attributes of class Asset: keep asset_name, asset_class, data_type, data_source? or only asset_name and asset_class?
#    - Add functionality: ticker search (see alpha vantage API)


if __name__ == "__main__":
    # MS = EquityAsset('MS')
    # GS = EquityAsset('GS')
    # JPM = EquityAsset('JPM')

    # dataMS = MS.get_ohlcv_data("2023-09-14", "2023.10.20", "weekly", "AlphaVantageAPI", "false", "false")
    # dataGS = GS.get_ohlcv_data("2023-10-14", "2023.10.20", "daily")
    # dataJPM = JPM.get_ohlcv_data("2023-10-14", "2023.10.20", "daily")

    # print(dataMS)
    # print(dataGS)
    # print(dataJPM)

    # MS.save_ohlcv_data("2023-10-14", "2023.10.20", "daily", "AlphaVantageAPI", "false", "false", "full", "csv", "MS_20231016")

    # USD = CurrencyFiatAsset('USD')
    # data_USD_EUR = USD.get_ohlc_data_fx("2023-09-14", "2023-10-20", "weekly", "USD", "EUR")
    # print(data_USD_EUR)
    # USD.save_ohlc_data_fx(from_symbol="EUR", to_symbol="USD", sd="2023-09-14", ed="2023-10-25", frequency_interval="daily", file_name="USD_EUR_20231014")

    # er = USD.get_latest_exchange_rate(from_currency="USD", to_currency="EUR")
    # r = USD.get_latest_exchange_rate_int(from_currency="USD", to_currency="EUR")
    # print (er)
    # print(r)

    # BTC = CurrencyDigitalAsset('BTC')
    # data_BTC_EUR = BTC.get_ohlc_data_crypto(sd="2023-09-14", ed="2023-10-20", frequency_interval="weekly", from_symbol="BTC", to_symbol="EUR")
    # print(data_BTC_EUR)
    # BTC.save_ohlc_data_crypto(from_symbol="BTC", to_symbol="EUR", sd="2023-09-14", ed="2023-10-25", frequency_interval="daily", file_name="BTC_EUR_20231014")

    # btc_rate = USD.get_latest_exchange_rate(from_currency="BTC", to_currency="EUR")
    # btc_r = USD.get_latest_exchange_rate_int(from_currency="BTC", to_currency="EUR")
    # print(btc_rate)
    # print(btc_r)

    # BRENT = CommodityAsset('BRENT')
    # historical_data_Brent = BRENT.get_historical_price_data(sd="2023-10-15", ed="2023-11-05", frequency_interval="daily", api_source=default_api_name)
    # print (historical_data_Brent)
    # BRENT.save_historical_price_data(sd="2023-10-15", ed="2023-11-05", frequency_interval="daily", api_source=default_api_name, file_name="BRENT_Price_202310")

    # sd = "2023.10.01"
    # ed = "2023.11.01"

    # sd = datetime.strptime(sd, "%Y.%m.%d")
    # ed = datetime.strptime(ed, "%Y.%m.%d")

    # delta=ed-sd
    # print(str(delta.days)+" D")
    import dateutil.parser
    yourdate = dateutil.parser.parse("2023.11.20")
    print(yourdate)

    MS = EquityAsset('MS')
    data = MS.get_ohlcv_data(sd="2023.11.01", ed="2023.11.20", frequency_interval="weekly", exchange='SMART', currency='USD',
                             api_source="InteractiveBrokersAPI", data_adjusted='false', extended_hours='false', output_size='full')
    print(data)

    BTC = CurrencyDigitalAsset('BTC')
    data = BTC.get_ohlcv_data(sd="2023.11.01", ed="2023.11.20", frequency_interval="weekly", exchange='SMART', currency='USD',
                             api_source="InteractiveBrokersAPI", data_adjusted='false', extended_hours='false', output_size='full')


    # Next steps in API library:
    # - Apply crypto ohlcv to InteractiveBrokersAPI
    # - Apply fx ohlcv to APIVantageAPI + InteractiveBrokersAPI
    # - Change methods so they can be accessed with the same name for every class (see IB library for example)

    # Extra steps in API library:
    # - Improve asset creation by copying ib_insync library:  each asset class is created from asset parent class
    # - Add extra data AlphaVantage (sentimental, fundamental, economic indicators, technical indicators)
