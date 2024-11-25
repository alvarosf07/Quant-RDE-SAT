from qrde_sat.qrde1_data import *
from qrde_sat.qrde2_research import *
from qrde_sat.qrde3_models import *
from qrde_sat.qrde4_strategies import *
from qrde_sat.qrde5_backtesting import *


MS = EquityAsset('MS')
GS = EquityAsset('GS')
JPM = EquityAsset('JPM')

dataMS = MS.get_ohlcv_data(sd="2023-09-14", ed="2023.10.20", frequency_interval="weekly", api_source="AlphaVantageAPI", data_adjusted="false", extended_hours="false")
dataGS = GS.get_ohlcv_data("2023-10-14", "2023.10.20", "daily")
dataJPM = JPM.get_ohlcv_data("2023-10-14", "2023.10.20", "daily")

print(dataMS)
print(dataGS)
print(dataJPM)

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
dataMS = MS.get_ohlcv_data(sd="2023.11.01", ed="2023.11.20", frequency_interval="weekly", exchange='SMART', currency='USD',
                            api_source="InteractiveBrokersAPI", data_adjusted='false', extended_hours='false', output_size='full')
print(dataMS)

BTC = CurrencyDigitalAsset('BTC')
dataBTC = BTC.get_ohlcv_data(sd="2023.11.01", ed="2023.11.20", frequency_interval="weekly", exchange='SMART', currency='USD',
                            api_source="InteractiveBrokersAPI", data_adjusted='false', extended_hours='false', output_size='full')


# Next steps in API library:
# - Apply crypto ohlcv to InteractiveBrokersAPI
# - Apply fx ohlcv to APIVantageAPI + InteractiveBrokersAPI
# - Change methods so they can be accessed with the same name for every class (see IB library for example)

# Extra steps in API library:
# - Improve asset creation by overwriting ib_insync library:  each asset class is created from asset parent class
# - Add extra data AlphaVantage (sentimental, fundamental, economic indicators, technical indicators)
