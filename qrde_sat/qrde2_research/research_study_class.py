### Import Utils
import sys, os
import glob, pprint, logging
import numpy as np, pandas as pd
import talib as tb
from datetime import datetime
from logging.handlers import RotatingFileHandler

### Import plotting tools
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns # pip install seaborn
import mplfinance as mpf # pip install --upgrade mplfinance
from matplotlib import style
from scipy.stats import norm, laplace, johnsonsu
from scipy.stats import normaltest
from statsmodels.graphics.gofplots import qqplot
from mplfinance.original_flavor import candlestick_ohlc 
from lib.mlfinlab.data_structures import standard_data_structures
#from mlfinlab.data_structures import get_ema_dollar_imbalance_bars, get_ema_tick_imbalance_bars, get_ema_volume_imbalance_bars
#from mlfinlab.data_structures import get_ema_dollar_run_bars, get_ema_tick_run_bars, get_ema_volume_run_bars


### Local Imports
from ..qrde2_research.asset_class import *
from ..qrde2_research.portfolio_class import *


### Set the style for the plots.
style.use('dark_background')


##################### LOG CONFIGURATION ###################
LOG_FORMAT = '%(levelname)s %(asctime)s - %(message)s'
logging.basicConfig(filename=f'./SystemComponent.log', 
                            level=logging.INFO,
                            format=LOG_FORMAT,
                            filemode='w')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.addHandler(RotatingFileHandler('./SystemComponent.log', mode='a', maxBytes=5*1024*1024, 
                                 backupCount=2, encoding=None, delay=0))
###########################################################

class ResearchStudy(Portfolio):

    '''
    Formulates an hypothesis and tries to confirm in on a Portfolio object.
    Specifically, it will take some data from that portfolio, makes some calculations and returns a result
    '''

    def __init__(self, assetsList, formOrRead, 
                 sampleFormat='', 
                 dateHourString='', 
                 readingPath='',
                 saveTheData='',
                 formerOrNew=''):

        # We will form the portofolio > depending on if we want to read or request the data.
        # The generated data will be in PORTFOLIO._portfolioDict:

        if formOrRead == 'form':

            self.PORTFOLIO = Portfolio(assetsList)
            self.PORTFOLIO._formPortfolioHistoricalData(sampleFormat)

        elif formOrRead == 'read':

            self.PORTFOLIO = Portfolio(assetsList)
            self.PORTFOLIO._readPortfolioHistoricalData(dateHourString)

        elif formOrRead == 'read_features':

            self.PORTFOLIO = Portfolio(assetsList)
            self.PORTFOLIO._readPortfolioFeaturesCsvs()

        elif formOrRead == 'form_darwin':

            self.PORTFOLIO = Portfolio(assetsList)
            self.PORTFOLIO._formPortfolioDARWINHistoricalData(saveTheData)

        elif formOrRead == 'read_darwin':

            self.PORTFOLIO = Portfolio(assetsList)
            self.PORTFOLIO._readPortfolioDarwinFeaturesCsvs(formerOrNew)

    def _saveGeneratedDataFrames(self, saveDirectory):

        # Save each dataframe:
        for eachAssetName, eachAssetDataFrame in self.PORTFOLIO._portfolioDict.items():

            logger.warning(f'[{self._saveGeneratedDataFrames.__name__}] - Looping for asset <{eachAssetName}>...')
            eachAssetDataFrame.to_csv(saveDirectory + f'/{eachAssetName}_DF.csv')

    def _saveGeneratedDataFramesOtherBars(self, saveDirectory):

        # Save each dataframe:
        for eachAssetName, eachAssetDataFrame in self.ALTERNATIVE_BARS.items():

            logger.warning(f'[{self._saveGeneratedDataFramesOtherBars.__name__}] - Looping for asset <{eachAssetName}>...')
            eachAssetDataFrame.to_csv(saveDirectory + f'/{eachAssetName}_Others_DF.csv')

    ######################### DARWIN ASSETS #########################

    def _generateResampledAndFilteredSeries(self, resampleRule):

        # Generates returns based on some representation of the data:
        for eachAssetName, eachAssetDataFrame in self.PORTFOLIO._portfolioDict.items():

            logger.warning(f'[{self._generateResampledAndFilteredSeries.__name__}] - Looping for asset <{eachAssetName}>...')

            # Resample it:
            eachAssetDataFrameResampled = eachAssetDataFrame.resample(rule=resampleRule).ohlc().dropna()
            # Change columns to get rid of the multiindex:
            eachAssetDataFrameResampled.columns = ['open', 'high', 'low', 'close']
            #eachAssetDataFrameResampled = eachAssetDataFrame.resample(rule='D').last().dropna()

            # Filter non-week days:
            eachAssetDataFrameResampled = eachAssetDataFrameResampled[eachAssetDataFrameResampled.index.dayofweek < 5]

            # Get it again:
            self.PORTFOLIO._portfolioDict[eachAssetName] = eachAssetDataFrameResampled
            print(self.PORTFOLIO._portfolioDict[eachAssetName].head())

    def _saveDarwinGeneratedDataFrames(self, saveDirectory):

        # Save each dataframe:
        for eachAssetName, eachAssetDataFrame in self.PORTFOLIO._portfolioDict.items():

            logger.warning(f'[{self._saveDarwinGeneratedDataFrames.__name__}] - Looping for asset <{eachAssetName}>...')
            eachAssetDataFrame.to_csv(saveDirectory + f'/{eachAssetName}_DF.csv')

    def _generateDARWINTickBars(self, threshold):

        # Generate tick bar representations:
        self.ALTERNATIVE_BARS = {}
        homeStr = os.path.expandvars('${HOME}')
        thresholdVariable = threshold

        # Loop for all the assets:
        for eachAssetName in self.PORTFOLIO._portfolioDict:

            # Tick Bars > We need to have ticks in the CSV no other form or aggregation.
            # The timestamp doesn't need to be as index > if it is as an index gives error.
            READ_PATH = f'{homeStr}/Desktop/quant-research-env/DARWINStrategyContentSeries/Data/{eachAssetName}_former_Quotes.csv'

            # Read the data:
            bars = pd.read_csv(READ_PATH, index_col=0, parse_dates=True, infer_datetime_format=True)

            # Generate the vol fake column and take the index out:
            bars['volume_fake'] = bars.quote * 100
            bars.reset_index(inplace=True)
            print(bars.head())

            # Get the suitable columns:
            bars = bars[['timestamp', 'quote', 'volume_fake']]

            # Generate the tick bars.
            bars = standard_data_structures.get_tick_bars(bars, threshold=thresholdVariable, batch_size=100000, verbose=False)

            # Get log returns for this bars:
            bars['Returns'] = np.log(bars.close/bars.close.shift(1))
            bars.dropna(how='any', inplace=True)
            print(f'TICK BARS for: {eachAssetName} >> Shape: {bars.shape}')
            print(bars.head())

            # Add them to the dict based on their symbol:
            self.ALTERNATIVE_BARS[eachAssetName] = bars

    def _generateDARWINDollarBars(self, threshold):

        # Generate dollar bar representations:
        self.ALTERNATIVE_BARS = {}
        homeStr = os.path.expandvars('${HOME}')
        thresholdVariable = threshold

        # Loop for all the assets:
        for eachAssetName in self.PORTFOLIO._portfolioDict:

            # Dollar Bars > We need to have ticks in the CSV no other form or aggregation.
            # The timestamp doesn't need to be as index > if it is as index gives error.
            READ_PATH = f'{homeStr}/Desktop/quant-research-env/DARWINStrategyContentSeries/Data/{eachAssetName}_former_Quotes.csv'

            # Read the data:
            bars = pd.read_csv(READ_PATH)

            # Read the data:
            bars = pd.read_csv(READ_PATH, index_col=0, parse_dates=True, infer_datetime_format=True)

            # Generate the vol fake column and take the index out:
            bars['volume_fake'] = bars.quote * 100
            bars.reset_index(inplace=True)
            print(bars.head())

            # Get the suitable columns:
            bars = bars[['timestamp', 'quote', 'volume_fake']]

            # Generate the tick bars.
            bars = standard_data_structures.get_dollar_bars(bars, threshold=thresholdVariable, batch_size=100000, verbose=True)

            # Get log returns for this bars:
            bars['Returns'] = np.log(bars.close/bars.close.shift(1))
            bars.dropna(how='any', inplace=True)
            print(f'DOLLAR BARS for: {eachAssetName} >> Shape: {bars.shape}')
            print(bars.head())

            # Add them to the dict based on their symbol:
            self.ALTERNATIVE_BARS[eachAssetName] = bars

    ######################### DARWIN ASSETS #########################

    ######################### RETURNS #########################

    def _generateLogReturns(self):

        # Generates returns based on some representation of the data:
        for eachAssetName, eachAssetDataFrame in self.PORTFOLIO._portfolioDict.items():

            logger.warning(f'[{self._generateLogReturns.__name__}] - Looping for asset <{eachAssetName}>...')

            # Generate the log returns and drop the empty data points:
            eachAssetDataFrame['Returns'] = np.log(eachAssetDataFrame.close/eachAssetDataFrame.close.shift(1))
            eachAssetDataFrame.dropna(how='any', inplace=True)

    def _generateRawReturns(self):

        # Generates returns based on some representation of the data:
        for eachAssetName, eachAssetDataFrame in self.PORTFOLIO._portfolioDict.items():

            logger.warning(f'[{self._generateRawReturns.__name__}] - Looping for asset <{eachAssetName}>...')

            # Generate the raw returns and drop the empty data points:
            eachAssetDataFrame['Returns'] = eachAssetDataFrame.close.pct_change()
            eachAssetDataFrame.dropna(how='any', inplace=True)

    def _generateMidPrice(self):

        # Loop for all the assets:
        for eachAssetName, eachAssetDataFrame in self.PORTFOLIO._portfolioDict.items():

            # Generate the mid column price:
            eachAssetDataFrame[f'{eachAssetName}_mid_price'] = round((eachAssetDataFrame[f'{eachAssetName}_bid_price'] + eachAssetDataFrame[f'{eachAssetName}_ask_price'])/2, 5)

    def _generateRollingMean(self, rollingWindow=100):

        # Generate rolling mean and std based on some rolling window:
        for eachAssetName, eachAssetDataFrame in self.PORTFOLIO._portfolioDict.items():

            eachAssetDataFrame[f'{eachAssetName}_roll_mean'] = eachAssetDataFrame['Returns'].rolling(rollingWindow).mean()
            eachAssetDataFrame[f'{eachAssetName}_roll_std'] = eachAssetDataFrame['Returns'].rolling(rollingWindow).std()

    ######################### RETURNS #########################

    ######################### REPRESENTATIONS #########################

    def _generateTickBars(self, endDate, threshold):

        # Generate tick bar representations:
        self.ALTERNATIVE_BARS = {}
        homeStr = os.path.expandvars('${HOME}')
        thresholdVariable = threshold

        # Loop for all the assets:
        for eachAssetName in self.PORTFOLIO._portfolioDict:

            # Tick Bars > We need to have ticks in the CSV no other form or aggregation.
            # The timestamp doesn't need to be as index > if it is as an index gives error.
            READ_PATH = f'{homeStr}/Desktop/quant-research-env/RegimeAnalysisContentSeries/Data/Data_Ticks/{eachAssetName}_BID_ASK_{endDate}.csv'

            # Read the data:
            bars = pd.read_csv(READ_PATH)

            # Generate the mid column price:
            bars[f'{eachAssetName}_mid_price'] = round((bars[f'{eachAssetName}_bid_price'] + bars[f'{eachAssetName}_ask_price'])/2, 5)

            # Get the suitable columns:
            bars = bars[[f'{eachAssetName}_timestamp', f'{eachAssetName}_mid_price', f'{eachAssetName}_ask_size']]
            print(bars.head())

            # Generate the tick bars.
            bars = standard_data_structures.get_tick_bars(bars, threshold=thresholdVariable, batch_size=100000, verbose=False)

            # Get log returns for this bars:
            bars['Returns'] = np.log(bars.close/bars.close.shift(1))
            bars.dropna(how='any', inplace=True)
            print(f'TICK BARS for: {eachAssetName} >> Shape: {bars.shape}')
            print(bars.head())

            # Add them to the dict based on their symbol:
            self.ALTERNATIVE_BARS[eachAssetName] = bars

    def _generateDollarBars(self, endDate, threshold):

        # Generate dollar bar representations:
        self.ALTERNATIVE_BARS = {}
        homeStr = os.path.expandvars('${HOME}')
        thresholdVariable = threshold

        # Loop for all the assets:
        for eachAssetName in self.PORTFOLIO._portfolioDict:

            # Dollar Bars > We need to have ticks in the CSV no other form or aggregation.
            # The timestamp doesn't need to be as index > if it is as index gives error.
            READ_PATH = f'{homeStr}/Desktop/quant-research-env/RegimeAnalysisContentSeries/Data/Data_Ticks/{eachAssetName}_BID_ASK_{endDate}.csv'

            # Read the data:
            bars = pd.read_csv(READ_PATH)

            # Generate the mid column price:
            bars[f'{eachAssetName}_mid_price'] = round((bars[f'{eachAssetName}_bid_price'] + bars[f'{eachAssetName}_ask_price'])/2, 5)

            # Get the suitable columns:
            bars = bars[[f'{eachAssetName}_timestamp', f'{eachAssetName}_mid_price', f'{eachAssetName}_ask_size']]
            print(bars.head())

            # Generate the tick bars.
            bars = standard_data_structures.get_dollar_bars(bars, threshold=thresholdVariable, batch_size=100000, verbose=True)

            # Get log returns for this bars:
            bars['Returns'] = np.log(bars.close/bars.close.shift(1))
            bars.dropna(how='any', inplace=True)
            print(f'DOLLAR BARS for: {eachAssetName} >> Shape: {bars.shape}')
            print(bars.head())

            # Add them to the dict based on their symbol:
            self.ALTERNATIVE_BARS[eachAssetName] = bars

    ######################### REPRESENTATIONS #########################

    ######################### PLOTS #########################

    def _plotLine(self, saveDirectory='', showIt=False):

        # Plot the returns of each asset in the portfolio:
        for eachAssetName, eachAssetDataFrame in self.PORTFOLIO._portfolioDict.items():

            logger.warning(f'[{self._plotReturns.__name__}] - Looping for asset <{eachAssetName}>...')

            # Plot the returns:
            f1, ax = plt.subplots(figsize = (10,5))
            f1.canvas.set_window_title('Close Plot')
            plt.plot(eachAssetDataFrame.close, label='Close')
            plt.grid(linestyle='dotted')
            plt.xlabel('Observations')
            plt.ylabel('Quote')
            plt.title(f'Asset: {eachAssetName} -- Close Plot')
            plt.legend(loc='best')
            plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)

            # In PNG:
            plt.savefig(saveDirectory + f'/closePlot_{eachAssetName}.png')

            # Show it:
            if showIt: 
                plt.show()

    def _plotCandleAndIndicators(self, saveDirectory='', showIt=False):

        # Plot the candles of each asset in the portfolio:
        for eachAssetName, eachAssetDataFrame in self.PORTFOLIO._portfolioDict.items():

            logger.warning(f'[{self._plotCandleAndIndicators.__name__}] - Looping for asset <{eachAssetName}>...')

            # Plot the candles and indicators:
            f1, ax = plt.subplots(figsize = (10,5))
            f1.canvas.set_window_title('Candle and Indicators Plot')

            # Get the candles and then indicators:
            #self.startBarNumber = startBarNumber
            #self.endBarNumber = endBarNumber
            #OHLC = eachAssetDataFrame.iloc[self.startBarNumber:self.endBarNumber,:]
            #eachAssetDataFrame.reset_index(inplace=True)  
            eachAssetDataFrame['date'] =  range(1, len(eachAssetDataFrame) + 1)
            candlestick_ohlc(ax, eachAssetDataFrame[['date', 'open', 'high', 'low', 'close']].values, width=.6, colorup='green', colordown='red', alpha=1)

            ### Plot the indicators:
            self._plotIndicators(eachAssetDataFrame)

            plt.grid(linestyle='dotted')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f'Asset: {eachAssetName} -- Candle and Indicators plot')
            plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)

            # In PNG:
            plt.savefig(saveDirectory + f'/candlePlot_{eachAssetName}.png')

            # Show it:
            if showIt: 
                plt.show()

    def _plotCandleAndIndicatorsOtherBars(self, saveDirectory='', showIt=False):

        # Plot the candles of each asset in the portfolio:
        for eachAssetName, eachAssetDataFrame in self.ALTERNATIVE_BARS.items():

            logger.warning(f'[{self._plotCandleAndIndicatorsOtherBars.__name__}] - Looping for asset <{eachAssetName}>...')

            # Plot the candles and indicators:
            f1, ax = plt.subplots(figsize = (10,5))
            f1.canvas.set_window_title('Candle and Indicators Plot')

            # Get the candles and then indicators:
            #self.startBarNumber = startBarNumber
            #self.endBarNumber = endBarNumber
            #OHLC = eachAssetDataFrame.iloc[self.startBarNumber:self.endBarNumber,:]
            #eachAssetDataFrame.reset_index(inplace=True)  
            eachAssetDataFrame['date'] =  range(1, len(eachAssetDataFrame) + 1)
            candlestick_ohlc(ax, eachAssetDataFrame[['date', 'open', 'high', 'low', 'close']].values, width=.6, colorup='green', colordown='red', alpha=1)

            ### Plot the indicators:
            self._plotIndicators(eachAssetDataFrame)

            plt.grid(linestyle='dotted')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f'Asset: {eachAssetName} -- Candle and Indicators plot')
            plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)

            # In PNG:
            plt.savefig(saveDirectory + f'/candlePlot_{eachAssetName}.png')

            # Show it:
            if showIt: 
                plt.show()

    def _plotIndicators(self, dataFrameToPlot):

        ##################### Plot BBands #####################
        dataFrameToPlot['uperBBand'], dataFrameToPlot['middleBBand'], dataFrameToPlot['lowerBBand'] = tb.BBANDS(dataFrameToPlot.close.values, 
                                                                                                                timeperiod=15, 
                                                                                                                nbdevup=2.5, 
                                                                                                                nbdevdn=2.5, 
                                                                                                                matype=0)

        ### Then plot it:
        plt.plot(dataFrameToPlot.date, dataFrameToPlot.uperBBand, linewidth=2.5)
        plt.plot(dataFrameToPlot.date, dataFrameToPlot.middleBBand, linewidth=2.5)
        plt.plot(dataFrameToPlot.date, dataFrameToPlot.lowerBBand, linewidth=2.5)
        #plt.fill_between(dataFrameToPlot.datePopBar, y1=dataFrameToPlot.lowerBBand, y2=dataFrameToPlot.uperBBand, color='#adccff', alpha='0.2')
        ##################### Plot BBands #####################

    def _plotCandleAndIndicatorsNEW(self, saveDirectory='', showIt=False, plotType='candle'):

        # Plot the candles of each asset in the portfolio:
        for eachAssetName, eachAssetDataFrame in self.PORTFOLIO._portfolioDict.items():

            logger.warning(f'[{self._plotCandleAndIndicators.__name__}] - Looping for asset <{eachAssetName}>...')

            # Plot:
            mpf.plot(eachAssetDataFrame, type=plotType)

            # In PNG:
            plt.savefig(saveDirectory + f'/candlePlotNEW_{eachAssetName}.png')

            # Show it:
            if showIt: 
                plt.show()

    def _plotReturns(self, saveDirectory='', showIt=False, rollingMeanOrNot=False):

        # Plot the returns of each asset in the portfolio:
        for eachAssetName, eachAssetDataFrame in self.PORTFOLIO._portfolioDict.items():

            logger.warning(f'[{self._plotReturns.__name__}] - Looping for asset <{eachAssetName}>...')

            # Plot the returns:
            f1, ax = plt.subplots(figsize = (10,5))
            f1.canvas.set_window_title('Returns Plot')
            plt.plot(eachAssetDataFrame.Returns.values, label='Returns')
            if rollingMeanOrNot:
                plt.plot(eachAssetDataFrame[f'{eachAssetName}_roll_mean'].values, label='RollingMean', linewidth=3.0)
                plt.plot(eachAssetDataFrame[f'{eachAssetName}_roll_std'].values, label='RollingStd', linewidth=3.0)
            plt.grid(linestyle='dotted')
            plt.xlabel('Observations')
            plt.ylabel('Returns')
            plt.title(f'Asset: {eachAssetName} -- Returns Plot (First difference)')
            plt.legend(loc='best')
            plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)

            # In PNG:
            plt.savefig(saveDirectory + f'/returnsPlot_{eachAssetName}.png')

            # Show it:
            if showIt: 
                plt.show()

    def _plotReturnsOtherBars(self, saveDirectory='', showIt=False, rollingMeanOrNot=False):

        # Plot the returns of each asset in the portfolio:
        for eachAssetName, eachAssetDataFrame in self.ALTERNATIVE_BARS.items():

            logger.warning(f'[{self._plotReturnsOtherBars.__name__}] - Looping for asset <{eachAssetName}>...')

            # Plot the returns:
            f1, ax = plt.subplots(figsize = (10,5))
            f1.canvas.set_window_title('Returns Plot')
            plt.plot(eachAssetDataFrame.Returns.values, label='Returns')
            if rollingMeanOrNot:
                plt.plot(eachAssetDataFrame[f'{eachAssetName}_roll_mean'].values, label='RollingMean', linewidth=3.0)
                plt.plot(eachAssetDataFrame[f'{eachAssetName}_roll_std'].values, label='RollingStd', linewidth=3.0)
            plt.grid(linestyle='dotted')
            plt.xlabel('Observations')
            plt.ylabel('Returns')
            plt.title(f'Asset: {eachAssetName} -- Returns Plot (First difference)')
            plt.legend(loc='best')
            plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)

            # In PNG:
            plt.savefig(saveDirectory + f'/returnsPlot_{eachAssetName}.png')

            # Show it:
            if showIt: 
                plt.show()

    def _plotDistribution(self, saveDirectory='', showIt=False):

        # Plot the returns of each asset in the portfolio:
        for eachAssetName, eachAssetDataFrame in self.PORTFOLIO._portfolioDict.items():

            logger.warning(f'[{self._plotDistribution.__name__}] - Looping for asset <{eachAssetName}>...')

            # Plot the distribution and KDE:
            f1, ax = plt.subplots(figsize = (10,5))
            f1.canvas.set_window_title('Distribution Plot')
            sns.distplot(eachAssetDataFrame.Returns.values, color="dodgerblue", label=f'Return Distribution', fit=norm, 
                    hist_kws={"rwidth":0.90,'edgecolor':'white', 'alpha':1.0},
                    fit_kws={"color":"coral", 'linewidth':2.5, 'label':'Fit (Normal) Line'},
                    kde_kws={"color":"limegreen", 'linewidth':2.5, 'label':'KDE Line'})
            sns.distplot(eachAssetDataFrame.Returns.values, color="dodgerblue", fit=laplace, 
                    hist_kws={"rwidth":0.90,'edgecolor':'white', 'alpha':1.0},
                    fit_kws={"color":"gold", 'linestyle':'solid', 'linewidth':2.5, 'label':'Fit (Laplace) Line'})
            sns.distplot(eachAssetDataFrame.Returns.values, color="dodgerblue", fit=johnsonsu, 
                    hist_kws={"rwidth":0.90,'edgecolor':'white', 'alpha':1.0},
                    fit_kws={"color":"darkviolet", 'linestyle':'solid', 'linewidth':2.5, 'label':'Fit (Johnson) Line'})

            # Add more than one distplot will add several KDE lines to see the different distributions.
            plt.grid(linestyle='dotted')
            plt.xlabel(f'Returns Values', horizontalalignment='center', verticalalignment='center', fontsize=14, labelpad=20)
            plt.title(f'Asset: {eachAssetName} -- Distribution Returns and KDE Plot > Skew: {round(eachAssetDataFrame.Returns.skew(),3)} // Kurtosis: {round(eachAssetDataFrame.Returns.kurtosis(),3)}')
            plt.legend(loc='best')
            plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)

            # Do test:
            statistic, p_value = normaltest(eachAssetDataFrame.Returns.values)
            print(f'Asset: {eachAssetName} -- Statistics: {statistic} // p-value: {p_value}')
            alpha = 0.05
            if p_value > alpha: 
                print(f'P-value is GREATER than alpha ({alpha}) >> Fail to reject H0 (Sample seems Normal)')
            else:
                print(f'P-value is LESS/EQUAL than alpha ({alpha}) >> Reject H0 (Sample does NOT look Normal)')

            # In PNG:
            plt.savefig(saveDirectory + f'/distributionPlot_{eachAssetName}.png')

            # Show it:
            if showIt:
                plt.show()

    def _plotDistributionOtherBars(self, saveDirectory='', showIt=False):

        # Plot the distribution of each asset in the portfolio:
        for eachAssetName, eachAssetDataFrame in self.ALTERNATIVE_BARS.items():

            logger.warning(f'[{self._plotDistributionOtherBars.__name__}] - Looping for asset <{eachAssetName}>...')

            # Plot the distribution and KDE:
            f1, ax = plt.subplots(figsize = (10,5))
            f1.canvas.set_window_title('Distribution Plot')
            sns.distplot(eachAssetDataFrame.Returns.values, color="dodgerblue", label=f'Return Distribution', fit=norm, 
                    hist_kws={"rwidth":0.90,'edgecolor':'white', 'alpha':1.0},
                    fit_kws={"color":"coral", 'linewidth':2.5, 'label':'Fit (Normal) Line'},
                    kde_kws={"color":"limegreen", 'linewidth':2.5, 'label':'KDE Line'})
            sns.distplot(eachAssetDataFrame.Returns.values, color="dodgerblue", fit=laplace, 
                    hist_kws={"rwidth":0.90,'edgecolor':'white', 'alpha':1.0},
                    fit_kws={"color":"gold", 'linestyle':'solid', 'linewidth':2.5, 'label':'Fit (Laplace) Line'})
            sns.distplot(eachAssetDataFrame.Returns.values, color="dodgerblue", fit=johnsonsu, 
                    hist_kws={"rwidth":0.90,'edgecolor':'white', 'alpha':1.0},
                    fit_kws={"color":"darkviolet", 'linestyle':'solid', 'linewidth':2.5, 'label':'Fit (Johnson) Line'})

            # Add more than one distplot will add several KDE lines to see the different distributions.
            plt.grid(linestyle='dotted')
            plt.xlabel(f'Returns Values', horizontalalignment='center', verticalalignment='center', fontsize=14, labelpad=20)
            plt.title(f'Asset: {eachAssetName} -- Distribution Returns and KDE Plot > Skew: {round(eachAssetDataFrame.Returns.skew(),3)} // Kurtosis: {round(eachAssetDataFrame.Returns.kurtosis(),3)}')
            plt.legend(loc='best')
            plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)

            # Do test:
            statistic, p_value = normaltest(eachAssetDataFrame.Returns.values)
            print(f'Asset: {eachAssetName} -- Statistics: {statistic} // p-value: {p_value}')
            alpha = 0.05
            if p_value > alpha: 
                print(f'P-value is GREATER than alpha ({alpha}) >> Fail to reject H0 (Sample seems Normal)')
            else:
                print(f'P-value is LESS/EQUAL than alpha ({alpha}) >> Reject H0 (Sample does NOT look Normal)')

            # In PNG:
            plt.savefig(saveDirectory + f'/distributionPlot_{eachAssetName}.png')

            # Show it:
            if showIt:
                plt.show()

    def _plotQQPlot(self, saveDirectory='', showIt=False):

        # Plot the quantile plot of each asset in the portfolio:
        for eachAssetName, eachAssetDataFrame in self.PORTFOLIO._portfolioDict.items():

            logger.warning(f'[{self._plotQQPlot.__name__}] - Looping for asset <{eachAssetName}>...')

            # Plot the QQplot:
            qqplot(eachAssetDataFrame.Returns.values, line='s')

            # Add more variables:
            plt.grid(linestyle='dotted')
            plt.xlabel('Theoretical Quantiles', horizontalalignment='center', verticalalignment='center', fontsize=14, labelpad=20)
            plt.ylabel('Sample Quantiles', horizontalalignment='center', verticalalignment='center', fontsize=14, labelpad=20)
            plt.title(f'Asset: {eachAssetName} -- Quantile-Quantile (QQ) Plot')
            plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)

            # In PNG:
            plt.savefig(saveDirectory + f'/QQPlot_{eachAssetName}.png')

            # Show it:
            if showIt:
                plt.show()

    def _plotQQPlotOtherBars(self, saveDirectory='', showIt=False):

        # Plot the quantile plot of each asset in the portfolio:
        for eachAssetName, eachAssetDataFrame in self.ALTERNATIVE_BARS.items():

            logger.warning(f'[{self._plotQQPlotOtherBars.__name__}] - Looping for asset <{eachAssetName}>...')

            # Plot the QQplot:
            qqplot(eachAssetDataFrame.Returns.values, line='s')

            # Add more variables:
            plt.grid(linestyle='dotted')
            plt.xlabel('Theoretical Quantiles', horizontalalignment='center', verticalalignment='center', fontsize=14, labelpad=20)
            plt.ylabel('Sample Quantiles', horizontalalignment='center', verticalalignment='center', fontsize=14, labelpad=20)
            plt.title(f'Asset: {eachAssetName} -- Quantile-Quantile (QQ) Plot')
            plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)

            # In PNG:
            plt.savefig(saveDirectory + f'/QQPlot_{eachAssetName}.png')

            # Show it:
            if showIt:
                plt.show()

    ######################### PLOTS #########################