# Import Utils
import sys, os
import numpy as np
import pandas as pd
import logging, pickle
from hmmlearn.hmm import GaussianHMM, GMMHMM   # Import the model package > pip install hmmlearn
logger = logging.getLogger()

# Import Plotting Tools
import matplotlib.pyplot as plt
from matplotlib import style, cm
style.use('dark_background')

# Local Imports
from ...qrde2_research import *
from ...qrde3_models import BaseModel

# To Do
class DeepMarkovModel(BaseModel):
    def __init__(self):
        # Create some assets:
        assetsList = [Asset('WS30', 'traditional', 'historical'), # Index US
                    Asset('XAUUSD', 'traditional', 'historical'), # Commodity
                    Asset('GDAXIm', 'traditional', 'historical'), # Index EUR
                    Asset('EURUSD', 'traditional', 'historical'), # Major
                    Asset('GBPJPY', 'traditional', 'historical')] # Minor

        # Initialize the ResearchStudy class:
        super().__init__('HiddenMarkovModel', assetsList)

        # Make a random seed to reproduce results:
        np.random.seed(33)

        # Print to see if working:
        #logger.warning(self.PORTFOLIO._portfolioDict['WS30'])

    def _defineModelParameters(self):
        # Define the model:
        #self.model = GaussianHMM(n_components=2, 
        #                         covariance_type="full", 
        #                         n_iter=200,
        #                         verbose=True)
        self.model = GMMHMM(n_components=2, 
                                covariance_type="full", 
                                n_iter=20,
                                verbose=True)