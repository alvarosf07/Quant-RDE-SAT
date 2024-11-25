# Import Utils
import sys, os
import glob
import pandas as pd


# Local Imports
from ..qrde2_research import *


class BaseModel(ResearchStudy):

    '''
        A Model will hold the data entry for the portfolio + create the decision making that strategy will use.

        - A Model will get some input data (i.e. X, attributes, random variables).
        - A Model will need to define its parameters.
        - A Model will need to be fitted to the data.
        - A Model will need to output a result after performing the previous steps.
    '''

    def __init__(self, name, assetsList, formOrRead='read_features', dateHourString=''):

        # Initialize the ResearchStudy class:
        super().__init__(assetsList, formOrRead, dateHourString)

        # Create the name of the model:
        self.name = name

    def _defineModelParameters(self):

        pass

    def _inputVariables(self, inputVars):

        pass
    
    def _outputVariable(self):

        pass

    def _fitTheModel(self):

        pass

    def _saveModel(self):

        pass    