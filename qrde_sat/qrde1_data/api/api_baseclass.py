# Import utils:

# Import Internal Classes:
from ....config.config_data.api_credentials import *

class APIBaseClass(object):
    """
    General Parent class for every API source.

    """

    def __init__(self):
        self.api_credentials = "default_Credential"

        self.__user = credentials_APIBaseClass["username"]
        self.__password = credentials_APIBaseClass["password"]
        self.__host = credentials_APIBaseClass["host"]
        self.__base_url = credentials_APIBaseClass["base_url"]
        self.__port = credentials_APIBaseClass["port"]

    @classmethod
    def get_api_sources(cls):
        api_sources = [a.__name__ for a in cls.__subclasses__()]
        return api_sources

# Improvements:
#    - Include method with info about all the symbols available, the asset classes, the functions for each....
#      (this will source and centralize info from all the available APIs)
