# Description: Database utilities.

from typing import Optional, Dict

from elasticsearch import Elasticsearch
from weaviate import Client, AuthApiKey

from sb_py_fw.database.elastic import ElasticConnection
from sb_py_fw.utils.config import read_conf


def get_elasticsearch_client() -> Optional[Elasticsearch]:
    """Returns an Elasticsearch client object."""
    try:
        ElasticConnection().es
    except Exception as e:
        print(e)
        return None


# weaviate stuff ---------------------------------------------------------------
class WeaviateClient:
    # TODO: improve with class logger
    section: str = "weaviate_wcs"

    def __init__(self, dynamic: bool = True) -> None:
        self.dynamic: bool = dynamic
        self.__read_config__()
        pass

    def __read_config__(self) -> None:
        weaviate_conf: Dict[str, any] = read_conf(section=self.section)

        for k, v in weaviate_conf.items():
            setattr(self, k.lower(), v)

    def get_client(self) -> Optional[Client]:
        auth_key = AuthApiKey(api_key=self.api_key)

        try:
            client = Client(url=self.host, auth_client_secret=auth_key)
            if self.dynamic:
                setattr(self, "client", client)
            else:
                return client
        except Exception as e:
            print(e)
            return None
