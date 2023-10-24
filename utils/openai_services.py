# Description: utilities needed to work with openai services.

import openai

from sb_py_fw.database.connections import read_conf


def setup_openai():
    conf = read_conf(section="openai_token")
    openai.api_key = conf["api_token"]
