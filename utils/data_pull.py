# Description: This file contains the functions that pull data from various sources.

from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Iterable, Optional, Union, Dict, Tuple

import pandas as pd
from elasticsearch.helpers import scan

from sb_py_fw.quseed.data import read_sql

from utils import get_elasticsearch_client
from utils.const import ALLOWED_DBS as allowed_dbs

# elasticsearch stuff ---------------------------------------------------------------


def pull_daily_documents(day: datetime, es_index: str = "docs_content") -> pd.DataFrame:
    """
    Pulls all documents from Elasticsearch for a given day.

    Args:
        day (datetime): day to pull documents for
        es_index (str, optional): Elasticsearch index to pull from. Defaults to "docs_content".

    Returns:
        pd.DataFrame: dataframe of documents
    """
    if not (ec := get_elasticsearch_client()):
        return pd.DataFrame([])

    es_query = {
        "query": {
            "range": {
                "timestamp": {
                    "gte": day.strftime("%Y-%m-%d"),
                    "lt": (day + relativedelta(days=1)).strftime("%Y-%m-%d"),
                }
            }
        }
    }
    try:
        es_scan: Iterable[Optional[dict]] = scan(
            client=ec,
            query=es_query,
            index=es_index,
            scroll="2m",
            size=300,
        )
        return pd.DataFrame([doc["_source"] for doc in es_scan])
    except Exception as e:
        print(e)
        return pd.DataFrame([])


# postgres stuff ---------------------------------------------------------------


def escape_postgres_string(string: str) -> str:
    """
    Escapes a string for PostgreSQL.

    Args:
        string (str): string to escape

    Returns:
        str: escaped string
    """
    return (
        string.replace("'", "''")
        .replace('"', '""')
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("\b", "\\b")
        .replace("\f", "\\f")
        .replace("\v", "\\v")
        .replace("\a", "\\a")
        .replace("\000", "\\000")
    )


def pull_categories_distances(
    database_name: str = "quseed",
    table_name: str = "categories_distances",
    output_type: str = "dict",
) -> Optional[Union[dict, pd.DataFrame]]:
    """
    Pulls the categories distances from the database.

    Args:
        database_name (str, optional): Database name to pull from. Defaults to "quseed".
        table_name (str, optional): Table name to pull from. Defaults to "categories_distances".
        output_type (str, optional): Output type. Defaults to "dict".

    Raises:
        ValueError: <database_name> must be one of <allowed_dbs.keys()>
        ValueError: <output_type> must be one of ["dict", "df"]
        ValueError: Table <table_name> does not exist in <database_name>

    Returns:
        Optional[Union[dict, pd.DataFrame]]: categories distances, if found, in the selected format
    """
    database_name = escape_postgres_string(database_name)
    table_name = escape_postgres_string(table_name)

    if not (actual_db := allowed_dbs.get(database_name, None)):
        raise ValueError(
            f"<database_name> must be one of {allowed_dbs.keys()}, not {database_name}"
        )

    if output_type not in ["dict", "df"]:
        raise ValueError(
            f'<output_type> must be one of ["dict", "df"], not {output_type}'
        )

    read_query: str = "select * from {table_name}".format(table_name=table_name)
    existing_table_query: str = "select exists(select * from information_schema.tables where table_name='{table_name}')".format(
        table_name=table_name
    )
    try:
        if not (
            table_exists := read_sql(
                query=existing_table_query, section=actual_db
            ).iloc[
                0, 0
            ]  # walrus operator needed because pandas is not thread safe (yet, orco! - @f-intini)
        ):
            raise ValueError(f"Table {table_name} does not exist in {database_name}")
    except Exception as e:
        print(
            "Error while checking table: {table_name} existance in {database_name}".format(
                table_name=table_name, database_name=database_name
            )
        )
        print(e)
        return None

    table_data: pd.DataFrame = read_sql(query=read_query, section=actual_db)

    if table_data.shape[0] == 0:
        print(f"No data found in {table_name} in {database_name}")
        return None

    if output_type == "dict":
        new_output: Dict[Tuple[str, str], float] = {}
        for _, row in table_data.iterrows():
            new_output[(row["first_category"], row["second_category"])] = row[
                "distance"
            ]
        return new_output
    return table_data
