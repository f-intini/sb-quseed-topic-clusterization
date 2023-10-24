# Description: This file contains the functions that pull data from various sources.

from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Iterable, Optional, Union, Dict, Tuple, List, Optional

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import numpy as np

from sb_py_fw.quseed.data import read_sql

from utils import get_elasticsearch_client
from utils.const import ALLOWED_DBS as allowed_dbs

# utils -----------------------------------------------------------------------------
empty_df: callable = lambda: pd.DataFrame([])


def parse_datetime(date: str) -> Optional[datetime]:
    """
    Parses a datetime string.

    Args:
        date (str): datetime string

    Raises:
        e: if the datetime string cannot be parsed

    Returns:
        Optional[datetime]: parsed datetime
    """
    if isinstance(date, list):
        date = date[0]
    try:
        return datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    except Exception as e:
        raise e


# elasticsearch stuff ---------------------------------------------------------------


def pull_daily_documents(day: datetime, es_index: str = "documents") -> pd.DataFrame:
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
                "date_created": {
                    "gte": day.strftime("%Y-%m-%d"),
                    "lt": (day + relativedelta(days=1)).strftime("%Y-%m-%d"),
                }
            },
        },
    }
    try:
        es_scan: Iterable[Optional[dict]] = scan(
            client=ec,
            query=es_query,
            index=es_index,
            fields=[
                "url",
                "title",
                "categories",
                "date_created",
                "title_embedding",
                "content_embedding",
                "summary_embedding",
            ],
            scroll="2m",
            size=300,
        )
        return pd.DataFrame([doc["fields"] for doc in es_scan])
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


def read_sql_data(
    query: str,
    table_name: str,
    database_name: str = "quseed",
    output_type: str = "df",
    method: str = "pandas",
) -> Optional[Union[dict, pd.DataFrame]]:
    """
    Pulls the categories distances from the database.

    Args:
        query (str): Query to run
        table_name (str): Table name to pull from.
        database_name (str, optional): Database name to pull from. Defaults to "quseed".
        output_type (str, optional): Output type. Defaults to "df".

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

    table_data: pd.DataFrame = read_sql(query=query, section=actual_db, method=method)

    if table_data.shape[0] == 0:
        print(f"No data found in {table_name} in {database_name}")
        return None

    if output_type == "dict":
        return table_data.to_dict(orient="records")
    return table_data


# hybrid stuff ---------------------------------------------------------------


def pull_documents_from_elastic(
    doc_urls: Iterable[str],
    es_index: str = "documents",
) -> pd.DataFrame:
    """
    Pulls documents from Elasticsearch.

    Args:
        doc_urls (Iterable[str]): list of document urls to pull
        es_index (str, optional): Elasticsearch index to pull from. Defaults to "docs_content".

    Raises:
        ValueError: if Elasticsearch cannot be connected to

    Returns:
        pd.DataFrame: dataframe of documents
    """
    if not (ec := get_elasticsearch_client()):
        raise ValueError("Cannot connect to Elasticsearch.")

    es_query = {"query": {"terms": {"url.keyword": doc_urls}}}

    try:
        es_resp: Iterable[Optional[dict]] = scan(
            client=ec,
            query=es_query,
            index=es_index,
            fields=[
                "url",
                "title",
                "categories",
                "date_created",
                "title_embedding",
                "content_embedding",
                "summary_embedding",
            ],
            scroll="2m",
            size=300,
        )

        documents: pd.DataFrame = pd.DataFrame([doc["fields"] for doc in es_resp])
    except Exception as e:
        print(f"Error while pulling documents from Elasticsearch, index: {es_index}")
        print(e)
        return empty_df()

    if documents.shape[0] == 0:
        return empty_df()

    ec.close()

    return documents


def get_clusters_titles(
    cluster_ids: List[str],
    documents_ix_es: str,
    es_client: Elasticsearch,
) -> List[str]:
    """
    Returns the titles of the documents in a cluster.

    Args:
        cluster_ids (List[str]):
        documents_ix_es (str):
        es_client (Elasticsearch):

    Returns:
        List[str]:
    """
    clusters_urls = read_sql_data(
        query="""
        select url
        from {clusterized_documents_table}
        where
            cluster_id in ({cluster_ids})
        """.format(
            clusterized_documents_table="clusterized_docs",
            cluster_ids=", ".join([f"'{cluster_id}'" for cluster_id in cluster_ids]),
        ),
        table_name="clusterized_docs",
        database_name="quseed",
    )

    if clusters_urls is None:
        return []

    clusters_urls = clusters_urls.url.tolist()

    es_resp = scan(
        client=es_client,
        query={"query": {"terms": {"url.keyword": clusters_urls}}},
        index=documents_ix_es,
        fields=["url", "title"],
        size=500,
    )

    return [
        doc["fields"]["title"][0]
        if isinstance(doc["fields"]["title"], list)
        else doc["fields"]["title"]
        for doc in es_resp
    ]


def get_community_data(
    community_id: str,
    database_name: str,
    community_table_name: str,
    clusters_table_name: str,
    cluster_communities_table_name: str,
) -> Optional[Dict[str, any]]:
    community_query = """
    select *
    from {communities_table}
    where community_id = '{community_id}'
    """.format(
        communities_table=community_table_name,
        community_id=community_id,
    )

    community_data = read_sql_data(
        query=community_query,
        table_name=community_table_name,
        database_name=database_name,
    )

    if community_data is None:
        return None

    community_data = community_data.to_dict(orient="records")[0]

    community_clusters_query: str = """
    select cluster_id
    from {cluster_communities_table}
    where
        community_id = '{community_id}'
    """.format(
        cluster_communities_table=cluster_communities_table_name,
        community_id=community_id,
    )
    community_clusters = read_sql_data(
        query=community_clusters_query,
        table_name=cluster_communities_table_name,
        database_name=database_name,
    )

    if community_clusters is None:
        return None

    community_clusters = community_clusters.cluster_id.tolist()

    clusters_centroids_query: str = """
    select centroid_embedding
    from {clusters_table}
    where
        cluster_id in ({cluster_ids})
    """.format(
        clusters_table=clusters_table_name,
        cluster_ids=", ".join([f"'{cluster_id}'" for cluster_id in community_clusters]),
    )
    clusters_centroids = read_sql_data(
        query=clusters_centroids_query,
        table_name=clusters_table_name,
        database_name=database_name,
    )

    if clusters_centroids is None:
        return None

    clusters_centroids = clusters_centroids.centroid_embedding.tolist()
    community_centroid = np.mean(clusters_centroids, axis=0)

    return {
        **community_data,
        "centroid": community_centroid,
    }
