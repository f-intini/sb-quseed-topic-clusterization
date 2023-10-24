# NOTE: postponed evaluation of annotations
import sys
import copy
from time import perf_counter
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union, Dict, Tuple, Set
from itertools import chain
from collections import Counter
import uuid

import numpy as np
from loguru import logger
import pandas as pd

from .clusterization_utils import (
    append_documents_to_existing_clusters,
    generate_cluster_data,
    pick_best_clustering,
)
from .dbscan_utils import dbscan_clustering, split_big_clusters
from .hdbscan_utils import outlier_clusters_splitting

from utils import (
    generic_date,
    pull_daily_documents,
    parse_generic_date,
    pull_categories_distances,
    escape_postgres_string,
    read_sql_data,
    pull_documents_from_elastic,
    get_and_reset_time,
    parse_datetime,
    cluster_label,
    setup_openai,
)
from utils.data_prep import ClusterizedDocumentsTable, ClustersTable
from utils.labels import generate_cluster_label_backoff


class DailyClusters:
    def __init__(
        self,
        ref_day: generic_date = datetime.now(),  # date filter for new documents
        cluster_TTL: int = 30,  # days, cluster time to live
        unclusterized_docs_TTL: int = 30,  # days, unclusterized documents time to live
        log_to_file: bool = False,
        log_to_console: bool = True,
        database_name: str = "quseed",
        categories_distance_table: str = "categories_distances",
        cluster_table: str = "clusters",
        clusterized_documents_table: str = "clusterized_docs",
        documents_es_index: str = "documents",
        embedding_column: str = "content_embedding",
    ):
        self.__init_logger__(log_to_file, log_to_console)
        self.ref_day: datetime = self.__parse_ref_day__(ref_day)
        self.database_name: str = escape_postgres_string(database_name)
        self.categories_distance_table: str = escape_postgres_string(
            categories_distance_table
        )
        self.cluster_table: str = escape_postgres_string(cluster_table)
        self.clusterized_documents_table: str = escape_postgres_string(
            clusterized_documents_table
        )
        self.unclusterized_docs_TTL: int = unclusterized_docs_TTL
        self.cluster_TTL: int = cluster_TTL
        self.documents_es_index: str = documents_es_index
        self.embedding_column: str = embedding_column
        setup_openai()

    def __init_logger__(
        self, log_to_file: bool = False, log_to_console: bool = True
    ) -> None:
        """
        Initializes the logger.

        Args:
            log_to_file (bool, optional): Set to True to allow log to file, file rotation is set to 1 day.
                Defaults to False.
            log_to_console (bool, optional): Set to True to allow log to console. Defaults to True.
        """
        logger.remove()
        if log_to_console:
            logger.add(
                sys.stdout,
                format=(
                    "<green>{time}</green> | <blue>[{level}]</blue> "
                    + "| <magenta>{name}.{module}.{function}</magenta> "
                    + "- <level>{message}</level>"
                ),
                level="DEBUG",
                colorize=True,
                backtrace=True,
                diagnose=True,
            )
        if log_to_file:
            logger.add(
                "../logs/clusterization_{time}.log",
                rotation="1 day",
                level="DEBUG",
                backtrace=True,
                diagnose=True,
            )
        self.logger = logger

    @logger.catch
    def __parse_ref_day__(self, ref_day: generic_date) -> datetime:
        """
        Parses a generic date into a datetime object.

        Args:
            ref_day (generic_date): date to parse

        Raises:
            TypeError: if <ref_day> is not a datetime, :date or :str

        Returns:
            datetime: parsed datetime object
        """
        if type(ref_day) not in [datetime, date, str]:
            logger.error(
                f"<ref_day> must be datetime, :date or :str, not {type(ref_day)}"
            )
            raise TypeError(
                f"<ref_day> must be datetime, :date or :str, not {type(ref_day)}"
            )
        else:
            return parse_generic_date(ref_day)

    @logger.catch
    def pull_documents(self) -> None:
        """
        Pulls all documents from Elasticsearch for a given day.
        """
        documents: pd.DataFrame = pull_daily_documents(self.ref_day)

        if documents.shape[0] == 0:
            logger.warning(f"No documents found for {self.ref_day}")
        else:
            logger.info(f"Pulled {documents.shape[0]} documents for {self.ref_day}")

        self.documents = documents

    @logger.catch
    def pull_categories_distance(self) -> None:
        """
        Pulls a dictionary containing the distance between all categories
        from the database.
        """
        categories_distances: pd.DataFrame = pull_categories_distances(
            database_name=self.database_name,
            table_name=self.categories_distance_table,
            output_type="dict",
        )

        if len(categories_distances) == 0:
            logger.warning("No categories distances found")
        else:
            logger.info(
                f"Pulled {len(categories_distances)} categories distances from the database"
            )

        self.categories_distances = categories_distances

    @logger.catch
    def pull_clusters_with_documents(self) -> None:
        """
        Pulls all clusters and their documents using elasticsearch and postgresql.

        Raises:
            ValueError: if no clusters are found
            e: if some clusterized documents cannot be pulled from postgresql
            e: if some documents cannot be pulled from Elasticsearch
            e: if some documents cannot be merged with the clusterized documents
            ValueError: if some documents are missing
        """

        cluster_query: str = """
        select *
        from {cluster_table}
        where timestamp > '{date_filter}'
        """.format(
            cluster_table=self.cluster_table,
            date_filter=(self.ref_day - relativedelta(days=self.cluster_TTL)).strftime(
                "%Y-%m-%d"
            ),
            ref_day=self.ref_day.strftime("%Y-%m-%d"),
        )

        clusters: pd.DataFrame = read_sql_data(
            query=cluster_query,
            table_name=self.cluster_table,
            database_name=self.database_name,
            output_type="df",
        )

        if clusters is None:
            clusters = pd.DataFrame([])

        if clusters.shape[0] == 0:
            logger.warning("No clusters found")
            self.clusters = pd.DataFrame([])
            self.clusterized_documents = pd.DataFrame([])
            return
        else:
            logger.info(f"Pulled {clusters.shape[0]} clusters from the database")

        if "cluster_id" in clusters.columns:
            clusters.rename(
                columns={"cluster_id": "id", "cluster_borders": "cluster_border"},
                inplace=True,
            )

        self.clusters = clusters

        cluster_id = clusters.id.unique().tolist()

        documents_query: str = """
        select *
        from {documents_table}
        where cluster_id in ({cluster_id})
        """
        postgresql_chunk_size: int = 50

        futures = []
        doc_chunks = []
        clusterized_documents: pd.DataFrame = pd.DataFrame([])

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                for idx in range(0, len(cluster_id), postgresql_chunk_size):
                    futures.append(
                        executor.submit(
                            read_sql_data,
                            query=documents_query.format(
                                documents_table=self.clusterized_documents_table,
                                cluster_id=",".join(
                                    [
                                        f"'{uuid}'"
                                        for uuid in cluster_id[
                                            idx : idx + postgresql_chunk_size
                                        ]
                                    ]
                                ),
                            ),
                            table_name=self.clusterized_documents_table,
                            database_name=self.database_name,
                            output_type="df",
                        )
                    )

                for future in as_completed(futures):
                    doc_chunks.append(future.result())
                    logger.debug(f"Pulled {len(doc_chunks)} chunks of documents")

            if not all([doc_chunk.shape[0] > 0 for doc_chunk in doc_chunks]):
                logger.error("Some documents are missing")
                raise ValueError("Some documents are missing")

            clusterized_documents: pd.DataFrame = pd.concat(
                doc_chunks, ignore_index=True
            )
        except Exception as e:
            logger.error(
                f"Error while pulling clusterized documents from postgresql. Database: {self.database_name}, Table: {self.clusterized_documents_table}"
            )
            raise e

        documents_url = clusterized_documents.url.unique().tolist()

        try:
            documents_body = pull_documents_from_elastic(
                doc_urls=documents_url, es_index=self.documents_es_index
            )
        except Exception as e:
            logger.error("Error while pulling documents from Elasticsearch.")
            raise e

        try:
            documents_body["url"] = documents_body["url"].apply(
                lambda x: x if not isinstance(x, list) else x[0]
            )
            clusterized_documents: pd.DataFrame = pd.merge(
                left=clusterized_documents,
                right=documents_body,
                how="inner",
                left_on="url",
                right_on="url",
                validate="1:1",
            )
        except Exception as e:
            logger.error(
                "Error while merging documents from Elasticsearch and PostgreSQL."
            )
            raise e

        self.clusterized_documents = clusterized_documents
        self.clusterized_documents[
            "timestamp"
        ] = self.clusterized_documents.date_created.apply(parse_datetime)
        self.clusterized_documents.rename(
            columns={"cluster_id": "cluster"},
            inplace=True,
        )

        if len(self.clusters):
            cluster_data: List[dict] = []

            for _, cluster in self.clusters.iterrows():
                cluster_documents = self.clusterized_documents.loc[
                    self.clusterized_documents.cluster == cluster.id
                ]
                if len(cluster_documents) == 0:
                    continue
                cluster_data.append(
                    generate_cluster_data(
                        cluster_id=cluster["id"],
                        documents=cluster_documents,
                        embedding_column="content_embedding",
                        categories_distance=self.categories_distances,
                    )
                )

            self.clusters = pd.DataFrame(cluster_data)

    @logger.catch
    def pull_unclusterized_documents(self) -> None:
        """
        Pulls all unclusterized documents from the database.

        Raises:
            e(Exception): if unclusterized documents cannot be pulled from postgresql
        """

        docs_query: str = """
        select *
        from {documents_table}
        where timestamp > '{date_filter}'
        and unclusterized = true
        """

        try:
            unclusterized_docs: pd.DataFrame = read_sql_data(
                query=docs_query.format(
                    documents_table=self.clusterized_documents_table,
                    date_filter=(
                        self.ref_day - relativedelta(days=self.unclusterized_docs_TTL)
                    ).strftime("%Y-%m-%d"),
                ),
                table_name=self.clusterized_documents_table,
                database_name=self.database_name,
                output_type="df",
                method="cx",
            )
        except Exception as e:
            logger.error("Error while pulling unclusterized documents from PostgreSQL.")
            raise e

        if unclusterized_docs is None:
            unclusterized_docs = pd.DataFrame([])

        if unclusterized_docs.shape[0] == 0:
            logger.warning("No unclusterized documents found")
            self.unclusterized_docs = unclusterized_docs
            return

        documents_url = unclusterized_docs.url.unique().tolist()
        if len(documents_url) == 0:
            logger.warning("No unclusterized documents found")
            self.unclusterized_docs = pd.DataFrame([])
            return

        try:
            documents_body = pull_documents_from_elastic(
                doc_urls=documents_url, es_index=self.documents_es_index
            )
        except Exception as e:
            logger.error("Error while pulling documents from Elasticsearch.")
            raise e

        documents_body["url"] = documents_body["url"].apply(
            lambda x: x if not isinstance(x, list) else x[0]
        )

        try:
            unclusterized_docs: pd.DataFrame = pd.merge(
                left=unclusterized_docs,
                right=documents_body,
                how="inner",
                left_on="url",
                right_on="url",
                validate="one_to_one",
            )
        except Exception as e:
            logger.error(
                "Error while merging unclusterized documents from Elasticsearch and PostgreSQL."
            )
            print(e)
            raise e

        self.unclusterized_docs = unclusterized_docs

    @logger.catch
    def pull_data(self) -> None:
        """
        Pulls all data from the database and Elasticsearch and prepares it for clusterization.

        Raises:
            e: Error while concatenating data to clusterize.
        """

        start_time: float = perf_counter()
        task_time: float = perf_counter()
        elapsed_time: float = 0.0

        self.pull_documents()
        elapsed_time, task_time = get_and_reset_time(task_time)
        logger.info(f"Pulled documents in :{elapsed_time} seconds")
        self.pull_categories_distance()
        elapsed_time, task_time = get_and_reset_time(task_time)
        logger.info(f"Pulled categories distances in :{elapsed_time} seconds")
        self.pull_clusters_with_documents()
        elapsed_time, task_time = get_and_reset_time(task_time)
        logger.info(f"Pulled clusters with documents in :{elapsed_time} seconds")
        self.pull_unclusterized_documents()
        elapsed_time, task_time = get_and_reset_time(task_time)
        logger.info(f"Pulled unclusterized documents in :{elapsed_time} seconds")

        logger.info(f"Data pull done in :{perf_counter() - start_time} seconds")

    @logger.catch
    def merge_documents(self) -> None:
        """
        Merges new documents (to clusterize) with old documents (unclusterized).

        Raises:
            e: Error while concatenating data to clusterize.
        """

        start_time: float = perf_counter()
        try:
            data_to_clusterize = [
                data
                for data in [
                    self.documents,
                    self.unclusterized_docs,
                ]
                if data.shape[0] > 0
            ]

            self.documents = pd.concat(data_to_clusterize, ignore_index=True)
            self.documents["url"] = self.documents["url"].apply(
                lambda x: x if not isinstance(x, list) else x[0]
            )
            self.documents.drop_duplicates(subset=["url"], inplace=True, keep="first")

            def adjust_clusters(row):
                if row.get("cluster", None):
                    if isinstance(row["cluster"], list):
                        return row["cluster"][0]
                    else:
                        return row["cluster"]
                return None

            self.documents["cluster"] = self.documents.apply(adjust_clusters, axis=1)
            self.documents["timestamp"] = self.documents.date_created.apply(
                parse_datetime
            )
            self.documents["url"] = self.documents["url"].apply(
                lambda x: x if not isinstance(x, list) else x[0]
            )
            self.documents.dropna(subset=["categories"], inplace=True)
        except Exception as e:
            logger.error("Error while concatenating data to clusterize.")
            raise e

        logger.info(
            f"Concatenated data to clusterize (new documents, old documents unclusterized) in :{perf_counter() - start_time} seconds"
        )

    @logger.catch
    def append_documents_to_existing_clusters(self) -> None:
        """
        Appends new documents to existing clusters.
        """
        if self.clusters.shape[0] == 0:
            self.docs_to_old_clusters: Dict[str, str] = {}
            return

        start_time: float = perf_counter()

        if not len(self.clusters):
            return

        self.docs_to_old_clusters = append_documents_to_existing_clusters(
            documents=self.documents,
            embedding_column="content_embedding",
            clusters=self.clusters,
            logger=self.logger,
            categories_distances=self.categories_distances,
        )

        logger.info(
            f"Appended documents to existing clusters in :{perf_counter() - start_time} seconds"
        )

    @logger.catch
    def dbscan_first_layer(
        self,
        eps: float = 0.53,
        min_samples: int = 2,
        embedding_column: str = "content_embedding",
        distance_args: Dict[str, float] = {
            "content": 0.1,
            "title": 0.8,
            "categories": 0.1,
        },
    ) -> None:
        """
        Runs a first layer of clustering on all documents using DBSCAN and after thats
        splits "big clusters" using HDBSCAN.
        Assign cluster labels in <self.documents.cluster> and stores cluster data
        into <self.clusters>.

        Args:
            eps (float, optional): eps parameter for dbscan. Defaults to 0.53.
            min_samples (int, optional): min_samples parameter for dbscan. Defaults to 2.
            embedding_column (str, optional): column containing embeddings. Defaults to "content_embedding".
            distance_args (_type_, optional): dictionary containing distance arguments. Defaults to {
                "content": 0.1,
                "title": 0.8,
                "categories": 0.1,
            }.
        """
        self.logger.debug("Starting dbscan first layer")
        b_time: float = perf_counter()
        clusterization_time: float = perf_counter()

        docs: List[dict] = self.documents.to_dict(orient="records")
        (
            doc_to_clusters,
            centroids,
            central_documents,
            cluster_borders,
        ) = dbscan_clustering(
            documents=docs,
            eps=eps,
            min_samples=min_samples,
            distance_args=distance_args,
            embedding_column="content_embedding",
            categories_distances=self.categories_distances,
        )
        logger.debug(
            f"First layer clustering done in :{perf_counter() - b_time} seconds"
        )

        new_clusters = []
        detected_clusters = list(central_documents.keys())
        cluster_label_mapping: Dict[int, str] = {}

        for cluster in detected_clusters:
            cluster_id = cluster_label(cluster, self.ref_day)
            cluster_label_mapping[cluster] = cluster_id

            cluster_centroid = centroids[cluster]
            cluster_central_document = central_documents[cluster]
            cluster_borders_info = cluster_borders[cluster]

            new_clusters.append(
                {
                    "id": cluster_id,
                    "label": cluster,
                    "timestamp": self.ref_day,
                    "centroid_embedding": np.array(cluster_centroid),
                    "central_doc": cluster_central_document,
                    "cluster_border": cluster_borders_info,
                }
            )
        logger.debug(f"New clusters: {len(new_clusters)}")
        logger.debug("Cluster mapping")

        for k, v in doc_to_clusters.items():
            doc_to_clusters[k] = [cluster_label_mapping[c] for c in v][0]

        self.documents["cluster"] = self.documents.apply(
            lambda x: value
            if (value := doc_to_clusters.get(x["url"]))
            else x["cluster"],
            axis=1,
        )
        self.documents["old_doc"] = self.documents.timestamp.apply(
            lambda x: x != self.ref_day
        )

        self.logger.debug("Outliers splitting")
        b_time = perf_counter()

        self.documents, outlier_clusters = outlier_clusters_splitting(
            documents=self.documents,
            embedding_col=embedding_column,
        )

        new_clusters.extend(outlier_clusters)
        self.clusters = pd.concat(
            [self.clusters, pd.DataFrame(new_clusters)], ignore_index=True
        )

        self.documents["first_layer"] = self.documents.url.apply(
            lambda x: not doc_to_clusters.get(x, "not-found-1").endswith("-1")
        )
        self.logger.info(
            f"First level clusterization done in :{perf_counter() - clusterization_time} seconds"
        )

    @logger.catch
    def clusters_sanity_check(self) -> None:
        """
        Runs a sanity check on clusters data and adjusts it if needed.
        """
        self.logger.info("Starting sanity check")
        b_time: float = perf_counter()

        self.documents.drop_duplicates(subset=["url"], inplace=True, keep="first")

        clusters_to_drop: Optional[List[str]] = []
        if not hasattr(self, "documents_corpus"):
            if len(self.clusterized_documents):
                clusterized_documents = self.clusterized_documents.loc[
                    ~self.clusterized_documents.url.isin(self.documents.url.tolist())
                ]
            else:
                clusterized_documents = pd.DataFrame([])

            document_corpus = pd.concat(
                [
                    data
                    for data in [
                        self.documents,
                        clusterized_documents,
                    ]
                    if len(data)
                ],
                ignore_index=True,
            )
            if document_corpus.shape[0] != 0:
                self.documents_corpus = document_corpus
                return

            document_corpus["url"] = document_corpus["url"].apply(
                lambda x: x if not isinstance(x, list) else x[0]
            )
            document_corpus["cluster"] = document_corpus["cluster"].apply(
                lambda x: x[0] if isinstance(x, list) else x
            )
            document_corpus.drop_duplicates(subset=["url"], inplace=True, keep="first")
            document_corpus = document_corpus[
                document_corpus["categories"].apply(
                    lambda x: isinstance(x, list) and len(x) > 0
                )
                | document_corpus["categories"].isna()
            ]

            document_corpus["timestamp"] = document_corpus.date_created.apply(
                parse_datetime
            )

            self.documents_corpus = document_corpus
        else:
            document_corpus = self.documents_corpus

        valid_clusters = []
        for _, cluster in self.clusters.iterrows():
            cluster_documents = document_corpus.loc[
                document_corpus.cluster == cluster.id
            ]
            if not len(cluster_documents):
                continue
            valid_clusters.append(
                generate_cluster_data(
                    cluster_id=cluster["id"],
                    documents=cluster_documents,
                    embedding_column="content_embedding",
                    categories_distance=self.categories_distances,
                )
            )
        if hasattr(self, "clusters_to_drop"):
            self.clusters_to_drop = list(set(clusters_to_drop + self.clusters_to_drop))
        else:
            self.clusters_to_drop = clusters_to_drop
        self.clusters = pd.DataFrame(valid_clusters)
        self.logger.info(f"Sanity check done in :{perf_counter() - b_time} seconds")

    @logger.catch
    def dbscan_second_layer(
        self,
        eps: float = 0.59,
        min_samples: int = 2,
        embedding_column: str = "content_embedding",
        distance_args: Dict[str, float] = {
            "content": 0.1,
            "title": 0.8,
            "categories": 0.1,
        },
    ) -> None:
        """
        Runs a second layer of clustering on unclusterized documents.

        Args:
            eps (float, optional): eps parameter for dbscan. Defaults to 0.59.
            min_samples (int, optional): min_samples parameter for dbscan. Defaults to 2.
            embedding_column (str, optional): column containing embeddings. Defaults to "content_embedding".
            distance_args (_type_, optional): dictionary containing distance arguments. Defaults to {
                "content": 0.1,
                "title": 0.8,
                "categories": 0.1,
            }.
        """
        b_time: float = perf_counter()
        self.logger.info("Starting dbscan second layer")
        documents: pd.DataFrame = self.documents.loc[
            self.documents.cluster.str.endswith("-1")
        ].copy(deep=True)

        if not len(documents):
            self.logger.info("No documents to clusterize in second layer")
            return

        docs: List[dict] = documents.to_dict(orient="records")
        (
            docs_to_cluster,
            centroids,
            central_documents,
            cluster_borders,
        ) = dbscan_clustering(
            documents=docs,
            eps=eps,
            min_samples=min_samples,
            distance_args=distance_args,
            embedding_column=embedding_column,
            categories_distances=self.categories_distances,
        )

        previous_id = len(self.clusters) + 1
        cluster_mapping: Dict[str, str] = {}
        for cluster in central_documents.keys():
            if cluster == -1:
                cluster_mapping[cluster] = cluster_label(-1, self.ref_day)
            else:
                cluster_mapping[cluster] = cluster_label(previous_id, self.ref_day)
                previous_id += 1

        centroids = {cluster_mapping[k]: v for k, v in centroids.items()}

        central_documents = {
            cluster_mapping[k]: v for k, v in central_documents.items()
        }

        docs_to_cluster = {
            doc: cluster_mapping[clusters[0]]
            for doc, clusters in docs_to_cluster.items()
        }

        cluster_borders = {
            cluster_mapping[cluster]: cluster_borders[cluster]
            for cluster in cluster_borders.keys()
        }

        new_clusters = []

        for cluster in cluster_mapping.values():
            if self.ref_day.strftime("%Y-%m-%d") not in cluster:
                cluster_id = cluster_label(cluster, self.ref_day)
            else:
                cluster_id = cluster

            cluster_centroid = centroids[cluster]
            cluster_central_document = central_documents[cluster]

            cluster_borders_info = cluster_borders[cluster]

            new_clusters.append(
                {
                    "id": cluster_id,
                    "label": cluster,
                    "timestamp": self.ref_day,
                    "centroid_embedding": np.array(cluster_centroid),
                    "central_doc": cluster_central_document,
                    "cluster_border": cluster_borders_info,
                }
            )

        self.documents["cluster"] = self.documents.apply(
            lambda x: value
            if (value := cluster_mapping.get(x["url"]))
            else x["cluster"],
            axis=1,
        )
        self.documents["second_layer"] = self.documents.url.apply(
            lambda x: not docs_to_cluster.get(x, "not-found-1").endswith("-1")
        )
        documents["cluster"] = documents.apply(
            lambda x: value
            if (value := cluster_mapping.get(x["url"]))
            else x["cluster"],
            axis=1,
        )

        documents, sub_clusters = outlier_clusters_splitting(
            documents=documents,
            embedding_col=embedding_column,
            logger=self.logger,
            current_date=self.ref_day,
        )

        if len(sub_clusters) > 0:
            cluster_mapper = {
                row["url"]: row["cluster"] for _, row in documents.iterrows()
            }
            self.documents["cluster"] = self.documents.apply(
                lambda x: value
                if (value := cluster_mapper.get(x["url"]))
                else x["cluster"],
                axis=1,
            )
            new_clusters.extend(sub_clusters)

        self.clusters = pd.concat(
            [self.clusters, pd.DataFrame(new_clusters)], ignore_index=True
        )
        self.logger.debug(f"Detected new {len(new_clusters)} clusters")
        self.logger.info(
            f"Second level clusterization done in :{perf_counter() - b_time} seconds"
        )

    @logger.catch
    def split_big_clusters(self) -> None:
        """
        Splits big clusters into smaller ones using dbscan.
        """
        self.logger.info("Starting big clusters splitting")
        b_time: float = perf_counter()

        if "first_layer" not in self.documents_corpus.columns:
            self.documents_corpus["first_layer"] = False
        self.documents_corpus["first_layer"].fillna(False, inplace=True)
        if "second_layer" not in self.documents_corpus.columns:
            self.documents_corpus["second_layer"] = False
        self.documents_corpus["second_layer"].fillna(False, inplace=True)

        self.documents_corpus, self.clusters = split_big_clusters(
            documents=self.documents_corpus,
            clusters=self.clusters,
            embedding_column="content_embedding",
            categories_distance=self.categories_distances,
            logger=self.logger,
            ref_day=self.ref_day,
        )

        self.logger.info(
            f"Big clusters splitting done in :{perf_counter() - b_time} seconds"
        )

    @logger.catch
    def prepare_clusterized_documents(self) -> None:
        """
        Prepares clusterized documents for upsert into the database.
        """
        output: List[dict] = []

        for _, document in self.documents_corpus.iterrows():
            cluster = (
                str(document["cluster"][0])
                if isinstance(document["cluster"], list)
                else str(document["cluster"])
            )
            document_data = {
                "url": document["url"],
                "cluster_id": cluster,
                "timestamp": document.get("timestamp", self.ref_day),
                "unclusterized": str(cluster).endswith("-1"),
            }

            if not document["cluster"]:
                document_data["cluster_id"] = None
                document_data["unclusterized"] = True
            output.append(document_data)

        self.output_clusterized_documents = output

    @logger.catch
    def prepare_clusters(self) -> None:
        """
        Prepares clusters for upsert into the database.
        """
        output: List[dict] = []

        for _, cluster in self.clusters.iterrows():
            cluster_data = {
                "uuid": uuid.uuid4(),
                "cluster_id": cluster["id"],
                "cluster_label": None,
                "centroid_embedding": cluster["centroid_embedding"],
                "central_doc": cluster["central_doc"],
                "cluster_borders": cluster["cluster_border"],
                "timestamp": cluster["timestamp"],
                "categories": [],
                "most_common_categories": [],
                "documents_titles": [],
            }
            if cluster["central_doc"].get("timestamp", None):
                del cluster["central_doc"]["timestamp"]

            cluster_data["categories"] = list(
                chain.from_iterable(
                    self.documents_corpus[
                        self.documents_corpus["cluster"] == cluster["id"]
                    ]["categories"]
                )
            )
            cluster_data["most_common_categories"] = Counter(
                cluster_data["categories"]
            ).most_common(20)
            cluster_data["documents_titles"] = self.documents_corpus[
                self.documents_corpus["cluster"] == cluster["id"]
            ]["title"].tolist()
            output.append(cluster_data)

        self.output_clusters = generate_cluster_label_backoff(output)

    @logger.catch
    def pick_best_clusterization(self) -> None:
        self.logger.info("Picking best clusterization")
        b_time: float = perf_counter()

        if not hasattr(self, "docs_to_old_clusters"):
            return

        if not len(self.docs_to_old_clusters):
            return

        document_mapping: Dict[str, str] = pick_best_clustering(
            clusters=self.clusters,
            documents=self.documents,
            document_to_old_cluster=self.docs_to_old_clusters,
            categories_distances=self.categories_distances,
        )

        if len(document_mapping):

            def apply_new_clusterization(row, document_mapping) -> str:
                if value := document_mapping.get(row["url"]):
                    if isinstance(value, dict):
                        return value["label"]
                    else:
                        return value
                else:
                    return row["cluster"]

            self.documents["cluster"] = self.documents.apply(
                lambda x: apply_new_clusterization(x, document_mapping),
                axis=1,
            )

        self.logger.info(
            f"Best clusterization picked in :{perf_counter() - b_time} seconds"
        )

    @logger.catch
    def get_clusters_to_drop(self) -> None:
        """
        Gets clusters that should be dropped from the database.
        """
        existing_cluster_query: str = f"""
        select uuid, cluster_id
        from {self.cluster_table}
        """
        current_clusters_df = read_sql_data(
            query=existing_cluster_query,
            table_name=self.cluster_table,
            database_name=self.database_name,
            output_type="df",
        )

        if current_clusters_df is None:
            return

        current_clusters = set(current_clusters_df.cluster_id.unique().tolist())

        documents_clusters_query: str = f"""
        select distinct cluster_id
        from {self.clusterized_documents_table}
        """
        documents_clusters = read_sql_data(
            query=documents_clusters_query,
            table_name=self.clusterized_documents_table,
            database_name=self.database_name,
            output_type="df",
        )

        if not len(documents_clusters):
            return

        documents_clusters = set(documents_clusters.cluster_id.unique().tolist())

        clusters_to_drop: List[Optional[str]] = list(
            documents_clusters.difference(current_clusters)
        )

        if len(clusters_to_drop):
            self.logger.debug(f"Found {len(clusters_to_drop)} clusters to drop")
            clusters_to_drop = current_clusters_df.loc[
                current_clusters_df.cluster_id.isin(clusters_to_drop)
            ]["uuid"].tolist()

        self.clusters_to_drop = clusters_to_drop

    @logger.catch
    def publish_output(self) -> None:
        """
        Publishes clusterized documents and clusters to the database.
        """
        self.logger.info("Publishing output to the database")

        self.logger.debug(f"Pushing {len(self.output_clusters) | 0} clusters")
        clusters_table = ClustersTable(
            database_name=self.database_name,
            table_name=self.cluster_table,
        )
        clusters_table.init(reset=False)
        clusters_table.upsert_records(self.output_clusters)

        self.logger.debug(
            f"Pushing {len(self.output_clusterized_documents) | 0} clusterized documents"
        )
        clusterized_documents_table = ClusterizedDocumentsTable(
            database_name=self.database_name,
            table_name=self.clusterized_documents_table,
        )
        clusterized_documents_table.init(reset=False)
        clusterized_documents_table.upsert_rows(self.output_clusterized_documents)

        self.logger.debug(f"Dropping {len(self.clusters_to_drop)} clusters")
        self.get_clusters_to_drop()
        clusters_table.delete_clusters_by_uuids(self.clusters_to_drop)

    @logger.catch
    def run(self) -> None:
        logger.info("Starting daily clusterization")
        begin_time: float = perf_counter()

        self.pull_data()
        self.append_documents_to_existing_clusters()
        self.merge_documents()
        self.dbscan_first_layer()
        self.dbscan_second_layer()
        self.pick_best_clusterization()
        self.clusters_sanity_check()
        self.split_big_clusters()
        self.clusters_sanity_check()
        self.prepare_clusterized_documents()
        self.prepare_clusters()
        self.publish_output()

        logger.info(
            f"Daily clusterization done in :{perf_counter() - begin_time} seconds"
        )
