import os, sys

sys.path.append(os.getcwd())

from typing import List, Dict, Tuple, Any
import copy
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from time import sleep, perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from collections import Counter
import json
import uuid

from loguru import logger
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch

from sb_py_fw.database.elastic import ElasticConnection

from .communities_utils import (
    create_communities,
    split_big_communities,
    test_cluster_distance,
)
from utils import read_sql_data, pull_categories_distances, cluster_label
from utils.data_pull import get_clusters_titles, get_community_data
from utils.labels import generate_community_label_gpt
from utils.openai_services import setup_openai
from utils.data_prep.clusters_community_table import ClusterCommunitiesTable
from utils.data_prep.communities_table import CommunitiesTable

setup_openai()


class CommunitiesClustering:
    def __init__(
        self,
        ref_date: datetime = datetime.now(),
        clusters_TTL: int = 30,  # days
        clusters_table_name: str = "clusters",
        communities_TTL: int = 30,  # days
        communities_table_name: str = "communities",
        cluster_communities_table_name: str = "cluster_communities",
        clusterized_documents_table: str = "clusterized_docs",
        categories_distance_table: str = "categories_distances",
        database_name: str = "quseed",
        log_to_file: bool = False,
        log_to_console: bool = True,
    ) -> None:
        self.__init_logger__(log_to_file, log_to_console)
        self.ref_date: datetime = ref_date
        self.clusters_TTL: int = clusters_TTL  # days
        self.clusters_table_name: str = clusters_table_name
        self.cluster_communities_table_name: str = cluster_communities_table_name
        self.communities_TTL: int = communities_TTL  # days
        self.communities_table_name: str = communities_table_name
        self.clusterized_documents_table: str = clusterized_documents_table
        self.categories_distances_table: str = categories_distance_table
        self.database_name: str = database_name

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
    def pull_clusters_data(self) -> None:
        """
        Pulls clusters data from the database.
        """
        clusters_query = """
        select *
        from {clusters_table_name}
        where
            timestamp >= '{filter_date}'
        """.format(
            clusters_table_name=self.clusters_table_name,
            filter_date=(
                self.ref_date - relativedelta(days=self.clusters_TTL)
            ).strftime("%Y-%m-%d"),
        )
        clusters: pd.DataFrame = read_sql_data(
            query=clusters_query,
            table_name=self.clusters_table_name,
            database_name=self.database_name,
        )

        if clusters is None:
            return

        cluster_sizes_query: str = """
        select cluster_id, count(*) as cluster_size
        from {cluserized_docs_table}
        group by cluster_id
        """.format(
            cluserized_docs_table=self.clusterized_documents_table
        )
        cluster_sizes: pd.DataFrame = read_sql_data(
            query=cluster_sizes_query,
            table_name=self.clusterized_documents_table,
            database_name=self.database_name,
        )

        if cluster_sizes is None:
            return

        clusters_data: List[Dict[str, Any]] = []
        for cluster in clusters.to_dict(orient="records"):
            if cluster["cluster_id"].endswith("-1"):
                continue

            cluster_size: int = cluster_sizes.loc[
                cluster_sizes["cluster_id"] == cluster["cluster_id"]
            ]
            if len(cluster_size) == 0:
                continue
            cluster["size"] = cluster_size["cluster_size"].values[0]

            clusters_data.append(cluster)

        if len(clusters_data):
            self.clusters = pd.DataFrame(clusters_data)

    @logger.catch
    def pull_categories_distance(self) -> None:
        """
        Pulls a dictionary containing the distance between all categories
        from the database.
        """
        categories_distances: pd.DataFrame = pull_categories_distances(
            database_name=self.database_name,
            table_name=self.categories_distances_table,
            output_type="dict",
        )

        if len(categories_distances) == 0:
            logger.warning("No categories distances found")
        else:
            logger.info(
                f"Pulled {len(categories_distances)} categories distances from the database"
            )

        self.categories_distances = categories_distances

    def pull_existing_communities(self) -> None:
        """
        Pulls existing communities from the database.
        """
        existing_communities_query: str = """
        select cluster_id, community_id, unclusterized
        from {cluster_communities_table}
        where
            timestamp >= '{filter_date}'
        """.format(
            cluster_communities_table=self.cluster_communities_table_name,
            filter_date=(
                self.ref_date - relativedelta(days=self.communities_TTL)
            ).strftime("%Y-%m-%d"),
        )

        existing_communities: pd.DataFrame = read_sql_data(
            query=existing_communities_query,
            table_name=self.communities_table_name,
            database_name=self.database_name,
        )

        if existing_communities is None:
            return

        existing_communities_clusters: List[str] = (
            existing_communities.loc[existing_communities["unclusterized"] == False][
                "cluster_id"
            ]
            .unique()
            .tolist()
        )
        old_communities: List[str] = (
            existing_communities.loc[existing_communities["unclusterized"] == False][
                "community_id"
            ]
            .unique()
            .tolist()
        )

        # remove clusterized clusters from candidates
        self.clusters = self.clusters.loc[
            ~self.clusters["cluster_id"].isin(existing_communities_clusters)
        ]

        if len(self.clusters) == 0:
            del self.clusters

        if len(old_communities):
            self.old_communities = old_communities

    def hydrate_old_communities(self) -> None:
        """
        Hydrates comunities data.
        """
        old_communities: List[dict] = []

        for community in self.old_communities:
            community_data = get_community_data(
                community_id=community,
                database_name=self.database_name,
                community_table_name=self.communities_table_name,
                clusters_table_name=self.clusters_table_name,
                cluster_communities_table_name=self.cluster_communities_table_name,
            )
            if community_data is None:
                continue
            old_communities.append(community_data)

        self.old_communities = old_communities

    def pull_data(self) -> None:
        """
        Macro function to pull all the data needed for the communities clustering.
        """
        self.pull_categories_distance()
        self.pull_clusters_data()
        self.pull_existing_communities()

        if hasattr(self, "old_communities"):
            self.hydrate_old_communities()

    def append_clusters_to_old_communities(self) -> None:
        """
        Appends clusters to old communities.
        """
        if not hasattr(self, "old_communities"):
            return
        communities_data = [
            {
                "community_id": community["community_id"],
                "centroid": community["centroid"],
                "categories": community["categories"],
            }
            for community in self.old_communities
        ]
        clusters_to_old_communities = []

        for cluster in self.clusters.to_dict(orient="records"):
            cluster_data = {
                "centroid": cluster["centroid_embedding"],
                "categories": cluster["categories"],
            }

            candidate_community = test_cluster_distance(
                cluster=cluster_data,
                communities=communities_data,
                categories_distances=self.categories_distances,
            )

            if candidate_community is not None:
                clusters_to_old_communities.append(
                    {
                        **cluster,
                        "community_id": candidate_community,
                    }
                )

        if len(clusters_to_old_communities):
            self.clusters_to_old_communities = pd.DataFrame(clusters_to_old_communities)
            self.logger.debug(
                f"Found {len(self.clusters_to_old_communities)} clusters to append to old communities."
            )
            self.clusters = self.clusters.loc[
                ~self.clusters["cluster_id"].isin(
                    self.clusters_to_old_communities["cluster_id"]
                )
            ]

    def first_level_clustering(self) -> None:
        """
        First (and only) layer of clusterization.
        """
        b_time: float = perf_counter()

        community_labels = create_communities(
            clusters=self.clusters,
            eps=0.216,
            dbscan_min_samples=2,
            distance_args={
                "centroid": 0.75,
                "categories": 0.25,
                "categories_distances": self.categories_distances,
            },
        )

        labels_mapper: Dict[int, str] = {
            label: cluster_label(label, self.ref_date)
            for label in set(community_labels)
        }

        self.clusters["community_id"] = community_labels
        self.clusters["community_id"] = self.clusters["community_id"].map(labels_mapper)

        logger.debug(f"First level clustering took {perf_counter() - b_time} seconds.")

    def split_big_communities(self) -> None:
        """
        Split big communities in smaller ones
        """
        b_time: float = perf_counter()

        if hasattr(self, "clusters_to_old_communities"):
            self.clusters = pd.concat([self.clusters, self.clusters_to_old_communities])

        split_big_communities(
            clusters=self.clusters,
            categories_distances=self.categories_distances,
            ref_date=self.ref_date,
        )

        self.logger.debug(
            f"Splitting big communities took {perf_counter() - b_time} seconds."
        )

    def prepare_outputs(self) -> None:
        """
        Prepares the outputs before publishing them.
        """
        b_time: float = perf_counter()

        es_client: Elasticsearch = ElasticConnection().es

        clusters_community: List[dict] = []
        communities: Dict[str, dict] = {}
        selected_columns = [
            "community_id",
            "cluster_id",
            "timestamp",
            "categories",
            "tags",
            "size",
            "cluster_label",
        ]

        for cluster in self.clusters[[*selected_columns]].to_dict(orient="records"):
            cluster["timestamp"] = cluster["timestamp"].strftime("%Y-%m-%d")
            cluster["unclusterized"] = cluster["community_id"].endswith("_-1")

            if cluster["unclusterized"]:
                # single cluster => community
                community_id = cluster_label(cluster["cluster_id"], self.ref_date)
                communities[community_id] = {
                    "community_id": community_id,
                    "community_label": cluster["cluster_label"],
                    "categories": [cluster["categories"]],
                    "timestamp": cluster["timestamp"],
                    "size": [cluster["size"]],
                    "tags": [cluster["tags"]],
                }
            if communities.get(cluster["community_id"]) is None:
                communities[cluster["community_id"]] = {
                    "community_id": cluster["community_id"],
                    "labels": [cluster["cluster_label"]],
                    "categories": [cluster["categories"]],
                    "timestamp": cluster["timestamp"],
                    "size": [cluster["size"]],
                    "clusters": [
                        cluster["cluster_id"],
                    ],
                    "tags": [cluster["tags"]],
                }
            else:
                communities[cluster["community_id"]]["labels"].append(
                    cluster["cluster_label"]
                )
                communities[cluster["community_id"]]["categories"].append(
                    cluster["categories"]
                )
                communities[cluster["community_id"]]["size"].append(cluster["size"])
                communities[cluster["community_id"]]["clusters"].append(
                    cluster["cluster_id"]
                )
                communities[cluster["community_id"]]["tags"].append(cluster["tags"])

                if (
                    cluster["timestamp"]
                    < communities[cluster["community_id"]]["timestamp"]
                ):
                    communities[cluster["community_id"]]["timestamp"] = cluster[
                        "timestamp"
                    ]

            clusters_community.append(
                {
                    "cluster_id": cluster["cluster_id"],
                    "community_id": cluster["community_id"],
                    "timestamp": cluster["timestamp"],
                    "unclusterized": cluster["unclusterized"],
                }
            )

        communities = list(communities.values())

        for community in communities:
            community["categories"] = list(chain.from_iterable(community["categories"]))
            community["most_common_categories"] = [
                x[0] for x in Counter(community["categories"]).most_common(20)
            ]
            community["tags"] = list(chain.from_iterable(community["tags"]))
            community["size"] = sum(community["size"])
            if community.get("clusters"):
                community["titles"] = get_clusters_titles(
                    cluster_ids=community["clusters"],
                    documents_ix_es="documents",
                    es_client=es_client,
                )
                community["community_label"] = generate_community_label_gpt(
                    cluster_labels=community["labels"],
                    community_titles=community["titles"],
                    community_tags=community["tags"],
                )
                del community["titles"]
                del community["tags"]
                del community["labels"]
                del community["clusters"]

        es_client.close()

        self.output_clusters_community = clusters_community
        self.output_communities = communities
        self.logger.debug(
            f"Output preparation done in {perf_counter() - b_time} seconds."
        )

    def publish_outputs(self) -> None:
        """
        Publish the outputs to the database.
        """
        b_time: float = perf_counter()

        clusters_community_table = ClusterCommunitiesTable(
            database_name=self.database_name,
            table_name=self.cluster_communities_table_name,
        )
        clusters_community_table.init(reset=False)
        clusters_community_table.upsert_rows(self.output_clusters_community)

        communities_table = CommunitiesTable(
            database_name=self.database_name,
            table_name=self.communities_table_name,
        )
        communities_table.init(reset=False)
        communities_table.upsert_records(self.output_communities)

        self.logger.debug(
            f"Output publishing done in {perf_counter() - b_time} seconds."
        )

    def sanity_check(self) -> None:
        """
        Checks clusters data and removes invalid ones.
        """
        valid_clusters_query: str = """
        select cluster_id
        from {clusters_table_name}
        where
            timestamp >= '{filter_date}'
        """.format(
            clusters_table_name=self.clusters_table_name,
            filter_date=(
                self.ref_date - relativedelta(days=self.clusters_TTL)
            ).strftime("%Y-%m-%d"),
        )

        valid_clusters: pd.DataFrame = read_sql_data(
            query=valid_clusters_query,
            table_name=self.clusters_table_name,
            database_name=self.database_name,
        )

        if valid_clusters is None:
            return

        valid_clusters = valid_clusters["cluster_id"].unique().tolist()

        community_clusters_query: str = """
        select cluster_id, community_id, unclusterized
        from {cluster_communities_table}
        where
            timestamp >= '{filter_date}'
        """.format(
            cluster_communities_table=self.cluster_communities_table_name,
            filter_date=(
                self.ref_date - relativedelta(days=self.communities_TTL)
            ).strftime("%Y-%m-%d"),
        )

        community_clusters: pd.DataFrame = read_sql_data(
            query=community_clusters_query,
            table_name=self.cluster_communities_table_name,
            database_name=self.database_name,
        )

        if community_clusters is None:
            return

        missing_clusters = community_clusters.loc[
            ~community_clusters["cluster_id"].isin(valid_clusters)
        ]
        if len(missing_clusters):
            # drop missing clusters
            clusters_community = ClusterCommunitiesTable(
                database_name=self.database_name,
                table_name=self.cluster_communities_table_name,
            )
            clusters_community.init(reset=False)
            clusters_community.delete_by_cluster_id(
                missing_clusters["cluster_id"].unique().tolist()
            )

            communities_to_drop = missing_clusters["community_id"].unique().tolist()
            communities_table = CommunitiesTable(
                database_name=self.database_name,
                table_name=self.communities_table_name,
            )
            communities_table.init(reset=False)
            communities_table.delete_comunities_by_community_id(communities_to_drop)

            del clusters_community
            del communities_table

            self.logger.info(f"Dropping {len(communities_to_drop)} communities.")

        # temporary communities
        temporary_communities = community_clusters.loc[
            community_clusters["unclusterized"] == True
        ]
        if len(temporary_communities):
            communities_to_drop = (
                temporary_communities["community_id"].unique().tolist()
            )

            communities_table = CommunitiesTable(
                database_name=self.database_name,
                table_name=self.communities_table_name,
            )
            communities_table.init(reset=False)
            communities_table.delete_comunities_by_community_id(communities_to_drop)

    def run(self) -> None:
        """
        Macro for class execution.
        """
        self.sanity_check()
        self.pull_data()

        if not hasattr(self, "clusters"):
            self.logger.info("No clusters to process.")
            return

        self.logger.info(f"Processing {len(self.clusters)} clusters.")
        self.logger.info("Appending clusters to old communities.")
        self.append_clusters_to_old_communities()
        self.logger.info("Starting first level clustering.")
        self.first_level_clustering()
        self.logger.info("First level clustering completed.")
        self.logger.info("Starting splitting big communities.")
        self.split_big_communities()
        self.logger.info("Splitting big communities completed.")
        self.logger.info("Preparing outputs.")
        self.prepare_outputs()
        self.logger.info("Outputs prepared.")
        self.logger.info("Publishing outputs.")
        self.publish_outputs()
        self.logger.info("Outputs published.")
