# Description: utilities used during community creation.
from typing import Dict, List, Tuple, Any, Iterable, Optional
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from utils import euclidean_distance, cluster_label


def get_most_common_cluster_categories(
    cluster_categories: List[str],
) -> List[Tuple[str, float]]:
    """
    Gets the most common categories in a cluster.

    Args:
        cluster_categories (List[str]): list of categories in a cluster.

    Returns:
        List[Tuple[str, float]]: a list of tuples containing the category and its percentage in the cluster.
    """
    categories: List[Tuple[str, float]] = []

    for category_frequency in sorted(
        Counter(cluster_categories).items(), key=lambda x: x[1], reverse=True
    ):
        if len(categories) < 3:
            categories.append(category_frequency)

        if sum([x[1] for x in categories]) > 0.2 * len(categories):
            break

        if len(categories) == 6:
            break

        categories.append(category_frequency)

    for i in range(len(categories)):
        categories[i] = (categories[i][0], categories[i][1] / len(cluster_categories))

    return categories


def cluster_distance(
    cluster1: Dict[str, Any],
    cluster2: Dict[str, Any],
    distance_args: Dict[str, Any],
    categories_alpha: bool = True,
) -> float:
    """
    Calculates the distance between two clusters.

    Args:
        cluster1 (Dict[str, Any]): cluster data.
        cluster2 (Dict[str, Any]): cluster data.
        distance_args (Dict[str, Any]): dictionary containing the distance arguments.
        categories_alpha (bool, optional): _description_. Defaults to True.

    Raises:
        ValueError: Cluster(<cluster1>, <cluster2>) must contain 'centroid' and
            'categories' keys.
        ValueError: <distance_args> must contain 'centroid' and 'categories_distances' keys.

    Returns:
        float: distance between the two clusters.
    """
    required_args = ["centroid", "categories"]
    for cluster in [cluster1, cluster2]:
        for arg in required_args:
            if arg not in cluster.keys():
                raise ValueError(f"Cluster must contain <{arg}> key.")

    required_args.append("categories_distances")
    for arg in required_args:
        if arg not in distance_args.keys():
            raise ValueError(f"<distance_args> must contain <{arg}> key.")

    result: float = 0.0
    categories_distances: Dict[Tuple[str, str], float] = distance_args[
        "categories_distances"
    ]

    result += (
        euclidean_distance(cluster1["centroid"], cluster2["centroid"])
        * distance_args["centroid"]
    )

    cluster1_categories = get_most_common_cluster_categories(cluster1["categories"])
    cluster2_categories = get_most_common_cluster_categories(cluster2["categories"])

    for c1 in cluster1_categories:
        for c2 in cluster2_categories:
            key = tuple(sorted([c1[0], c2[0]]))
            if key in categories_distances:
                if categories_alpha:
                    result += (
                        categories_distances[key]
                        * distance_args["categories"]
                        * c1[1]
                        * c2[1]
                    )
                else:
                    result += categories_distances[key] * distance_args["categories"]

    if result > 1:
        result = 1.0
    if result < 0:
        result = 0.0

    return result


def create_communities(
    clusters: pd.DataFrame,
    eps: float = 0.216,
    distance_args: Dict[str, float] = {
        "centroid": 0.75,
        "categories": 0.25,
    },
    categories_alpha: bool = True,
    dbscan_min_samples: int = 2,
    dbscan_parallel_procs: int = -1,
) -> Iterable[int]:
    """
    Creates communities from clusters.

    Args:
        clusters (pd.DataFrame): dataframe containing clusters data
        eps (float, optional): distance used in DBSCAN clustering of clusters. Defaults to 0.216.
        distance_args (_type_, optional): arguments used in cluster_distance function.
            Defaults to { "centroid": 0.75, "categories": 0.25, }.
        categories_alpha (bool, optional): _description_. Defaults to True.
        dbscan_min_samples (int, optional): minimum number of samples in a community. Defaults to 2.
        dbscan_parallel_procs (int, optional): number of processes to use for dbscan clustering.
            Defaults to -1, which means all available processes.

    Returns:
        Iterable[int]: list of communities labels.
    """
    clustes_centroid = [
        {
            "centroid": np.array(cluster["centroid_embedding"]),
            "categories": cluster["categories"],
        }
        for _, cluster in clusters.iterrows()
    ]

    num_docs = len(clustes_centroid)
    dist_matrix: np.ndarray = np.zeros((num_docs, num_docs))

    for i in range(num_docs):
        for j in range(num_docs):
            if i == j:
                dist_matrix[i][j] = 0.0
            else:
                dist_matrix[i][j] = cluster_distance(
                    clustes_centroid[i],
                    clustes_centroid[j],
                    distance_args,
                    categories_alpha,
                )

    dbscan_args: Dict[str, Any] = {
        "eps": eps,
        "min_samples": dbscan_min_samples,
        "metric": "precomputed",
        "n_jobs": dbscan_parallel_procs,
    }

    dbscan = DBSCAN(**dbscan_args).fit(dist_matrix)
    labels = dbscan.labels_

    return labels


def split_big_communities(
    clusters: pd.DataFrame,
    categories_distances: Dict[Tuple[str, str], float],
    ref_date: datetime,
) -> None:
    """
    Split big communities in smaller ones

    Args:
        clusters (pd.DataFrame): dataframe containing clusters data
        categories_distances (Dict[Tuple[str, str], float]): dictionary containing the distances between categories.
        ref_date (datetime): date used to filter data.
    """
    communities = clusters.loc[~clusters["community_id"].str.endswith("-1")]

    if len(communities) == 0:
        return

    community_sizes: List[Tuple[str, int]] = []
    for community in communities.community_id.unique():
        if community.endswith("-1"):
            continue
        community_sizes.append(
            (
                community,
                communities.loc[communities["community_id"] == community]["size"].sum(),
            )
        )

    sizes = [community[1] for community in community_sizes]
    cluster_mean = np.mean(sizes)
    cluster_std = np.std(sizes)

    threshold = cluster_mean + 2 * cluster_std
    outlier_indices = np.where(sizes > threshold)[0]

    outliers = [community_sizes[idx][0] for idx in outlier_indices]

    for outlier in outliers:
        community = communities.loc[communities["community_id"] == outlier]

        sub_communities = create_communities(
            clusters=community,
            eps=0.15,
            distance_args={
                "centroid": 0.8,
                "categories": 0.2,
                "categories_distances": categories_distances,
            },
            categories_alpha=False,
        )
        n_communities: int = clusters.community_id.nunique()
        labels_mapper: Dict[int, str] = {}

        for label in set(sub_communities):
            if label != -1:
                new_label = label + n_communities
            else:
                new_label = label

            labels_mapper[label] = cluster_label(new_label, ref_date)

        clusters.loc[clusters["community_id"] == outlier, "community_id"] = [
            labels_mapper[label] for label in sub_communities
        ]


def test_cluster_distance(
    cluster: Dict[str, Any],
    communities: List[Dict[str, Any]],
    categories_distances: Dict[Tuple[str, str], float],
) -> Optional[str]:
    """
    Function used to calculate the distance between clusters.

    Args:
        cluster (Dict[str, Any]): dictionary containing clusters data.
        communities (List[Dict[str, Any]]): list of communities.
        categories_distances (Dict[Tuple[str, str], float]): dictionary containing the distances
            between categories.

    Returns:
        Optional[str]: community id of the closest community.
    """
    distances = [
        {
            "distance_args": {
                "centroid": 0.75,
                "categories": 0.25,
                "categories_distances": categories_distances,
            },
            "categories_alpha": True,
            "threshold": 0.216,
        },
        {
            "distance_args": {
                "centroid": 0.8,
                "categories": 0.2,
                "categories_distances": categories_distances,
            },
            "categories_alpha": False,
            "threshold": 0.15,
        },
    ]

    results: List[Tuple[str, float]] = []
    for distance in distances:
        for community in communities:
            candidate = cluster_distance(
                cluster1=cluster,
                cluster2=community,
                distance_args=distance["distance_args"],
                categories_alpha=distance["categories_alpha"],
            )

            if candidate < distance["threshold"]:
                results.append((community["community_id"], candidate))

    if len(results):
        candidate_community = sorted(results, key=lambda x: x[1], reverse=False)[0][0]

        return candidate_community
    return None
