# Description: Function needed to the DBSCAN clusterization flow.
from typing import List, Dict, Tuple, Union, Callable, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from loguru._logger import Logger
from sklearn.cluster import DBSCAN

from utils.distances import custom_distance, euclidean_distance
from utils import cluster_label


DBSACAN_CLUSTERING_OUTPUT_TYPE = Tuple[
    Dict[str, List[np.int64]],  # doc to clusters
    Dict[np.int64, List[np.float64]],  # centroids
    Dict[np.int64, Dict[str, any]],  # central documents
    Dict[
        np.int64, Dict[int, Union[Tuple[dict, float], Tuple[dict, float], float]]
    ],  # cluster borders
]


def dbscan_clustering(
    documents: List[Dict[str, any]],
    eps: float,
    min_samples: int,
    categories_distances: dict,
    embedding_column: str = "content_embedding",
    distance: Callable[[any], float] = custom_distance,
    distance_args: Dict[str, float] = {"categories": 0.1, "title": 0.2, "content": 0.7},
    num_processes: int = -1,
) -> Tuple[
    Dict[str, List[np.int64]],  # doc to clusters
    Dict[np.int64, List[np.float64]],  # centroids
    Dict[np.int64, Dict[str, any]],  # central documents
    Dict[
        np.int64, Dict[int, Union[Tuple[dict, float], Tuple[dict, float], float]]
    ],  # cluster borders
]:
    """
    Cluster documents using DBSCAN using a custom distance function.

    Args:
        documents (List[Dict[str, any]]): list of documents.
        eps (float): epsilon parameter of DBSCAN.
        min_samples (int): minimum number of samples parameter of DBSCAN.
        categories_distances (dict): dictionary containing the distance between all categories.
        embedding_column (str, optional): Column containing
        distance (Callable[[any], float], optional): function used to calculate the distance between
            two documents. Defaults to custom_distance.
        distance_args (Dict[str, float], optional): arguments used by the distance function.
            Defaults to {"categories": 0.1, "title": 0.2, "content": 0.7}.
        num_processes (int, optional): Number of processes used by DBSCAN.
            Defaults to -1 (all available processes).

    Returns:
        Tuple[
            Dict[str, List[np.int64]],  # doc to clusters
            Dict[np.int64, List[np.float64]],  # centroids
            Dict[np.int64, Dict[str, any]],  # central documents
            Dict[
                np.int64, Dict[int, Union[Tuple[dict, float], Tuple[dict, float], float]]
            ],  # cluster borders
        ]: returns a tuple (python convention) containing:
            - doc_to_clusters: dictionary containing the cluster of each document.
            - centroids: dictionary containing the centroid of each cluster.
            - central_documents: dictionary containing the central document of each cluster.
            - cluster_borders: dictionary containing the "border" of each cluster, i.e.
                the document that is the closest and the farthest, and the average distance of
                the documents to the centroid.
    """
    embeddings: List[dict] = [
        {
            "title": np.array(doc["title_embedding"]),
            "content": np.array(doc["content_embedding"]),
            "categories": doc["categories"],
        }
        for doc in documents
    ]

    distance_args["categories_distances"] = categories_distances

    num_docs: int = len(embeddings)
    dist_matrix: np.array = np.zeros((num_docs, num_docs))

    for i in range(num_docs):
        for j in range(num_docs):
            dist_matrix[i, j] = distance(embeddings[i], embeddings[j], **distance_args)

    db = DBSCAN(
        eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=num_processes
    ).fit(dist_matrix)
    labels: List[str] = db.labels_

    clusters: Dict[str, List[str]] = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(documents[idx]["url"])

    doc_to_clusters: Dict[str, List[str]] = {doc["url"]: [] for doc in documents}
    for label, urls in clusters.items():
        for url in urls:
            doc_to_clusters[url].append(label)

    centroids = get_clusters_centroids(documents, doc_to_clusters, embedding_column)

    central_documents: Dict[str, dict] = {}
    cluster_borders: Dict[str, list] = {}
    for label, docs in clusters.items():
        central_doc = get_central_document(
            documents, centroids[label], embedding_column
        )
        central_documents[label] = central_doc

        cluster_docs = [doc for doc in documents if doc["url"] in docs]

        cluster_borders[label] = get_cluster_borders(
            cluster_docs, embedding_column, central_doc, categories_distances
        )

    return doc_to_clusters, centroids, central_documents, cluster_borders


def get_clusters_centroids(
    documents: List[Dict[str, any]],
    doc_to_clusters: Dict[str, List[str]],
    embedding_column="content_embedding",
) -> Dict[str, np.array]:
    """
    Get the centroid of each cluster.

    Args:
        documents (List[Dict[str, any]]): list of documents.
        doc_to_clusters (Dict[str, List[str]]): dictionary containing the cluster of each document.
        embedding_column (str, optional): Column containing the embedding that will
            be used to calculate the centroid. Defaults to "content_embedding".

    Returns:
        Dict[str, np.array]: dictionary containing the centroid of each cluster.
    """
    clusters_embeddings: Dict[str, np.array] = {}
    for url, labels in doc_to_clusters.items():
        embedding: np.array = np.array(
            [doc[embedding_column] for doc in documents if doc["url"] == url][0]
        )
        for label in labels:
            if label not in clusters_embeddings:
                clusters_embeddings[label] = []
            clusters_embeddings[label].append(embedding)

    centroids: Dict[str, np.array] = {}
    for label, embeddings in clusters_embeddings.items():
        centroids[label] = np.mean(embeddings, axis=0)

    return centroids


def get_central_document(
    cluster_documents: List[dict],
    centroid_embedding: np.array,
    embedding_column: str = "content_embedding",
) -> Dict[str, any]:
    """
    Get the central document of a cluster.

    Args:
        cluster_documents (List[dict]): list of documents that belong to the same cluster.
        centroid_embedding (np.array): centroid of the cluster.
        embedding_column (str, optional): embedding that will be used to calculate the distance.
            Defaults to "content_embedding".

    Returns:
        Dict[str, any]: central document of the cluster.
    """

    min_distance = float("inf")
    central_document: Optional[Dict[str, any]] = None

    for doc in cluster_documents:
        dist: float = euclidean_distance(doc[embedding_column], centroid_embedding)
        if dist < min_distance:
            min_distance: float = dist
            central_document: dict = doc
    return central_document


def get_cluster_borders(
    documents: List[dict],
    embedding_col: str,
    cluster_central_document: dict,
    categories_distances: dict,
) -> Dict[int, Union[Tuple[dict, float], Tuple[dict, float], float]]:
    """
    Get the border documents of a cluster.

    Args:
        documents (List[dict]): list of documents that belong to the same cluster.
        embedding_col (str): embedding that will be used to calculate the distance.
        cluster_central_document (dict): central document of the cluster.

    Returns:
        Dict[int, Union[Tuple[dict, float], Tuple[dict, float], float]]: dictionary containing
            the "border" of the cluster, i.e. the document that is the closest and the farthest,
            and the average distance of the documents to the centroid.
    """

    distances_args: List[Dict[str, float]] = [
        {
            "content": 0.1,
            "title": 0.8,
            "categories": 0.1,
            "categories_distances": categories_distances,
        },
        {
            "content": 0.8,
            "categories": 0.2,
            "title": 0.0,
            "categories_distances": categories_distances,
        },
    ]

    doc_embeddings: List[dict] = [
        {
            "title": np.array(doc["title_embedding"]),
            "content": np.array(doc[embedding_col]),
            "categories": doc["categories"],
        }
        for doc in documents
    ]

    cluster_centroid_embedding: Dict[str, Union[np.array, List[str]]] = {
        "title": np.array(cluster_central_document["title_embedding"]),
        "content": np.array(cluster_central_document[embedding_col]),
        "categories": cluster_central_document["categories"],
    }

    res: Dict[int, dict] = {}

    for idx, distance_args in enumerate(distances_args):
        distances = [
            custom_distance(cluster_centroid_embedding, doc, **distance_args)
            for doc in doc_embeddings
        ]

        max_distance: float = max(distances)
        max_distance_doc: dict = documents[distances.index(max_distance)]

        min_distance: float = min(distances)
        min_distance_doc: dict = documents[distances.index(min_distance)]

        avg_distance: float = np.mean(distances)

        res[idx] = {
            "min": (min_distance_doc, min_distance),
            "max": (max_distance_doc, max_distance),
            "avg": avg_distance,
        }

    return res


# big clusters -----------------------------------------------------------------


def split_big_clusters(
    documents: pd.DataFrame,
    clusters: pd.DataFrame,
    embedding_column: str = "content_embedding",
    categories_distance: dict = {},
    logger: Logger = None,
    ref_day: datetime = datetime.now(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split big clusters using the appropriate distance function.

    Args:
        documents (pd.DataFrame): dataframe containing documents.
        clusters (pd.DataFrame): dataframe containing clusters.
        embedding_column (str, optional): column containing embeddings. Defaults to "content_embedding".
        categories_distance (dict, optional): dictionary containing the distance between all categories.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: returns a tuple (python convention) containing:
            - documents: dataframe containing documents.
            - clusters: dataframe containing clusters.
    """

    if len(categories_distance) == 0:
        raise ValueError("categories_distance must be a non-empty dict")

    new_clusters = []
    distances: Dict[str, Union[dict, float]] = [
        {
            "args": {
                "content": 0.1,
                "title": 0.8,
                "categories": 0.1,
            },
            "threshold": 0.53,
        },
        {
            "args": {
                "content": 0.8,
                "categories": 0.2,
                "title": 0.0,
            },
            "threshold": 0.59,
        },
    ]

    for cluster in documents.cluster.unique():
        if cluster.endswith("-1"):
            continue
        cluster_docs = documents[documents.cluster == cluster]
        if not len(cluster_docs):
            continue

        candidate_distance: Optional[dict] = None

        most_frequent_value = lambda x: x.value_counts().index[0]
        if most_frequent_value(cluster_docs["first_layer"]):
            candidate_distance = distances[0]

        if most_frequent_value(cluster_docs["second_layer"]):
            candidate_distance = distances[1]

        if candidate_distance is None:
            continue

        docs = cluster_docs.to_dict(orient="records")
        (
            docs_to_cluster,
            centroids,
            central_documents,
            cluster_border,
        ) = dbscan_clustering(
            documents=docs,
            eps=candidate_distance["threshold"],
            min_samples=2,
            embedding_column=embedding_column,
            distance_args=candidate_distance["args"],
            categories_distances=categories_distance,
        )

        detected_clusters = list(central_documents.keys())

        # no variation
        if len(detected_clusters) == 1 and detected_clusters[0] == 0:
            continue

        message = f"cluster: {cluster} splitted in {len(detected_clusters)} clusters"
        if logger:
            logger.debug(message)
        else:
            print(message)

        # cluster splitted
        cluster_mapper = {}
        for new_cluster in detected_clusters:
            cluster_id = cluster_label(cluster, ref_day)
            cluster_mapper[new_cluster] = cluster_id
            new_clusters.append(
                {
                    "id": cluster_id,
                    "label": cluster,
                    "timestamp": None,
                    "centroid_embedding": np.array(centroids[new_cluster]),
                    "central_doc": central_documents[new_cluster],
                    "cluster_border": cluster_border[new_cluster],
                }
            )

        documents["cluster"] = documents.apply(
            lambda x: cluster_mapper[value[0]]
            if (value := docs_to_cluster.get(x["url"]))
            else x["cluster"],
            axis=1,
        )
    message = f"New clusters: {len(new_clusters)}"
    if logger:
        logger.debug(message)
    else:
        print(message)
    clusters = pd.concat([clusters, pd.DataFrame(new_clusters)], ignore_index=True)

    return documents, clusters
