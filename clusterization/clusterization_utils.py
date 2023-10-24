# Description: Function needed to the clusterization flow.
from typing import List, Dict, Tuple, Union, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import numpy as np
from loguru._logger import Logger

from .dbscan_utils import get_central_document, get_cluster_borders
from utils.distances import custom_distance

# NOTE: qui modificare affinchè venga usato il content/summary embedding


def test_document_distance(
    document_id: str,  # document url
    embedding: dict,  # document embedding
    clusters: Dict[str, np.array],
    categories_distances: Dict[Tuple[str, str], float],
) -> Optional[Tuple[str, str]]:
    """
    Test if a document belongs to a cluster.

    Args:
        document_id (str): Document url.
        embedding (dict): Document embedding in dictionary structured as follow:
            - title(np.array): Document title embedding.
            - content(np.array): Document content embedding
            - categories(List[str]): Document categories

    Returns:
        Optional[Tuple[str, str]]: Tuple containing document url and cluster if the document belongs to a cluster,
    """
    matched_clusters = []
    distances = [
        {
            "distance": custom_distance,
            "args": {
                "content": 0.1,
                "title": 0.8,
                "categories": 0.1,
                "categories_distances": categories_distances,
            },
            "distance_threshold": 0.53,
        },
        {
            "distance": custom_distance,
            "args": {
                "content": 0.8,
                "categories": 0.2,
                "title": 0.0,
                "categories_distances": categories_distances,
            },
            "distance_threshold": 0.59,
        },
    ]

    for distance in distances:
        for cluster, cluster_embedding in clusters.items():
            cluster_distance = distance["distance"](
                embedding, cluster_embedding["central_doc"], **distance["args"]
            )

            if cluster_distance < (
                cluster_embedding.get("cluster_border", {}).get("max", [0, -1])[1]
            ):
                matched_clusters.append(
                    {
                        "cluster_id": cluster,
                        "distance": cluster_distance,
                        "method": "border",
                    }
                )

            if cluster_distance < distance["distance_threshold"]:
                matched_clusters.append(
                    {
                        "cluster_id": cluster,
                        "distance": cluster_distance,
                        "method": "regular",
                    }
                )

    if matched_clusters:
        candidate_cluster = sorted(
            matched_clusters, key=lambda x: x["distance"], reverse=True
        )[0]

        return document_id, candidate_cluster
    return None


def append_documents_to_existing_clusters(
    documents: pd.DataFrame,
    embedding_column: "str",
    clusters: pd.DataFrame,
    categories_distances: Dict[Tuple[str, str], float],
    logger: Logger,
) -> Dict[str, dict]:
    """
    Append documents to existing clusters, runs in multithreading.

    Args:
        documents (pd.DataFrame): dataframe containing documents
        embedding_column (str): column name of the content embedding to be used.
        clusters (pd.DataFrame): dataframe containing clusters data
        logger (Logger): logger object

    Returns:
        Dict[str, dict]: Dictionary containing the document url as key and the cluster as value.
    """
    thread_count: int = 5

    central_doc_data: Callable[[dict], dict] = lambda x: {
        "title": np.array(x["title_embedding"]),
        "content": np.array(x[embedding_column]),
        "categories": x["categories"],
    }

    clusters_dict: Dict[str, dict] = {
        row["id"]: {
            "central_doc": central_doc_data(row["central_doc"]),
            "cluster_border": row["cluster_border"],
        }
        for _, row in clusters.iterrows()
    }

    doc_to_cluster: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = [
            executor.submit(
                test_document_distance,
                doc["url"],
                {
                    "title": np.array(doc["title_embedding"]),
                    "content": np.array(doc[embedding_column]),
                    "categories": doc["categories"],
                },
                clusters_dict,
                categories_distances,
            )
            for _, doc in documents.iterrows()
        ]

        for future in as_completed(futures):
            try:
                test_result = future.result()
            except Exception as e:
                logger.warning(f"Error appending document to old cluster")
                logger.warning(e)
                test_result = None

            if isinstance(test_result, tuple):
                doc_to_cluster[test_result[0][0]] = test_result[1]

    return doc_to_cluster


# clusters stuff -----------------------------------------------------------------------------------------------
def generate_cluster_data(
    cluster_id: str,
    documents: pd.DataFrame,
    ref_date: datetime = None,
    embedding_column: str = "content_embedding",
    categories_distance: Dict[Tuple[str, str], float] = {},
) -> Optional[dict]:
    """
    Generate cluster data.

    Args:
        cluster_id (str): cluster id.
        documents (pd.DataFrame): dataframe containing documents.
        embedding_column (str, optional): column containing embeddings. Defaults to "content_embedding".
        categories_distance (Dict[Tuple[str, str], float], optional): dictionary containing categories distances.
            Defaults to {}.

    Raises:
        ValueError: if categories_distance is empty.

    Returns:
        dict: cluster data.
    """
    if len(categories_distance) == 0:
        raise ValueError("categories_distance must be a non-empty dict")

    cluster_data = {
        "id": cluster_id,
        "label": str(cluster_id).split("_")[-1] or "1",
        "timestamp": None,
        "centroid_embedding": None,
        "central_doc": None,
        "cluster_border": None,
    }

    cluster_documents = documents.loc[documents["cluster"] == cluster_id]
    cluster_embeddings = cluster_documents[embedding_column].values.tolist()
    cluster_data["centroid_embedding"]: np.array = np.mean(cluster_embeddings, axis=0)

    cluster_data["timestamp"] = cluster_documents.timestamp.min()
    cluster_documents = documents.to_dict(orient="records")

    if not len(cluster_documents):
        return None

    cluster_data["central_doc"]: dict = get_central_document(
        cluster_documents=cluster_documents,
        centroid_embedding=cluster_data["centroid_embedding"],
        embedding_column=embedding_column,
    )

    cluster_data["cluster_border"]: dict = get_cluster_borders(
        documents=cluster_documents,
        embedding_col=embedding_column,
        cluster_central_document=cluster_data["central_doc"],
        categories_distances=categories_distance,
    )

    return cluster_data


# ----------------------------------------------------------------------------------------------------------------


def pick_best_clustering(
    documents: pd.DataFrame,
    document_to_old_cluster: Dict[str, str],
    clusters: pd.DataFrame,
    categories_distances: Dict[Tuple[str, str], float],
) -> Dict[str, Union[None, str, dict]]:
    """
    Pick the best clustering.

    Args:
        documents (pd.DataFrame): dataframe containing documents.
        document_to_old_cluster (Dict[str, str]): dictionary containing document url as key and old cluster as value.
        clusters (pd.DataFrame): dataframe containing clusters data.

    Returns:
        Dict[str, Union[None, str, dict]]: dictionary containing document url as key and cluster as value.
    """
    cd = documents.to_dict(orient="records")
    clusters_dict = {row["id"]: row["central_doc"] for _, row in clusters.iterrows()}
    clusters_dict = {
        cluster: {
            "central_doc": {
                "title": np.array(central_doc["title_embedding"]),
                "content": np.array(central_doc["content_embedding"]),
                "categories": central_doc["categories"],
            },
        }
        for cluster, central_doc in clusters_dict.items()
    }

    pending_docs = {}
    for doc in cd:
        old_cluster = None
        new_cluster = None

        if doc["url"] in document_to_old_cluster:
            old_cluster = document_to_old_cluster[doc["url"]]

        if not doc["cluster"].endswith("-1"):
            new_cluster = doc["cluster"]

        if not old_cluster and not new_cluster:
            pending_docs[doc["url"]] = None
            continue

        if not new_cluster:
            pending_docs[doc["url"]] = {
                "label": old_cluster["cluster_id"],
                "new": False,
            }
            continue

        # calculate new cluster distance
        new_cluster_data = {new_cluster: clusters_dict[new_cluster]}

        new_cluster_distance = test_document_distance(
            doc["url"],
            {
                "title": np.array(doc["title_embedding"]),
                "content": np.array(doc["content_embedding"]),
                "categories": doc["categories"],
            },
            new_cluster_data,
            categories_distances,
        )

        if not old_cluster and not isinstance(old_cluster, str):
            pending_docs[doc["url"]] = list(new_cluster_data.keys())[0]
            continue

        if new_cluster_distance is not None:
            if new_cluster_distance[1]["distance"] < old_cluster["distance"]:
                pending_docs[doc["url"]] = {
                    "label": new_cluster_distance[1]["cluster_id"],
                    "new": True,
                }
        else:
            pending_docs[doc["url"]] = {
                "label": old_cluster["cluster_id"],
                "new": False,
            }
    return pending_docs
