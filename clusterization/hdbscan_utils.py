# Description: Contains code needed to clusterize documents using HDBSCAN.
from typing import List, Dict, Tuple, Union, Callable, Optional, Union
from datetime import datetime

import pandas as pd
import numpy as np
from umap import UMAP
import hdbscan
from loguru._logger import Logger

from utils.distances import custom_distance
from clusterization.dbscan_utils import (
    get_clusters_centroids,
    get_central_document,
    get_cluster_borders,
)
from utils import cluster_label

# things used for outlier clusters splitting ----------------------------


def umap_embeddings(
    documents: pd.DataFrame,
    n_components: int = 2,
    embedding_column: str = "content_embedding",
):
    """
    Embeds documents using UMAP.

    Args:
        documents (pd.DataFrame): dataframe containing documents.
        n_components (int, optional): number of components to reduce to. Defaults to 2.
        embedding_column (str, optional): column containing embeddings. Defaults to "content_embedding".

    Returns:
        _type_: embedding reduced using universal manifold approximation and projection.
    """
    embeddings: List[np.array] = documents[embedding_column].tolist()
    reducer = UMAP(n_components=n_components, n_jobs=-1, low_memory=True)
    embeddings = reducer.fit_transform(embeddings)  # type: ignore
    return embeddings


def hdbscan_clustering(
    documents: pd.DataFrame,
    min_cluster_size: int = 2,
    umap_n_components: int = 2,
    embedding_column: str = "content_embedding",
    distance: Callable[[dict, dict], float] = custom_distance,
):
    """
    Clusters documents using HDBSCAN.

    Args:
        documents (pd.DataFrame): dataframe containing documents.
        min_cluster_size (int, optional): minimum number of documents in a cluster. Defaults to 2.
        umap_n_components (int, optional): number of components to reduce to. Defaults to 2.
        embedding_column (str, optional): column containing embeddings. Defaults to "content_embedding".
        distance (Callable[[dict, dict], float], optional): distance function to use. Defaults to <custom_distance>.

    Returns:
        _type_: _description_
    """
    embeddings = umap_embeddings(
        documents=documents,
        n_components=umap_n_components,
        embedding_column=embedding_column,
    )
    docs_data: List[Dict[str, Union[np.array, List[str]]]] = [
        {
            "vector": np.array(embedding),
            "title": np.array(doc["ada_title_embedding"]),
            "categories": doc["categories"],
        }
        for embedding, doc in zip(embeddings, documents)
    ]
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed")
    distance_matrix = np.array(
        [[distance(e1, e2) for e2 in docs_data] for e1 in docs_data]
    )
    labels = clusterer.fit_predict(distance_matrix)

    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(documents[idx]["url"])

    doc_to_clusters = {doc["url"]: [] for doc in documents}
    for label, urls in clusters.items():
        for url in urls:
            doc_to_clusters[url].append(label)

    centroids = get_clusters_centroids(
        clusters=clusters,
        documents=documents,
        embedding_column=embedding_column,
    )

    cluster_borders = {}
    central_documents = {}
    for label, docs in clusters.items():
        cluster_embeddings = [
            doc[embedding_column] for doc in documents if doc["url"] in docs
        ]
        centroid_embedding = np.mean(cluster_embeddings, axis=0)
        central_doc = get_central_document(
            cluster_documents=documents,
            centroid_embedding=centroid_embedding,
            embedding_column=embedding_column,
        )
        central_documents[label] = central_doc
        cluster_borders[label] = get_cluster_borders(
            documents, embedding_column, central_doc
        )

    return doc_to_clusters, centroids, central_documents, cluster_borders


def outlier_clusters_splitting(
    documents: pd.DataFrame,
    embedding_col="summary_embedding",
    alpha: Union[int, float] = 2,
    logger: Logger = None,
    current_date: datetime = datetime.now(),
):
    new_clusters = []

    clusters = [
        (cluster, len(documents.loc[documents["cluster"] == cluster]))
        for cluster in documents.cluster.unique()
        if not cluster.endswith("_-1")
    ]
    cluster_sizes = [cluster[1] for cluster in clusters]
    data_mean = np.mean(cluster_sizes)
    data_std_dev = np.std(cluster_sizes)

    threshold = data_mean + alpha * data_std_dev

    outlier_indices = np.where(cluster_sizes > threshold)[0]

    outliers = [clusters[idx][0] for idx in outlier_indices if clusters[idx][1] > 10]

    if logger:
        logger.debug(f"Detected outliers: {len(outliers)}")
    else:
        print(f"Detected outliers: {len(outliers)}")

    for outlier in outliers:
        old_id = outlier.split("_")
        old_date, old_id = old_id[0], old_id[-1]

        outlier_df = documents.loc[documents["cluster"] == outlier]

        docs = outlier_df.to_dict("records")
        (
            docs_to_cluster,
            centroids,
            central_documents,
            cluster_borders,
        ) = hdbscan_clustering(docs, min_cluster_size=2, embedding_column=embedding_col)

        # adjust labels
        cluster_mapping = {}

        for id, cluster in enumerate(central_documents.keys()):
            if cluster == -1:
                cluster_id = -1
            else:
                cluster_id = len(clusters) + 1 + id
            cluster_id = cluster_label(cluster_id, current_date)
            cluster_mapping[cluster] = cluster_id

        centroids = {cluster_mapping[k]: v for k, v in centroids.items()}

        central_documents = {
            cluster_mapping[k]: v for k, v in central_documents.items()
        }

        docs_to_cluster = {
            doc: [cluster_mapping[cluster] for cluster in clusters]
            for doc, clusters in docs_to_cluster.items()
        }

        cluster_borders = {cluster_mapping[k]: v for k, v in cluster_borders.items()}

        # add new clusters
        for cluster in centroids.keys():
            new_clusters.append(
                {
                    "id": cluster,
                    "label": cluster,
                    "timestamp": current_date,
                    "centroid_embedding": np.array(centroids[cluster]),
                    "central_doc": central_documents[cluster],
                    "cluster_border": cluster_borders[cluster],
                }
            )

        if logger:
            logger.debug(f"New clusters: {len(new_clusters)}")

        # update documents
        for doc, clusters in docs_to_cluster.items():
            documents.loc[documents["url"] == doc, "cluster"] = clusters[0]

    return documents, new_clusters
