# Description: file containing functions needed to generate labels for clusters.
from typing import Tuple, Optional, Generator, Any, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from datetime import datetime
from time import sleep

import openai

from utils.backoff import retry_with_exponential_backoff

# OPENAI CONSTRAINS -----------------------------------------------------------------------
MAX_REQUESTS_PER_MINUTE = 60
REQUESTS_PER_CYCLE = 10
CYCLE_TIME = 60 / MAX_REQUESTS_PER_MINUTE

# cluster id -----------------------------------------------------------------------------


def cluster_label(cluster_label: int, d: datetime) -> str:
    """
    Generate a cluster label, used for <cluster_id> in the database.

    Args:
        cluster_label (int): numeric label of the cluster.
        d (datetime): date of the cluster.

    Returns:
        str: cluster label.
    """
    hex = sha256(
        f"{d.strftime('%Y-%m-%d')}_{cluster_label}".encode("utf-8")
    ).hexdigest()
    return f"{d.strftime('%Y-%m-%d')}_{hex}_{cluster_label}"


# cluster label -----------------------------------------------------------------------------
def generate_message(titles: str, max_length: int = 4097) -> str:
    """
    Generate a message to be used as input for GPT-3.

    Args:
        titles (str): list of titles of documents inside the cluster.
        max_length (int, optional): max length of the message on GPT3.5-turbo.
            Defaults to 4097.

    Returns:
        str: message to be used as input for GPT-3.
    """
    message: str = ""
    for title in titles:
        title_str = f"TITLE: {title}\n"
        if len(message) + len(title_str) <= max_length:
            message += title_str
        else:
            break
    return message


@retry_with_exponential_backoff
def generate_cluster_label_gpt(
    cluster_id: str, message: str
) -> Tuple[str, str, List[str]]:
    """
    Generate a cluster label using GPT-3.

    Args:
        cluster_id (str): cluster id.
        message (str): message that will be sent to OpenAI API.

    Returns:
        Tuple[str, str, List[str]]: cluster id, cluster label, cluster tags.
    """

    SYSTEM_PROMPT_CLUSTER: str = """
    You are a text classification tool, your work is to create a label for a cluster of documents.
    You are given a list of documents title and you have to respond only with a list of keyword and a label for the cluster.
    The cluster label mustnt be a document title or generic like "news" or "articles".
    The output must follow this structure:
    - cluster label: <label>
    - cluster tags: <list of tags>
    """

    cluster_label: Optional[str] = None

    response = None
    while response is None:
        try:
            response: Generator[Any] = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_CLUSTER},
                    {"role": "user", "content": message},
                ],
                temperature=1,
                max_tokens=448,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

        except Exception as e:
            print(e)

    cluster_data: str = response.choices[0].message.content.split("\n")
    cluster_label = ""
    cluster_tags = []
    if len(cluster_data) > 1:
        cluster_label, cluster_tags = cluster_data[0], cluster_data[1]
    else:
        cluster_label = cluster_data[0]

    if "cluster label:" in cluster_label:
        cluster_label = cluster_label.replace("cluster label:", "").strip()
    if "Cluster label:" in cluster_label:
        cluster_label = cluster_label.replace("Cluster label:", "").strip()
    if cluster_label.startswith('"') and cluster_label.endswith('"'):
        cluster_label = cluster_label[1:-1].strip()
    if cluster_label.startswith("-"):
        cluster_label = cluster_label[1:].strip()

    if cluster_tags:
        if "cluster tags:" in cluster_tags:
            cluster_tags = cluster_tags.replace("cluster tags:", "").strip()
        if "Cluster tags:" in cluster_tags:
            cluster_tags = cluster_tags.replace("Cluster tags:", "").strip()
        if cluster_tags.startswith("-"):
            cluster_tags = cluster_tags[1:].strip()
        if cluster_tags.startswith('"') and cluster_tags.endswith('"'):
            cluster_tags = cluster_tags[1:-1].strip()
        if cluster_tags.startswith("[") and cluster_tags.endswith("]"):
            cluster_tags = cluster_tags[1:-1].strip()
        cluster_tags = [tag.strip() for tag in cluster_tags.split(",")]

    return cluster_id, cluster_label, cluster_tags


def generate_cluster_label_backoff(
    cluster_data: List[Dict[str, any]]
) -> List[Dict[str, any]]:
    for cluster in cluster_data:
        if cluster["cluster_id"].endswith("-1"):
            cluster["cluster_label"] = ""
            cluster["tags"] = []
        else:
            message = generate_message(cluster["documents_titles"])
            response = generate_cluster_label_gpt(cluster["cluster_id"], message)
            cluster["cluster_label"] = response[1]
            cluster["tags"] = response[2]
        del cluster["documents_titles"]

    return cluster_data


def generate_cluster_labels_multiprocessing(
    cluster_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Generate cluster labels using multiprocessing.

    Args:
        cluster_data (List[Dict[str, Any]]): list of cluster data.

    Returns:
        List[Dict[str, Any]]: list of cluster data with labels.
    """
    responses: Dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                generate_cluster_label_gpt,
                cluster["cluster_id"],
                cluster["documents_titles"],
            )
            for cluster in cluster_data
        ]

        for future in as_completed(futures):
            try:
                result = future.result()
                sleep(1)
            except Exception as e:
                print(e)
                result = None

            if result:
                responses[result[0]] = {"label": result[1], "tags": result[2]}

    for cluster in cluster_data:
        if cluster["cluster_id"] in responses:
            cluster["cluster_label"] = responses[cluster["cluster_id"]]["label"]
            cluster["tags"] = responses[cluster["cluster_id"]]["tags"]
            del cluster["documents_titles"]
    return cluster_data


def generate_community_message(
    cluster_labels, community_titles: List[str], community_tags: List[str]
) -> str:
    """
    Generate a message to be used as input for GPT-3.

    Args:
        community_titles (List[str]): list of titles of documents inside the community.
        community_tags (List[str]): list of tags of documents inside the community.

    Returns:
        str: message to be used as input for GPT-3.
    """
    message: str = ""
    labels_count = 0
    for label in cluster_labels:
        if labels_count == 30:
            break
        label_str = f"LABEL: {label}\n"
        if len(message) + len(label_str) <= 4097:
            message += label_str
        else:
            break
        labels_count += 1

    title_count = 0
    for title in community_titles:
        if title_count == 30:
            break
        title_str = f"TITLE: {title}\n"
        if len(message) + len(title_str) <= 4097:
            message += title_str
        else:
            break
        title_count += 1

    for tag in set(community_tags):
        tag_str = f"TAG: {tag}\n"
        if len(message) + len(tag_str) <= 4097:
            message += tag_str
        else:
            break

    return message


@retry_with_exponential_backoff
def generate_community_label_gpt(
    cluster_labels: List[str], community_titles: List[str], community_tags: List[str]
) -> str:
    SYSTEM_PROMPT_COMMUNITY: str = """
    You are a cluster classification tool, your work is to create a label for a community of clusters.
    You are given a list of cluster labels, a list of titles of documents in the community and a list of tags associated to the community of cluster.
    Community label must be related to the data given to you and must be descriptive of the cluster.
    You have to create label that sounds like news articles titles and must be relevant to the given data.
    You must answer only with the community label, in this way: 
    COMMUNITY: <label>
    """

    message = generate_community_message(
        cluster_labels, community_titles, community_tags
    )

    response = None
    while response is None:
        try:
            response: Generator[Any] = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_COMMUNITY},
                    {"role": "user", "content": message},
                ],
                temperature=0.3,
                max_tokens=448,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

        except Exception as e:
            print(e)

        community_label: str = response.choices[0].message.content

        if "COMMUNITY:" in community_label:
            community_label = community_label.replace("COMMUNITY:", "").strip()
        if "Community:" in community_label:
            community_label = community_label.replace("Community:", "").strip()
        if community_label.startswith("-"):
            community_label = community_label[1:].strip()
        if community_label.startswith('"') and community_label.endswith('"'):
            community_label = community_label[1:-1].strip()

    return community_label
