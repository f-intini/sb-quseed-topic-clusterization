# Description: Distance functions for vectors.
from typing import Dict, List, Union

import numpy as np
from sklearn.metrics.pairwise import manhattan_distances


def euclidean_distance(x: np.array, y: np.array) -> float:
    """Euclidean distance between two points."""
    return np.linalg.norm(x - y)


def cosine_distance(x: np.array, y: np.array) -> float:
    """Cosine distance between two points."""
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def cosine_similarity(x: np.array, y: np.array) -> float:
    """Cosine similarity between two points."""
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def manhattan_distance(x: np.array, y: np.array) -> float:
    """Manhattan distance between two points."""
    return manhattan_distances(x, y)[0][0]


def custom_distance(
    x: Dict[str, Union[np.array, List[str]]],
    y: Dict[str, Union[np.array, List[str]]],
    categories: float = 0.1,
    content: float = 0.7,
    title: float = 0.2,
    categories_distances: dict = {},
) -> float:
    """
    Custom distance function between two documents. (Used in document clusterization)

    Args:
        x (Dict[str, Union[np.array, List[str]]]): dictionary containing information of a document,
            must have keys 'vector', 'categories' and 'title'
        y (Dict[str, Union[np.array, List[str]]]): dictionary containing information of a document,
            must have keys 'vector', 'categories' and 'title'
        categories (float, optional): Categories score weight. Defaults to 0.1.
        content (float, optional): Content score weight. Defaults to 0.7.
        title (float, optional): Title score weight. Defaults to 0.2.
        categories_distances (dict, optional): Dictionary containing the distance between all categories.

    Raises:
        ValueError: <x> and <y> must be dictionaries
        ValueError: <x> and <y> must have keys 'vector', 'categories' and 'title'
        ValueError: Categories distances not found, if relying of default values,
            please make sure the table exists in the database.

    Returns:
        float: _description_
    """
    for obj in [x, y]:
        if not isinstance(obj, dict):
            raise ValueError(
                f"<x> and <y> must be dictionaries, not {type(obj)} and {type(obj)}"
            )

        if not all([key in obj.keys() for key in ["vector", "categories", "title"]]):
            raise ValueError(
                f"<x> and <y> must have keys 'vector', 'categories' and 'title'"
            )

    if not len(categories_distances.keys()):
        raise ValueError(
            "Categories distances not found, if relying of default values, "
            + "please make sure the table exists in the database."
        )

    result: float = 0.0
    result = content * euclidean_distance(x["vector"], y["vector"])

    categories_distance: float = 0.0
    x_categories = [cat for cat in x["categories"] if cat != "unknown"]
    y_categories = [cat for cat in y["categories"] if cat != "unknown"]

    if len(x_categories) and len(y_categories):
        for c1 in x_categories:
            for c2 in y_categories:
                key = tuple(sorted([c1, c2]))
                categories_distance += categories_distances.get(key, 1.0)
        categories_distance /= len(x_categories) * len(y_categories)
    else:
        categories_distance = 1.0

    categories_distance = (
        1.0
        if categories_distance >= 1.0
        else categories_distance
        if categories_distance > 0.0
        else 0.0
    )

    result += categories * categories_distance

    title_distance = euclidean_distance(x["title"], y["title"])
    result += title * title_distance

    return result
