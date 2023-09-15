# Description: This file contains custom file types and related functions.

from typing import Union
from datetime import datetime, date

generic_date = Union[datetime, date, str]


def parse_generic_date(date_value: generic_date) -> datetime:
    """
    Parses a generic date into a datetime object.

    Args:
        date_value (generic_date): date to parse

    Raises:
        TypeError: if <date_value> is not a datetime, :date or :str

    Returns:
        datetime: parsed datetime object
    """
    if isinstance(date_value, datetime):
        return date_value
    elif isinstance(date_value, date):
        return datetime.combine(date_value, datetime.min.time())
    elif isinstance(date_value, str):
        return datetime.strptime(date_value, "%Y-%m-%d")
    else:
        raise TypeError(
            f"<date_value> must be datetime, :date or :str, not {type(date_value)}"
        )
