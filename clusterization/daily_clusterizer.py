# NOTE: postponed evaluation of annotations
import sys
from datetime import datetime, date

from loguru import logger
import pandas as pd

from utils import (
    generic_date,
    pull_daily_documents,
    parse_generic_date,
    pull_categories_distances,
    escape_postgres_string,
)


class DailyClusters:
    def __init__(
        self,
        ref_day: generic_date = datetime.now(),
        log_to_file: bool = False,
        log_to_console: bool = True,
        database_name: str = "quseed",
        table_name: str = "categories_distances",
    ):
        self.__init_logger__(log_to_file, log_to_console)
        self.ref_day: datetime = self.__parse_ref_day__(ref_day)
        self.database_name: str = escape_postgres_string(database_name)
        self.table_name: str = escape_postgres_string(table_name)

    @logger.catch
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
        if log_to_console:
            logger.add(
                sys.stdout,
                format=(
                    "<green>{time}</green> <blue>[{level}]</blue> "
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
        if not isinstance(ref_day, generic_date):
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
            table_name=self.table_name,
            output_type="dict",
        )

        if categories_distances.shape[0] == 0:
            logger.warning("No categories distances found")
        else:
            logger.info(
                f"Pulled {categories_distances.shape[0]} categories distances from the database"
            )

        self.categories_distances = categories_distances
