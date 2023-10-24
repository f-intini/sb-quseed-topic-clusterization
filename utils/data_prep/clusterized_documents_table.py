# Description: Contains code needed to create clusterized documents on DB.
import sys, os

sys.path.append(os.getcwd())
from typing import List, Dict, Any

from sqlalchemy import (
    Column,
    Sequence,
    Integer,
    String,
    UniqueConstraint,
    Index,
    inspect,
    text,
    Date,
    Boolean,
)
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError

from sb_py_fw.database.postgres import create_pg_engine

from utils.const import ALLOWED_DBS as allowed_dbs


class ClusterizedDocumentsTable:
    def __init__(
        self, database_name: str = "quseed", table_name: str = "clusterized_docs"
    ) -> None:
        """
        Creates clusters table on DB.

        Args:
            database_name (str, optional): Database where table will be created. Defaults to "quseed".
            table_name (str, optional): Name of the table that will be created. Defaults to "clusters".
        """
        if not database_name in allowed_dbs.keys():
            raise ValueError(
                f"<database_name> must be one of {allowed_dbs.keys()}, not {database_name}"
            )
        self.table_name = table_name
        self.engine = create_pg_engine(section=allowed_dbs[database_name])
        self.Base = declarative_base()
        self.__set_table_definitions__()
        self.session_factory = sessionmaker(bind=self.engine)

    def __set_table_definitions__(self) -> None:
        """
        Sets table (<self.table_name>) definition.
        """

        class ClusterizedDocuments(self.Base):
            __tablename__ = self.table_name

            id = Column(
                Integer, Sequence("clusterized_docs_id_seq", start=1), primary_key=True
            )
            url = Column(String(500), nullable=False)
            cluster_id = Column(String(250), nullable=True)
            timestamp = Column(Date, nullable=False)
            unclusterized = Column(Boolean, nullable=False, default=False)

            __table_args__ = (
                UniqueConstraint("url", name=f"uix_{self.table_name}_url"),
                Index(f"idx_{self.table_name}_url", "url"),
                Index(f"idx_{self.table_name}_cluster_id", "cluster_id"),
                Index(f"idx_{self.table_name}_timestamp", "timestamp"),
                Index(f"idx_{self.table_name}_unclusterized", "unclusterized"),
            )

            def __repr__(self) -> str:
                return f"<ClusterizedDocuments(url={self.url}, cluster_id={self.cluster_id}, timestamp={self.timestamp}, unclusterized={self.unclusterized})>"

        self.ClusterizedDocuments = ClusterizedDocuments

    def table_exists(self) -> bool:
        """
        Checks if the table <self.table_name> exists in the database.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        return inspect(self.engine).has_table(self.table_name)

    def init_db_session(self) -> None:
        """
        Initializes a database session.
        """
        self.session = self.session_factory()

    def close_db_session(self) -> None:
        """
        Closes the database session.
        """
        if hasattr(self, "session"):
            self.session.close()

    def create_table(self, reset: bool = False) -> None:
        """
        Creates the table <self.table_name> on the database.

        Args:
            reset (bool, optional): Flag variable, set to true to delete the existing table. Defaults to False.

        Raises:
            e: if the table cannot be created
            e: if the table cannot be dropped (only if reset=True)
        """

        if self.table_exists() and reset:
            try:
                self.init_db_session()
                self.session.execute(text(f"drop table {self.table_name} cascade;"))
                self.session.commit()
                self.close_db_session()
                print(f"Table {self.table_name} dropped.")

                print(f"Recreating table {self.table_name}.")
                self.Base.metadata.create_all(self.engine)  # type: ignore
                print(f"Table {self.table_name} created.")
            except Exception as e:
                print(e)
                raise e
        else:
            try:
                print(f"Creating table: {self.table_name}.")
                self.Base.metadata.create_all(self.engine)
                print(f"Table {self.table_name} created.")
            except Exception as e:
                print(f"Error creating table: {self.table_name}.")
                print(e)
                raise e

    def init(self, reset: bool = False) -> None:
        """
        Creates the table <self.table_name> on the database.

        Args:
            reset (bool, optional): Flag variable, set to true to delete the existing table. Defaults to False.
        """
        self.create_table(reset=reset)

    def upsert_rows(self, rows: List[Dict[str, Any]]) -> None:
        """
        Upsert rows into the table.

        Args:
            rows (List[Dict[str, Any]]): List of dictionaries representing row data.

        Raises:
            IntegrityError: If there's an issue with the integrity of the database.
        """
        try:
            self.init_db_session()
            for row_data in rows:
                # find duplicates
                existing_record = (
                    self.session.query(self.ClusterizedDocuments)
                    .filter_by(url=row_data["url"])
                    .first()
                )

                if existing_record:
                    # update
                    existing_record.cluster_id = row_data.get("cluster_id", None)
                    existing_record.timestamp = row_data["timestamp"]
                    existing_record.unclusterized = row_data["unclusterized"]
                else:
                    # create
                    new_record = self.ClusterizedDocuments(**row_data)
                    self.session.add(new_record)

            self.session.commit()
        except IntegrityError as e:
            self.session.rollback()
            raise e
        finally:
            self.close_db_session()
