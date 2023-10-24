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


class ClusterCommunitiesTable:
    def __init__(
        self, database_name: str = "quseed", table_name: str = "cluster_communities"
    ) -> None:
        """
        Creates clusters table on DB.

        Args:
            database_name (str, optional): Database where table will be created. Defaults to "quseed".
            table_name (str, optional): Name of the table that will be created. Defaults to "cluster_communities".
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

        class ClusterCommunities(self.Base):
            __tablename__ = self.table_name

            id = Column(
                Integer,
                Sequence("cluster_communities_id_seq", start=1),
                primary_key=True,
            )
            cluster_id = Column(String(250), nullable=False)
            community_id = Column(String(500), nullable=True)
            timestamp = Column(Date, nullable=False)
            unclusterized = Column(Boolean, nullable=False, default=False)

            __table_args__ = (
                UniqueConstraint(
                    "cluster_id", name=f"uix_{self.table_name}_cluster_id"
                ),
                Index(f"idx_{self.table_name}_cluster_id", "cluster_id"),
                Index(f"idx_{self.table_name}_community_id", "community_id"),
                Index(f"idx_{self.table_name}_timestamp", "timestamp"),
                Index(f"idx_{self.table_name}_unclusterized", "unclusterized"),
            )

            def __repr__(self):
                return f"<ClusterCommunities(id={self.id}, cluster_id={self.cluster_id}, community_id={self.community_id}, timestamp={self.timestamp}, unclusterized={self.unclusterized})>"

        self.ClusterCommunities = ClusterCommunities

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
                existing_record = (
                    self.session.query(self.ClusterCommunities)
                    .filter_by(cluster_id=row_data["cluster_id"])
                    .first()
                )

                if existing_record:
                    for key, value in row_data.items():
                        setattr(existing_record, key, value)

                else:
                    new_record = self.ClusterCommunities(**row_data)
                    self.session.add(new_record)

            self.session.commit()
        except IntegrityError as e:
            self.session.rollback()
            raise e
        finally:
            self.close_db_session()

    def delete_by_cluster_id(self, cluster_id_list: List[str]) -> None:
        """
        Delete rows from the table by cluster_id.

        Args:
            cluster_id_list (List[str]): List of cluster_ids to delete.
        """
        try:
            self.init_db_session()
            for cluster_id in cluster_id_list:
                self.session.execute(
                    text(
                        f"delete from {self.table_name} where cluster_id = :cluster_id"
                    ),
                    {"cluster_id": cluster_id},
                )
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e
        finally:
            self.close_db_session()
