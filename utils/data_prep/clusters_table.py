# Description: Contains code needed to create clusters table on DB.
import sys, os

sys.path.append(os.getcwd())
from typing import Iterable, Dict, List
from collections import Counter
import math

from sqlalchemy import (
    Column,
    String,
    Double,
    UniqueConstraint,
    Index,
    inspect,
    text,
    Date,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError

from sb_py_fw.database.postgres import create_pg_engine

from utils.const import ALLOWED_DBS as allowed_dbs


class ClustersTable:
    def __init__(
        self, database_name: str = "quseed", table_name: str = "clusters"
    ) -> None:
        """
        Creates clusters table on DB.

        Args:
            database_name (str, optional): Database where table will be created.
                Defaults to "quseed".
            table_name (str, optional): Name of the table that will be created.
                Defaults to "clusters".
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

        class Clusters(self.Base):
            __tablename__ = self.table_name

            uuid = Column(UUID(as_uuid=True), primary_key=True)
            cluster_id = Column(
                String(250)
            )  # label used during the clusterization process
            cluster_label = Column(String(150))
            timestamp = Column(Date)
            centroid_embedding = Column(ARRAY(Double))
            central_doc = Column(JSONB)
            cluster_borders = Column(ARRAY(JSONB))
            tags = Column(ARRAY(String(100)))
            categories = Column(ARRAY(String(100)))  # all the categorie of the cluster
            most_common_categories = Column(
                ARRAY(String(100))
            )  # the 20 most common categories of the cluster

            __table_args__ = (
                UniqueConstraint(
                    "cluster_id",
                    "cluster_label",
                    name="unique_cluster_id_cluster_label",
                ),
                Index("ix_clusters_uuid", "uuid"),
                Index("ix_clusters_cluster_id", "cluster_id"),
                Index("ix_clusters_cluster_label", "cluster_label"),
                Index("ix_clusters_timestamp", "timestamp"),
                Index("ix_clusters_most_common_categories", "most_common_categories"),
                Index("ix_clusters_tags", "tags"),
            )

            def __repr__(self) -> str:
                return f"<Clusters(id={self.id}, cluster_id={self.cluster_id}, cluster_label={self.cluster_label}, timestamp={self.timestamp}, centroid_embedding={self.centroid_embedding}, central_doc={self.central_doc}, cluster_border={self.cluster_border}, categories={self.categories}, most_common_categories={self.most_common_categories})>"  # noqa: E501

        self.Clusters = Clusters

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
            reset (bool, optional): Flag variable, set to true to delete the existing table.
                Defaults to False.

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

    def sanitize_input(self, data: dict) -> dict:
        if isinstance(data, dict):
            data_copy = data.copy()
            for chiave, valore in data.items():
                if isinstance(valore, (list, dict)):
                    data_copy[chiave] = self.sanitize_input(valore)
                elif valore is None or (
                    isinstance(valore, float) and math.isnan(valore)
                ):
                    del data_copy[chiave]
            return data_copy
        elif isinstance(data, list):
            return [self.sanitize_input(elem) for elem in data]
        else:
            return data

    def upsert_records(self, records: Iterable[dict]):
        """
        Upsert records into the Clusters table.

        Args:
            records (list[dict]): A list of dictionaries representing records to upsert.
        """
        self.init_db_session()
        try:
            for record in records:
                record = self.sanitize_input(record)
                existing_record = (
                    self.session.query(self.Clusters)
                    .filter_by(cluster_id=record["cluster_id"])
                    .first()
                )
                if existing_record:
                    for key, value in record.items():
                        setattr(existing_record, key, value)
                else:
                    new_record = self.Clusters(**record)
                    self.session.add(new_record)

            self.session.commit()
        except IntegrityError as e:
            self.session.rollback()
            print("IntegrityError:", e)
        except Exception as e:
            self.session.rollback()
            print("Error:", e)
        finally:
            self.close_db_session()

    def delete_clusters_by_uuids(self, uuid_list: List[str]) -> None:
        """
        Delete clusters by UUIDs.

        Args:
            uuid_list (List[str]): List of UUIDs to delete.

        Returns:
            None
        """
        if not uuid_list:
            print("No UUIDs provided for deletion.")
            return

        self.init_db_session()
        try:
            self.session.query(self.Clusters).filter(
                self.Clusters.uuid.in_(uuid_list)
            ).delete(synchronize_session=False)
            self.session.commit()
            print(f"{len(uuid_list)} cluster(s) deleted successfully.")
        except Exception as e:
            self.session.rollback()
            print("Error:", e)
        finally:
            self.close_db_session()
