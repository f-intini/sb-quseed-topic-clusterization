# Description: Contains code needed to create clusters table on DB.
import sys, os

sys.path.append(os.getcwd())
from typing import Iterable, Dict, List
from collections import Counter
import math

from sqlalchemy import (
    Column,
    String,
    Integer,
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


class CommunitiesTable:
    def __init__(
        self, database_name: str = "quseed", table_name: str = "communities"
    ) -> None:
        """
        Creates clusters table on DB.

        Args:
            database_name (str, optional): Database where table will be created.
                Defaults to "quseed".
            table_name (str, optional): Name of the table that will be created.
                Defaults to "communities".
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

        class Communities(self.Base):
            __tablename__ = self.table_name

            uuid = Column(
                UUID(as_uuid=True),
                primary_key=True,
                server_default=text("uuid_generate_v4()"),
            )
            community_id = Column(String(250))
            community_label = Column(String(150))
            timestamp = Column(Date)
            size = Column(Integer)
            tags = Column(ARRAY(String(100)))
            categories = Column(
                ARRAY(String(100))
            )  # all the categorie of the community
            most_common_categories = Column(
                ARRAY(String(100))
            )  # the 20 most common categories of the community

            __table_args__ = (
                UniqueConstraint(
                    "community_id",
                    "community_label",
                    name=f"unique_{self.table_name}_id_community_label",
                ),
                Index(f"ix_{self.table_name}_uuid", "uuid"),
                Index(f"ix_{self.table_name}_community_id", "community_id"),
                Index(f"ix_{self.table_name}_community_label", "community_label"),
                Index(f"ix_{self.table_name}_timestamp", "timestamp"),
                Index(
                    f"ix_{self.table_name}_most_common_categories",
                    "most_common_categories",
                ),
            )

            def __repr__(self) -> str:
                return f"<Communities(uuid={self.uuid}, community_id={self.community_id}, community_label={self.community_label}, timestamp={self.timestamp}, size={self.size}, tags={self.tags}, categories={self.categories}, most_common_categories={self.most_common_categories})>"

        self.Communities = Communities

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
        Upsert records into the Communities table.

        Args:
            records (list[dict]): A list of dictionaries representing records to upsert.
        """
        self.init_db_session()
        try:
            for record in records:
                record = self.sanitize_input(record)
                existing_record = (
                    self.session.query(self.Communities)
                    .filter_by(community_id=record["community_id"])
                    .first()
                )
                if existing_record:
                    for key, value in record.items():
                        setattr(existing_record, key, value)
                else:
                    new_record = self.Communities(**record)
                    self.session.add(new_record)

            self.session.commit()
        except IntegrityError as e:
            self.session.rollback()
            print("IntegrityError:", e)
        except Exception as e:
            self.session.rollback()
            print("Error:", e)
            print("Record:", record)
            print(e)
        finally:
            self.close_db_session()

    def delete_communities_by_uuids(self, uuid_list: List[str]) -> None:
        """
        Delete communities by UUIDs.

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
            self.session.query(self.Communities).filter(
                self.Communities.uuid.in_(uuid_list)
            ).delete(synchronize_session=False)
            self.session.commit()
            print(f"{len(uuid_list)} community(ies) deleted successfully.")
        except Exception as e:
            self.session.rollback()
            print("Error:", e)
        finally:
            self.close_db_session()

    def delete_comunities_by_community_id(self, community_id_list: List[str]) -> None:
        """
        Delete communities by community_id.

        Args:
            community_id_list (List[str]): List of community_ids to delete.

        Returns:
            None
        """
        if not community_id_list:
            print("No community_ids provided for deletion.")
            return

        self.init_db_session()
        try:
            self.session.query(self.Communities).filter(
                self.Communities.community_id.in_(community_id_list)
            ).delete(synchronize_session=False)
            self.session.commit()
            print(f"{len(community_id_list)} community(ies) deleted successfully.")
        except Exception as e:
            self.session.rollback()
            print("Error:", e)
        finally:
            self.close_db_session()
