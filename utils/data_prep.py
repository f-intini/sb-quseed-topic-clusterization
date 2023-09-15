# Description: Data needed for the execution of the clusterization process.
import pickle
import sys, os

sys.path.append(os.getcwd())

from sqlalchemy import (
    Column,
    Integer,
    String,
    Double,
    UniqueConstraint,
    Index,
    inspect,
    text,
)
from sqlalchemy.orm import sessionmaker, declarative_base

from sb_py_fw.database.postgres import create_pg_engine

from utils.const import ALLOWED_DBS as allowed_dbs


class CategoriesDistanceIngestion:
    # TODO: add logging => !FUTURE!
    def __init__(
        self, database_name: str = "quseed", table_name: str = "categories_distances"
    ) -> None:
        """
        Ingests the categories distances data into a database.

        Args:
            database_name (str, optional): Database to populate. Defaults to "quseed".
            table_name (str, optional): Name of the table that will be created. Defaults to "categories_distances".

        Raises:
            ValueError: if <database_name> is not one of allowed_dbs.keys()
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
        self.__read_local_data__()

    def table_exists(self) -> bool:
        """
        Checks if the table <self.table_name> exists in the database.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        return inspect(self.engine).has_table(self.table_name)

    def __read_local_data__(self) -> None:
        """
        Reads the categories distances data from the local file.

        Raises:
            e: if the file cannot be read
        """
        try:
            with open("./categories_distances.pkl", "rb") as f:
                self.categories_distances = pickle.load(f)
        except Exception as e:
            print(e)
            raise e

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

    def __set_table_definitions__(self) -> None:
        """
        Sets the table definitions using sqlalchemy ORM.
        """

        class CategoriesDistance(self.Base):
            __tablename__ = self.table_name

            id = Column(Integer, primary_key=True)
            first_category = Column(String(100))
            second_category = Column(String(100))
            distance = Column(Double)
            __table_args__ = (
                UniqueConstraint(
                    "first_category",
                    "second_category",
                    name="categories_distances_unique_category_pair",
                ),
                Index("categories_distances_id_idx", "id"),
                Index(
                    "categories_distances_categories_idx",
                    "first_category",
                    "second_category",
                ),
            )

            def __repr__(self):
                return f"<CategoriesDistance(id={self.id}, first_category={self.first_category}, second_category={self.second_category}, distance={self.distance})>"

        self.CategoriesDistance = CategoriesDistance

    def create_table(self, reset: bool = False) -> None:
        """
        Creates the table <self.table_name> in the database.

        Args:
            reset (bool, optional): Flag variable, set to true to delete the existing table. Defaults to False.

        Raises:
            exception (e): if the table cannot be created
            exception (e): if the table cannot be dropped
        """
        if reset:
            if self.table_exists():
                try:
                    self.init_db_session()
                    self.session.execute(text(f"drop table {self.table_name} cascade;"))
                    self.session.commit()
                    self.close_db_session()
                    print(f"Table {self.table_name} dropped.")
                except Exception as e:
                    print(e)
                    raise e
            else:
                print(f"Table {self.table_name} does not exist.")

        if not self.table_exists():
            try:
                self.Base.metadata.create_all(self.engine)  # type: ignore
                print(f"Table {self.table_name} created.")
            except Exception as e:
                print(e)
                raise e
        else:
            print(f"Table {self.table_name} already exists.")

    def load_data(self) -> None:
        """
        Loads the data into the database.

        Raises:
            ValueError: if the table does not exist
            e: if the data cannot be loaded
        """
        if not self.table_exists():
            print(f"Table {self.table_name} does not exist.")
            raise ValueError(f"Table {self.table_name} does not exist.")

        data_to_load = list(self.categories_distances.items())

        self.init_db_session()
        for i in range(0, len(data_to_load), 1000):
            print(f"Pushing data - from: {i} to: {i+1000} of {len(data_to_load)}.")
            try:
                data = [
                    {
                        "first_category": d[0][0],
                        "second_category": d[0][1],
                        "distance": d[1],
                    }
                    for d in data_to_load[i : i + 1000]
                ]
                self.session.bulk_insert_mappings(self.CategoriesDistance, data)
                self.session.commit()
            except Exception as e:
                print(e)
                raise e
        self.close_db_session()

    def run(self, reset: bool = False) -> None:
        """
        Runs the ingestion process.

        Args:
            reset (bool, optional): Flag variable, set to true to delete the existing table. Defaults to False.
        """
        self.create_table(reset=reset)
        self.load_data()
