import json
import logging
import os

import pandas as pd
from database_tools.adapters.postgresql import PostgresqlAdapter
from database_tools.lightning_uploader import LightningUploader
from testcontainers.postgres import PostgresContainer

from src.database.database import Base
from src.database.database import NenContractRaw

logger = logging.getLogger(__name__)

# Add a tuple with the CSV file for testing
tables_sources = [
    (NenContractRaw, '')
]

table_source_not_orm = [
    ('', '')
]


def load_json(file_path):
    with open(file_path, 'r') as file:
        data_json = json.load(file)
    return data_json['test_data']


def load_csv(file_path):
    df = pd.read_csv(file_path)
    # Convert the DataFrame to a list of dictionaries (records) for bulk insert
    return df.to_dict(orient='records')


class TestDatabaseManager:
    def __init__(self):
        """Initialize the test database manager."""
        self.postgres_container = None
        self.database = None

    def create_test_database(self):
        """Create a PostgreSQL test database using Docker."""
        logger.info("Starting test database setup...")

        # Start PostgreSQL container
        logger.info("Starting PostgreSQL container...")
        self.postgres_container = PostgresContainer("postgres:15")
        self.postgres_container.start()
        logger.info(f"Container started on port {self.postgres_container.get_exposed_port(5432)}")

        # Set environment variables from container - USE CORRECT VARIABLE NAMES
        container_host = self.postgres_container.get_container_host_ip()
        container_port = str(self.postgres_container.get_exposed_port(5432))
        container_db = self.postgres_container.dbname
        container_user = self.postgres_container.username
        container_password = self.postgres_container.password

        os.environ.update({
            'DB_HOST': container_host,
            'DB_PORT': container_port,
            'DB_NAME': container_db,
            'DB_USER': container_user,
            'DB_PASS': container_password,  # ✅ Fixed: Use DB_PASS not DB_PASSWORD
            'DB_SCHEMA': 'test_schema'
        })

        # Debug: Print environment variables
        logger.info(f"Database connection details:")
        logger.info(f"  Host: {container_host}")
        logger.info(f"  Port: {container_port}")
        logger.info(f"  DB Name: {container_db}")
        logger.info(f"  User: {container_user}")
        logger.info(f"  Password: {container_password}")

        # Create database adapter directly with explicit parameters
        # Don't use from_env_vars() due to global variable caching issue
        self.database = PostgresqlAdapter(
            host=container_host,
            port=int(container_port),
            database_name=container_db,
            user=container_user,
            password=container_password,
            schema_name='test_schema'
        )

        # Initialize schema
        self.database.init_schema(Base.metadata)

        # Load test data (only if files exist)
        self._load_test_data()

        logger.info("Test database setup complete")
        return self.database

    def _load_test_data(self):
        """Load test data into the database."""
        # Skip loading test data if files are empty/don't exist
        for model, file_name in tables_sources:
            if not file_name:  # Skip empty file names
                continue

            data_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f'data/{file_name}'
            )

            if not os.path.exists(data_path):
                logger.info(f"Test data file not found: {data_path}, skipping")
                continue

            # Determine the file extension to choose the correct loading function
            if file_name.endswith('.json'):
                data = load_json(data_path)
            elif file_name.endswith('.csv'):
                data = load_csv(data_path)
            else:
                raise ValueError(f"Unsupported file type for {file_name}")

            self.database.bulk_insert(model, data)

        # Skip non-ORM tables if they're empty
        up = LightningUploader(schema='recommender', table='', database=self.database)
        for table, file_name in table_source_not_orm:
            if not table or not file_name:  # Skip empty table/file names
                continue

            data_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f'data/{file_name}'
            )

            if not os.path.exists(data_path):
                logger.info(f"Test data file not found: {data_path}, skipping")
                continue

            data = load_csv(data_path)
            up.table = table
            up.upload_data(data)

    def cleanup(self):
        """Stop container and clean up."""
        errors = []

        if self.database:
            try:
                self.database.close()
            except Exception as e:
                errors.append(f"Database cleanup error: {e}")

        if self.postgres_container:
            try:
                self.postgres_container.stop()
            except Exception as e:
                errors.append(f"Container cleanup error: {e}")

        # Clean up environment variables
        env_vars = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASS', 'DB_SCHEMA']
        for key in env_vars:
            os.environ.pop(key, None)

        if errors:
            logger.warning(f"Cleanup errors occurred: {'; '.join(errors)}")
