from database_tools.adapters.postgresql import PostgresqlAdapter

tender_database = PostgresqlAdapter.from_env_vars()
