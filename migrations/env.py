"""DATABASE_URL is shared with the application, never stored in alembic.ini."""
from alembic import context
from sqlalchemy import create_engine, pool

from persistence.database import DatabaseConfig
from persistence.models import Base

config = context.config
target_metadata = Base.metadata
# Tests can supply a prevalidated connection; CLI always uses DATABASE_URL.
connection = config.attributes.get("connection")

if context.is_offline_mode():
    context.configure(url=DatabaseConfig.from_env().url,
                      target_metadata=target_metadata, literal_binds=True,
                      dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()
elif connection is not None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()
else:
    engine = create_engine(DatabaseConfig.from_env().url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()
