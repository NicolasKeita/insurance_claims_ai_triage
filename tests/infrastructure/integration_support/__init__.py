"Shared, test-only defaults and isolated PostgreSQL setup (no import-time I/O)."

from tests.infrastructure.integration_support._config import (
    get_test_database_config as get_test_database_config,
)
from tests.infrastructure.integration_support._config import get_test_qdrant_url as get_test_qdrant_url
from tests.infrastructure.integration_support._constants import (
    DEFAULT_TEST_DATABASE_URL as DEFAULT_TEST_DATABASE_URL,
)
from tests.infrastructure.integration_support._constants import (
    DEFAULT_TEST_QDRANT_URL as DEFAULT_TEST_QDRANT_URL,
)
from tests.infrastructure.integration_support._constants import PROJECT_ROOT as PROJECT_ROOT
from tests.infrastructure.integration_support._migrations import (
    migrate_test_schema as migrate_test_schema,
)
from tests.infrastructure.integration_support._postgres import (
    _create_test_schema as _create_test_schema,
)
from tests.infrastructure.integration_support._postgres import (
    isolated_postgres_schema as isolated_postgres_schema,
)
