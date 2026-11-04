# Pydantic Settings Migration Guide

## BaseSettings Migration

### Modern pydantic-settings Package

In Pydantic v1, `BaseSettings` was bundled inside the core `pydantic` package (`from pydantic import BaseSettings`). In Pydantic v2, settings management was extracted into a standalone package named `pydantic-settings` to keep the core library lean and focused on validation.

Key modernization details:
- **Dependency installation**: Add the separate dependency via `uv add pydantic-settings`.
- **Import source**: Import `BaseSettings` and `SettingsConfigDict` from `pydantic_settings` (never import `BaseSettings` from `pydantic`).
- **Configuration syntax**: Replace the deprecated inner `class Config:` with `model_config = SettingsConfigDict(...)`.
- **Common settings options**:
  - `env_file`: Path to the `.env` file (e.g., `'.env'` or `('.env', '.env.local')`).
  - `env_file_encoding`: File encoding, usually `'utf-8'`.
  - `env_prefix`: String prefix prepended to all environment variable lookups (e.g., `'APP_'`).
  - `case_sensitive`: Boolean indicating whether environment variable matching is case sensitive (default `False`).
  - `extra`: Behaviour for extra variables (`'ignore'` or `'forbid'`).
  - `env_nested_delimiter`: Delimiter for populating nested models from environment variables (e.g., `'__'`).

```python
# ==========================================
# Legacy Pydantic v1 Pattern (Deprecated)
# ==========================================
from pydantic import BaseSettings

class DatabaseSettings(BaseSettings):
    host: str = "localhost"
    port: int = 5432
    db_name: str
    username: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "DB_"
```

```python
# ==========================================
# Modern Pydantic v2 Pattern (Standard)
# ==========================================
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    """Modern application configuration using pydantic-settings."""

    app_name: str = Field(default="Migration Copilot", description="Application service name")
    debug: bool = Field(default=False, description="Enable debug logging and hot-reload")
    database_url: str = Field(..., description="PostgreSQL async connection string")
    api_port: int = Field(default=8000, description="FastAPI HTTP listener port")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
        extra="ignore",
    )

if __name__ == "__main__":
    # Settings instance loads variables from environment and .env automatically
    config = AppConfig(database_url="postgresql+asyncpg://admin:secret@127.0.0.1:5432/app")
    print(f"Service '{config.app_name}' running on port {config.api_port}")
```

---

## Environment Parsing & Precedence

### Configuration Precedence

When initializing a `BaseSettings` subclass, values can come from multiple sources. `pydantic-settings` resolves configuration values using a strict, deterministic priority order.

The resolution precedence (highest priority to lowest priority) is:

1. **Explicit Initializer Arguments (`kwargs`)**:
   - Values passed directly when creating the settings instance (e.g., `AppSettings(port=9000)`).
   - Initializer kwargs always override all external environment sources.

2. **System Environment Variables**:
   - Variables exported in the current operating system shell environment (`os.environ`).
   - If `env_prefix` is configured, variables must match the prefixed name (e.g., `APP_PORT=8080`).

3. **Dotenv File (`.env`)**:
   - Values defined in the files specified by `env_file` (e.g., `env_file=".env"`).
   - If multiple files are passed in a tuple (e.g., `env_file=('.env', '.env.prod')`), later files in the tuple take priority over earlier ones.

4. **Model Field Defaults**:
   - The default values defined directly on the model attributes or via `Field(default=...)`.
   - Used only when no other source supplies a value for the field.

```python
# ==========================================
# Configuration Precedence Demonstration
# ==========================================
import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class ServiceConfig(BaseSettings):
    host: str = Field(default="127.0.0.1")  # Level 4: Model default
    port: int = Field(default=8000)         # Level 4: Model default

    model_config = SettingsConfigDict(
        env_file=".env",                    # Level 3: Dotenv file
        env_file_encoding="utf-8",
        env_prefix="SERVICE_",
        extra="ignore",
    )

# Case 1: Falling back to field default
# If no env var or .env entry exists, port is 8000
cfg_default = ServiceConfig()
assert cfg_default.port == 8000

# Case 2: System environment variable overrides .env and defaults
os.environ["SERVICE_PORT"] = "8080"
cfg_env = ServiceConfig()
assert cfg_env.port == 8080

# Case 3: Explicit kwargs override system env, .env, and defaults
cfg_override = ServiceConfig(port=9090)
assert cfg_override.port == 9090
```
