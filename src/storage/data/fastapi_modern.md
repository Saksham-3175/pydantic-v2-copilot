# Modern FastAPI Migration Guide

## Application Lifespan

### Lifespan Context Manager

In legacy FastAPI applications, application startup and teardown routines were registered using the `@app.on_event("startup")` and `@app.on_event("shutdown")` decorators. In modern FastAPI (FastAPI >= 0.93.0 following Starlette's ASGI lifespan protocol), `@app.on_event` is deprecated in favor of the `lifespan` context manager.

Key architectural benefits:
- **Unified lifecycle**: Startup logic is executed before the `yield`, and shutdown logic is executed immediately after `yield` within a single context manager.
- **Robust error handling**: Standard Python `try...finally` blocks ensure reliable cleanup even if startup tasks fail or an unhandled exception occurs during runtime.
- **Shared application state**: Resources such as database connection pools, HTTP client sessions, and AI model weights can be initialized on `app.state` or yielded as a dictionary to be shared across the entire application lifecycle.
- **Async native**: Built directly on `contextlib.asynccontextmanager` for seamless non-blocking asynchronous resource handling.

```python
# ==========================================
# Legacy FastAPI Event Pattern (Deprecated)
# ==========================================
from fastapi import FastAPI

app = FastAPI()

# Deprecated event handlers
@app.on_event("startup")
async def startup_database():
    # Setup database connection
    print("Database connecting...")

@app.on_event("shutdown")
async def shutdown_database():
    # Teardown database connection
    print("Database disconnecting...")
```

```python
# ==========================================
# Modern FastAPI Lifespan Pattern (Standard)
# ==========================================
import typing
from contextlib import asynccontextmanager
from fastapi import FastAPI

class MockDatabaseClient:
    """Mock asynchronous database pool for lifecycle illustration."""
    async def connect(self) -> None:
        print("[Lifespan] Connecting to database cluster...")

    async def disconnect(self) -> None:
        print("[Lifespan] Closing database connection pool...")

    async def ping(self) -> bool:
        return True

@asynccontextmanager
async def lifespan(app: FastAPI) -> typing.AsyncIterator[None]:
    # ----------------------------------------------------
    # STARTUP: Executed before accepting incoming requests
    # ----------------------------------------------------
    db_client = MockDatabaseClient()
    await db_client.connect()
    app.state.db_client = db_client

    yield  # Application is live and serving requests

    # ----------------------------------------------------
    # SHUTDOWN: Executed during graceful server termination
    # ----------------------------------------------------
    await db_client.disconnect()

# Register the lifespan handler during FastAPI initialization
app = FastAPI(
    title="Modern Lifespan API",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
async def health_check() -> dict[str, str]:
    db: MockDatabaseClient = app.state.db_client
    is_alive = await db.ping()
    return {"status": "healthy" if is_alive else "unhealthy"}
```

---

## Dependency Injection

### Annotated Dependencies

In legacy FastAPI applications, route dependencies were commonly declared using default function parameter assignments (e.g., `db: Session = Depends(get_db)`). While functional, this pattern conflates the type annotation with runtime metadata and causes friction with static analysis tools and unit testing.

Modern FastAPI adopts Python's standard `typing.Annotated` for dependency injection (`typing.Annotated[DepType, Depends(get_dep)]`).

Key advantages:
- **Clean static type checking**: Type linters (MyPy, Pyright, Ruff) see `DepType` directly without being confused by `Depends(...)` being assigned as a default parameter value.
- **Reusable dependency aliases**: You can define clear type aliases (e.g., `DatabaseDep = typing.Annotated[AsyncSession, Depends(get_db)]`) once and reuse them across multiple route definitions.
- **Frictionless unit testing**: Because route handlers do not require `Depends` in their default values, you can invoke the route function in unit tests directly with mock arguments without overriding FastAPI dependencies.
- **Preserved default parameters**: Standard Python default parameters can still be used for optional query parameters alongside Annotated dependencies.

```python
# ==========================================
# Legacy Parameter Pattern (Not Recommended)
# ==========================================
from fastapi import FastAPI, Depends

app = FastAPI()

def get_auth_token():
    return "legacy_token_123"

@app.get("/legacy/user")
async def legacy_get_user(token: str = Depends(get_auth_token)):
    return {"token": token}
```

```python
# ==========================================
# Modern Annotated Pattern (Standard)
# ==========================================
import typing
from fastapi import FastAPI, Depends, Header, HTTPException, status

app = FastAPI()

# Dependency callable
async def get_current_user_token(
    authorization: typing.Annotated[typing.Optional[str], Header()] = None,
) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Valid Bearer token required in Authorization header",
        )
    return authorization.removeprefix("Bearer ").strip()

# Type alias using Annotated and Depends
CurrentUserToken = typing.Annotated[str, Depends(get_current_user_token)]

@app.get("/api/v1/profile")
async def get_user_profile(token: CurrentUserToken) -> dict[str, str]:
    """Protected endpoint consuming the modern Annotated dependency."""
    return {
        "status": "authenticated",
        "user_token": token,
    }

@app.get("/api/v1/projects")
async def list_projects(
    token: CurrentUserToken,
    limit: int = 10,  # Standard optional parameter with default
) -> dict[str, typing.Any]:
    return {
        "authenticated_token": token,
        "limit": limit,
        "projects": ["Alpha", "Beta", "Gamma"],
    }
```
