"""Golden evaluation fixtures representing the 6-change Minimal Spanning Basis for Pydantic v2."""

from __future__ import annotations

import typing
from pydantic import BaseModel, ConfigDict, Field


class GoldenEvalFixture(BaseModel):
    """Specification of an orthogonal evaluation benchmark fixture."""

    model_config = ConfigDict(frozen=True)

    eval_id: str
    topic: str
    description: str
    legacy_code: str
    forbidden_tokens: list[str] = Field(default_factory=list)
    required_tokens: list[str] = Field(default_factory=list)
    test_payload: typing.Optional[dict[str, typing.Any]] = None


GOLDEN_EVALS: list[GoldenEvalFixture] = [
    # -------------------------------------------------------------------------
    # Eval 1: Inner class Config -> ConfigDict
    # -------------------------------------------------------------------------
    GoldenEvalFixture(
        eval_id="eval_01_config_dict",
        topic="pydantic_config",
        description="Migrate legacy inner class Config with orm_mode and allow_mutation to model_config = ConfigDict",
        legacy_code="""from pydantic import BaseModel

class UserModel(BaseModel):
    id: int
    name: str

    class Config:
        orm_mode = True
        allow_mutation = False
""",
        forbidden_tokens=["class Config:", "orm_mode", "allow_mutation"],
        required_tokens=["model_config", "ConfigDict", "from_attributes", "frozen"],
        test_payload={"id": 101, "name": "Alice"},
    ),
    # -------------------------------------------------------------------------
    # Eval 2: @validator -> @field_validator with @classmethod
    # -------------------------------------------------------------------------
    GoldenEvalFixture(
        eval_id="eval_02_field_validator",
        topic="pydantic_validators",
        description="Migrate legacy @validator with pre=True to @field_validator with mode='before' and mandatory @classmethod",
        legacy_code="""from pydantic import BaseModel, validator

class UserModel(BaseModel):
    name: str

    @validator('name', pre=True)
    def check_name(cls, v):
        return v.strip().title()
""",
        forbidden_tokens=["@validator", "pre=True"],
        required_tokens=["@field_validator", "@classmethod"],
        test_payload={"name": "  alice  "},
    ),
    # -------------------------------------------------------------------------
    # Eval 3: @root_validator -> @model_validator
    # -------------------------------------------------------------------------
    GoldenEvalFixture(
        eval_id="eval_03_model_validator",
        topic="pydantic_validators",
        description="Migrate legacy @root_validator to @model_validator(mode='after')",
        legacy_code="""from pydantic import BaseModel, root_validator

class SignUpModel(BaseModel):
    password: str
    confirm_password: str

    @root_validator
    def verify_passwords_match(cls, values):
        pw = values.get('password')
        cpw = values.get('confirm_password')
        if pw and cpw and pw != cpw:
            raise ValueError('passwords do not match')
        return values
""",
        forbidden_tokens=["@root_validator"],
        required_tokens=["@model_validator"],
        test_payload={"password": "secret", "confirm_password": "secret"},
    ),
    # -------------------------------------------------------------------------
    # Eval 4: .dict() and .json() serialization -> .model_dump() and .model_dump_json()
    # -------------------------------------------------------------------------
    GoldenEvalFixture(
        eval_id="eval_04_serialization",
        topic="pydantic_serialization",
        description="Migrate legacy export methods .dict() and .json() to .model_dump() and .model_dump_json()",
        legacy_code="""from pydantic import BaseModel

class Item(BaseModel):
    title: str
    price: float

item = Item(title="Book", price=19.99)
d = item.dict(exclude={"price"})
j = item.json()
""",
        forbidden_tokens=[".dict(", ".json("],
        required_tokens=[".model_dump(", ".model_dump_json("],
        test_payload=None,
    ),
    # -------------------------------------------------------------------------
    # Eval 5: BaseSettings package boundary -> pydantic_settings
    # -------------------------------------------------------------------------
    GoldenEvalFixture(
        eval_id="eval_05_settings",
        topic="pydantic_settings",
        description="Migrate BaseSettings import from pydantic to pydantic_settings and replace inner Config",
        legacy_code="""from pydantic import BaseSettings

class AppSettings(BaseSettings):
    app_name: str = "MyService"
    debug: bool = False

    class Config:
        env_prefix = "MYAPP_"
""",
        forbidden_tokens=["from pydantic import BaseSettings", "class Config:"],
        required_tokens=["from pydantic_settings import BaseSettings", "model_config"],
        test_payload=None,
    ),
    # -------------------------------------------------------------------------
    # Eval 6: FastAPI @app.on_event -> @asynccontextmanager lifespan
    # -------------------------------------------------------------------------
    GoldenEvalFixture(
        eval_id="eval_06_fastapi_lifespan",
        topic="fastapi_lifespan",
        description="Migrate deprecated FastAPI @app.on_event startup/shutdown to async lifespan context manager",
        legacy_code="""from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    print("Application starting up")

@app.on_event("shutdown")
async def shutdown_event():
    print("Application shutting down")
""",
        forbidden_tokens=["@app.on_event"],
        required_tokens=["lifespan", "asynccontextmanager"],
        test_payload=None,
    ),
]
