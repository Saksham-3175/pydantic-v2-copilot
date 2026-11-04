"""Unit tests for ASTCritic deterministic verification gates."""

import pytest
from core_agents.critic import ASTCritic


@pytest.fixture
def critic() -> ASTCritic:
    """Fixture providing an instance of ASTCritic."""
    return ASTCritic()


def test_critic_clean_v2_code_passes(critic: ASTCritic) -> None:
    """Gate 1-4 pass on clean modern Pydantic v2 code."""
    code = """
from pydantic import BaseModel, ConfigDict, field_validator

class UserModel(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str
    age: int

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return v.strip()

user = UserModel(name="  Alice  ", age=30)
dumped = user.model_dump()
dumped_json = user.model_dump_json()
"""
    verdict = critic.evaluate(code)
    assert verdict.passed is True
    assert len(verdict.failures) == 0
    assert len(verdict.suggested_fixes) == 0


def test_critic_legacy_validator_fails_gate_2(critic: ASTCritic) -> None:
    """Gate 2 flags legacy @validator decorator."""
    code = """
from pydantic import BaseModel, validator

class UserModel(BaseModel):
    name: str

    @validator("name")
    def validate_name(cls, v: str) -> str:
        return v
"""
    verdict = critic.evaluate(code)
    assert verdict.passed is False
    assert any(f.gate == 2 for f in verdict.failures)
    failure = next(f for f in verdict.failures if f.gate == 2)
    assert "validator" in failure.message
    assert any("field_validator" in fix for fix in verdict.suggested_fixes)


def test_critic_legacy_root_validator_fails_gate_2(critic: ASTCritic) -> None:
    """Gate 2 flags legacy @root_validator decorator."""
    code = """
from pydantic import BaseModel, root_validator

class UserModel(BaseModel):
    name: str

    @root_validator
    def validate_all(cls, values):
        return values
"""
    verdict = critic.evaluate(code)
    assert verdict.passed is False
    assert any(f.gate == 2 for f in verdict.failures)
    failure = next(f for f in verdict.failures if f.gate == 2)
    assert "validator" in failure.message
    assert any("model_validator" in fix for fix in verdict.suggested_fixes)


def test_critic_legacy_class_config_fails_gate_2(critic: ASTCritic) -> None:
    """Gate 2 flags legacy inner 'class Config:'."""
    code = """
from pydantic import BaseModel

class UserModel(BaseModel):
    name: str

    class Config:
        str_strip_whitespace = True
"""
    verdict = critic.evaluate(code)
    assert verdict.passed is False
    assert any(f.gate == 2 for f in verdict.failures)
    failure = next(f for f in verdict.failures if f.gate == 2)
    assert "Config" in failure.message
    assert any("model_config" in fix for fix in verdict.suggested_fixes)


def test_critic_legacy_dict_method_fails_gate_2(critic: ASTCritic) -> None:
    """Gate 2 flags legacy .dict() and .json() export methods."""
    code = """
from pydantic import BaseModel

class UserModel(BaseModel):
    name: str

user = UserModel(name="Alice")
d = user.dict()
j = user.json()
"""
    verdict = critic.evaluate(code)
    assert verdict.passed is False
    assert any(f.gate == 2 for f in verdict.failures)
    dict_failures = [f for f in verdict.failures if f.gate == 2 and "export method" in f.message]
    assert len(dict_failures) >= 1
    assert any("model_dump" in fix for fix in verdict.suggested_fixes)


def test_critic_missing_classmethod_fails_gate_3(critic: ASTCritic) -> None:
    """Gate 3 flags @field_validator missing the @classmethod decorator."""
    code = """
from pydantic import BaseModel, field_validator

class UserModel(BaseModel):
    name: str

    @field_validator("name")
    def validate_name(cls, v: str) -> str:
        return v
"""
    verdict = critic.evaluate(code)
    assert verdict.passed is False
    assert any(f.gate == 3 for f in verdict.failures)
    failure = next(f for f in verdict.failures if f.gate == 3)
    assert "@classmethod" in failure.message
    assert any("@classmethod" in fix for fix in verdict.suggested_fixes)


def test_critic_syntax_error_fails_gate_1(critic: ASTCritic) -> None:
    """Gate 1 halts immediately on broken Python syntax."""
    code = "def invalid_code(:"
    verdict = critic.evaluate(code)
    assert verdict.passed is False
    assert len(verdict.failures) == 1
    failure = verdict.failures[0]
    assert failure.gate == 1
    assert failure.gate_name == "AST Parseability"
    assert failure.offending_line == 1
    assert len(verdict.suggested_fixes) == 1
    assert "syntax error" in verdict.suggested_fixes[0].lower()


def test_critic_field_regex_fails_gate_2(critic: ASTCritic) -> None:
    """Gate 2 flags Field(regex=...) replaced by pattern=."""
    code = """
from pydantic import BaseModel, Field

class Item(BaseModel):
    name: str = Field(..., regex="^[a-z]+$")
"""
    verdict = critic.evaluate(code)
    assert verdict.passed is False
    assert any(f.gate == 2 for f in verdict.failures)
    failure = next(f for f in verdict.failures if f.gate == 2 and "regex" in f.message)
    assert "pattern" in failure.message


def test_critic_basesettings_import_fails_gate_2(critic: ASTCritic) -> None:
    """Gate 2 flags importing BaseSettings from pydantic."""
    code = """
from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "TestApp"
"""
    verdict = critic.evaluate(code)
    assert verdict.passed is False
    assert any(f.gate == 2 for f in verdict.failures)
    failure = next(f for f in verdict.failures if f.gate == 2 and "BaseSettings" in f.message)
    assert "pydantic_settings" in failure.message


def test_critic_on_event_fails_gate_2(critic: ASTCritic) -> None:
    """Gate 2 flags @app.on_event("startup") deprecated in favor of lifespan."""
    code = """
from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
async def startup():
    pass
"""
    verdict = critic.evaluate(code)
    assert verdict.passed is False
    assert any(f.gate == 2 for f in verdict.failures)
    failure = next(f for f in verdict.failures if f.gate == 2 and "on_event" in f.message)
    assert "lifespan" in failure.message


def test_critic_invalid_model_config_fails_gate_3(critic: ASTCritic) -> None:
    """Gate 3 flags invalid assignment to model_config."""
    code = """
from pydantic import BaseModel

class UserModel(BaseModel):
    model_config = "this_should_be_a_ConfigDict"
    name: str
"""
    verdict = critic.evaluate(code)
    assert verdict.passed is False
    assert any(f.gate == 3 for f in verdict.failures)
    failure = next(f for f in verdict.failures if f.gate == 3 and "model_config" in f.message)
    assert "ConfigDict" in failure.message


def test_critic_runtime_error_fails_gate_4(critic: ASTCritic) -> None:
    """Gate 4 captures runtime exceptions in candidate code."""
    code = """
from pydantic import BaseModel, field_validator

class BadModel(BaseModel):
    x: int

    @field_validator("x", mode="invalid_mode")
    @classmethod
    def check_x(cls, v):
        return v
"""
    verdict = critic.evaluate(code)
    assert verdict.passed is False
    assert any(f.gate == 4 for f in verdict.failures)
    failure = next(f for f in verdict.failures if f.gate == 4)
    assert failure.gate_name == "Runtime Execution"
    assert "Runtime evaluation error" in failure.message


def test_critic_gracefully_handles_missing_third_party_import(critic: ASTCritic) -> None:
    """Gate 4 does not fail on uninstalled external library imports."""
    code = """
import non_existent_3rdparty_module
from uninstalled_package.submodule import ExternalClient

class SimpleModel:
    pass
"""
    verdict = critic.evaluate(code)
    assert verdict.passed is True
    assert len(verdict.failures) == 0
