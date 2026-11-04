# Pydantic V2 Migration Guide

## Field Validation

### Modern Field Validator (@field_validator)

In Pydantic v1, field-level validation relied on the `@validator` decorator. In Pydantic v2, `@validator` is deprecated in favor of `@field_validator`.

Key modernization details:
- **`@classmethod` requirement**: Every `@field_validator` must be decorated with `@classmethod`.
- **Validation timing (`mode`)**:
  - `mode='after'` (default): Runs after Pydantic performs standard type parsing and coercion. The validator receives the parsed data type.
  - `mode='before'`: Runs before Pydantic parses or validates the input. The validator receives the raw input object.
  - `mode='wrap'`: Wraps the validation logic around Pydantic's internal validator using a handler function.
  - `mode='plain'`: Bypasses Pydantic's internal validation entirely for that field.
- **Context via `ValidationInfo`**: Instead of v1's `values`, `config`, and `field` parameters, v2 provides a unified `info: ValidationInfo` object. Previously validated fields can be accessed via `info.data` (a dictionary). Note that in v2, fields are validated strictly in the order they are defined on the model.

```python
# ==========================================
# Legacy Pydantic v1 Pattern (Deprecated)
# ==========================================
from pydantic import BaseModel, validator

class UserModel(BaseModel):
    name: str
    age: int

    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip().title()

    @validator('age')
    def validate_age(cls, v, values):
        if v < 0:
            raise ValueError("Age must be non-negative")
        return v
```

```python
# ==========================================
# Modern Pydantic v2 Pattern (Standard)
# ==========================================
from pydantic import BaseModel, field_validator, ValidationInfo

class UserModel(BaseModel):
    name: str
    age: int

    @field_validator('name', mode='after')
    @classmethod
    def validate_name(cls, v: str, info: ValidationInfo) -> str:
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip().title()

    @field_validator('age', mode='after')
    @classmethod
    def validate_age(cls, v: int, info: ValidationInfo) -> int:
        if v < 0:
            raise ValueError("Age must be non-negative")
        return v
```

---

## Model Configuration

### ConfigDict Specification

In Pydantic v1, model configuration was specified using an inner class named `Config`. In Pydantic v2, inner `class Config:` is deprecated and replaced by the `model_config = ConfigDict(...)` attribute.

Key configuration mappings:
- `ConfigDict` is imported directly from `pydantic`.
- String cleaning: `str_strip_whitespace=True` automatically removes leading and trailing whitespace from string fields.
- Assignment checks: `validate_assignment=True` enforces validation rules when model attributes are reassigned after instantiation.
- ORM conversion: `orm_mode = True` is replaced by `from_attributes=True`.
- Aliases: `allow_population_by_field_name = True` is replaced by `populate_by_name=True`.
- Extra attributes: `extra = 'forbid'` / `'ignore'` / `'allow'` continues as strings or `Extra` enum.

```python
# ==========================================
# Legacy Pydantic v1 Pattern (Deprecated)
# ==========================================
from pydantic import BaseModel

class UserProfile(BaseModel):
    username: str
    email: str

    class Config:
        str_strip_whitespace = True
        validate_assignment = True
        orm_mode = True
        allow_population_by_field_name = True
```

```python
# ==========================================
# Modern Pydantic v2 Pattern (Standard)
# ==========================================
from pydantic import BaseModel, ConfigDict

class UserProfile(BaseModel):
    username: str
    email: str

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        from_attributes=True,
        populate_by_name=True,
    )
```

---

## Root and Model Validators

### Modern Model Validator (@model_validator)

In Pydantic v1, cross-field validation and whole-model data transformations were performed using `@root_validator`. In Pydantic v2, `@root_validator` is deprecated and replaced by `@model_validator`.

Modes of `@model_validator`:
- **`mode='after'`**: Runs after individual field validation completes.
  - Defined as a standard instance method: `def check_rules(self) -> Self:`.
  - Operates on the typed, validated fields via `self.field_name`.
  - Must return `self`.
- **`mode='before'`**: Runs before any field validation or type coercion takes place.
  - Defined as a `@classmethod`: `@classmethod def pre_process(cls, data: Any) -> Any:`.
  - Receives the raw dictionary or input structure.
  - Returns the modified dictionary or data structure.

```python
# ==========================================
# Legacy Pydantic v1 Pattern (Deprecated)
# ==========================================
from pydantic import BaseModel, root_validator

class PasswordChangeRequest(BaseModel):
    new_password: str
    confirm_password: str

    @root_validator
    def verify_passwords_match(cls, values):
        new_pw = values.get('new_password')
        confirm_pw = values.get('confirm_password')
        if new_pw and confirm_pw and new_pw != confirm_pw:
            raise ValueError("Passwords do not match")
        return values
```

```python
# ==========================================
# Modern Pydantic v2 Pattern (Standard)
# ==========================================
import typing
from pydantic import BaseModel, model_validator

class PasswordChangeRequest(BaseModel):
    new_password: str
    confirm_password: str

    @model_validator(mode='after')
    def verify_passwords_match(self) -> typing.Self:
        if self.new_password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self
```

```python
# ==========================================
# Modern Pydantic v2 Before-Validator Pattern
# ==========================================
import typing
from pydantic import BaseModel, model_validator

class UserRegistration(BaseModel):
    display_name: str
    email: str

    @model_validator(mode='before')
    @classmethod
    def normalize_input(cls, data: typing.Any) -> typing.Any:
        if isinstance(data, dict):
            # Normalize legacy field names before field parsing
            if "name" in data and "display_name" not in data:
                data["display_name"] = data["name"]
        return data
```

---

## Serialization

### Model Dump Methods

In Pydantic v1, models were serialized into dictionaries using `.dict()` and into JSON strings using `.json()`. In Pydantic v2, both methods are deprecated in favor of `.model_dump()` and `.model_dump_json()`.

Key serialization features:
- `.model_dump()`: Returns a Python dictionary. Supports `mode='python'` (preserves Python objects like `datetime`, `UUID`) and `mode='json'` (converts non-JSON types to strings/numbers).
- `.model_dump_json()`: Serializes the model directly into an RFC 8259 compliant JSON string, implemented in high-performance Rust via `pydantic-core`.
- Supported arguments include:
  - `include`: Set or dict of fields to include.
  - `exclude`: Set or dict of fields to exclude.
  - `by_alias`: Whether field aliases should be used as output keys (default `False`).
  - `exclude_unset`: Only include fields explicitly set during initialization.
  - `exclude_defaults`: Exclude fields that equal their default values.
  - `exclude_none`: Exclude fields with `None` values.
  - `indent`: Format JSON output with indentation spaces.

```python
# ==========================================
# Legacy Pydantic v1 Pattern (Deprecated)
# ==========================================
from pydantic import BaseModel

class Product(BaseModel):
    id: int
    name: str
    price: float

product = Product(id=101, name="Mechanical Keyboard", price=149.99)

# Legacy v1 calls
legacy_dict = product.dict(exclude_none=True)
legacy_json = product.json(indent=2)
```

```python
# ==========================================
# Modern Pydantic v2 Pattern (Standard)
# ==========================================
from pydantic import BaseModel

class Product(BaseModel):
    id: int
    name: str
    price: float

product = Product(id=101, name="Mechanical Keyboard", price=149.99)

# Modern v2 calls
modern_dict = product.model_dump(exclude_none=True)
json_safe_dict = product.model_dump(mode="json")
modern_json_str = product.model_dump_json(indent=2)
```
