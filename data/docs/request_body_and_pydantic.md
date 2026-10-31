# Request Body and Pydantic Validation

When you need to send data from a client (e.g., a frontend browser) to your API, you send it as a request body. To declare request bodies, use Pydantic models.

## Declaring Request Bodies with Pydantic

Import `BaseModel` from `pydantic` and define your data model:

```python
from typing import Optional
from fastapi import FastAPI, status
from pydantic import BaseModel, Field

app = FastAPI()

class Item(BaseModel):
    name: str
    description: Optional[str] = Field(default=None, max_length=300)
    price: float = Field(gt=0, description="The price must be greater than zero")
    tax: Optional[float] = None
    tags: list[str] = []

@app.post("/items/", status_code=status.HTTP_201_CREATED)
async def create_item(item: Item):
    item_dict = item.model_dump()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict
```

## Pydantic Field Validation and Metadata

Use `pydantic.Field` to declare validation rules and metadata for model attributes:

- `gt`: Greater than
- `ge`: Greater than or equal to
- `lt`: Less than
- `le`: Less than or equal to
- `min_length`: Minimum string length
- `max_length`: Maximum string length
- `pattern`: Regular expression pattern matching

```python
class UserProfile(BaseModel):
    username: str = Field(min_length=3, max_length=30, pattern=r"^[a-zA-Z0-9_]+$")
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    age: int = Field(ge=18, le=120)
```

## Nested Models and Hierarchical Data

Pydantic models can be nested inside other models to handle arbitrary hierarchical JSON structures:

```python
class Image(BaseModel):
    url: str
    name: str

class Offer(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    items: list[Item]
    image: Optional[Image] = None

@app.post("/offers/")
async def create_offer(offer: Offer):
    return offer
```

FastAPI automatically parses the nested JSON, validates each leaf node, generates JSON Schema definitions in OpenAPI, and converts request payloads directly into Python objects.
