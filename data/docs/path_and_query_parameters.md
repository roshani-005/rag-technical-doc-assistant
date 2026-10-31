# Path Parameters and Query Parameters in FastAPI

FastAPI enables declaring path and query parameters using Python type annotations. FastAPI automatically validates data types, serializes responses, and documents constraints in OpenAPI.

## Path Parameters

You can declare path "parameters" or "variables" with the same syntax used by Python format strings:

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    # item_id is automatically parsed and validated as an integer
    return {"item_id": item_id}
```

If a client sends `/items/foo`, FastAPI returns an HTTP 422 Unprocessable Entity error with a clear JSON describing that `item_id` should be an integer.

### Path Order Matters

When creating path operations, you may have fixed paths like `/users/me` and dynamic paths like `/users/{user_id}`. Always declare fixed paths *before* parameterized paths:

```python
@app.get("/users/me")
async def read_user_me():
    return {"user_id": "current user"}

@app.get("/users/{user_id}")
async def read_user(user_id: str):
    return {"user_id": user_id}
```

## Query Parameters

Function parameters that are not part of the path parameters are automatically interpreted as "query parameters":

```python
from typing import Optional

@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10, search: Optional[str] = None):
    return {"skip": skip, "limit": limit, "search": search}
```

When accessing `GET /items/?skip=20&limit=50&search=shoes`:
- `skip` becomes `20`
- `limit` becomes `50`
- `search` becomes `"shoes"`

## Query Parameter Validation with Annotated and Query

To add constraints (min length, max length, regex, title, description), import `Query` and use `typing.Annotated`:

```python
from typing import Annotated
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/items/filter")
async def filter_items(
    q: Annotated[
        str | None, 
        Query(
            min_length=3, 
            max_length=50, 
            title="Search Query",
            description="Query string for the items to search in the database",
            pattern="^[a-zA-Z0-9_-]+$"
        )
    ] = None,
    tags: Annotated[list[str], Query()] = ["popular", "latest"]
):
    results = {"q": q, "tags": tags}
    return results
```

### Required Query Parameters

If you omit the default value or use `...` (Ellipsis), the query parameter is required:

```python
@app.get("/search")
async def search_term(query: Annotated[str, Query(min_length=1)]):
    return {"query": query}
```
