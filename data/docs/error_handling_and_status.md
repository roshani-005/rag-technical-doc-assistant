# Error Handling and HTTP Status Codes in FastAPI

Handling exceptions and returning clear, actionable HTTP status codes is essential for building resilient APIs with FastAPI.

## Using HTTPException

To return HTTP error responses to the client, raise `fastapi.HTTPException`:

```python
from fastapi import FastAPI, HTTPException, status

app = FastAPI()

items = {"foo": "The Foo Item"}

@app.get("/items/{item_id}")
async def read_item(item_id: str):
    if item_id not in items:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item not found in database",
            headers={"X-Error": "ResourceMissing"}
        )
    return {"item": items[item_id]}
```

When `HTTPException` is raised, FastAPI catches it and converts it into an HTTP response with JSON payload:
```json
{
  "detail": "Item not found in database"
}
```

## Custom Exception Handlers

You can create custom exception classes and register custom handlers on the FastAPI application instance:

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

class UnicornValidationError(Exception):
    def __init__(self, name: str):
        self.name = name

app = FastAPI()

@app.exception_handler(UnicornValidationError)
async def unicorn_exception_handler(request: Request, exc: UnicornValidationError):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} broke the unicorn rules."}
    )

@app.get("/unicorns/{name}")
async def get_unicorn(name: str):
    if name == "yolo":
        raise UnicornValidationError(name=name)
    return {"unicorn": name}
```

## Overriding RequestValidationError

When client input fails validation, FastAPI raises `fastapi.exceptions.RequestValidationError`. You can override how validation errors are formatted:

```python
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"errors": exc.errors(), "body": exc.body, "custom_hint": "Please verify request types"}
    )
```

## HTTP Status Codes Reference

FastAPI provides convenient status constants via `fastapi.status`:
- `status.HTTP_200_OK`: Default for GET/PUT/PATCH
- `status.HTTP_201_CREATED`: Standard for successful POST creating a resource
- `status.HTTP_204_NO_CONTENT`: For DELETE or actions returning no payload
- `status.HTTP_400_BAD_REQUEST`: Client syntax error
- `status.HTTP_401_UNAUTHORIZED`: Missing or invalid credentials
- `status.HTTP_403_FORBIDDEN`: Valid credentials but insufficient permissions
- `status.HTTP_404_NOT_FOUND`: Resource does not exist
- `status.HTTP_422_UNPROCESSABLE_ENTITY`: Validation failure on request body/params
