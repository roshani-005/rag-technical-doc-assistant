# Dependency Injection in FastAPI

FastAPI includes a powerful, intuitive Dependency Injection system that enables code reuse, shared database connections, security enforcement, and role-based access control.

## What is a Dependency?

In programming, a dependency is something your code requires to function before executing logic (e.g., a database session, an authenticated user token, or common query parameters).

FastAPI uses `fastapi.Depends` to declare dependencies in path operations:

```python
from typing import Annotated
from fastapi import Depends, FastAPI

app = FastAPI()

async def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: Annotated[dict, Depends(common_parameters)]):
    return commons

@app.get("/users/")
async def read_users(commons: Annotated[dict, Depends(common_parameters)]):
    return commons
```

## Class-based Dependencies

You can also use a Python class as a dependency. The class constructor will be invoked automatically:

```python
class CommonQueryParams:
    def __init__(self, q: str | None = None, skip: int = 0, limit: int = 50):
        self.q = q
        self.skip = skip
        self.limit = limit

@app.get("/products/")
async def read_products(params: Annotated[CommonQueryParams, Depends()]):
    # When Depends() is called with empty arguments inside Annotated[Class, Depends()],
    # FastAPI automatically infers CommonQueryParams as the dependency!
    return {"q": params.q, "skip": params.skip, "limit": params.limit}
```

## Dependencies with `yield` (Context Managers / Teardown)

FastAPI supports dependencies that run code before and after a request (e.g., acquiring and closing a database session or managing resources):

```python
async def get_db():
    db = DBSession()
    try:
        yield db
    finally:
        db.close()

@app.get("/customers/")
async def list_customers(db: Annotated[DBSession, Depends(get_db)]):
    return db.query_customers()
```

The code before `yield` runs prior to executing the endpoint handler. The code after `yield` executes after the response has been sent to the client, even if an exception was raised.

## Security and Authentication Dependencies

Dependencies can enforce authentication headers like Bearer tokens or API keys:

```python
from fastapi import Header, HTTPException, status

async def verify_token(x_token: Annotated[str, Header()]):
    if x_token != "super-secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid or missing X-Token header"
        )
    return x_token

@app.get("/admin/", dependencies=[Depends(verify_token)])
async def admin_dashboard():
    return {"message": "Welcome to Admin"}
```
