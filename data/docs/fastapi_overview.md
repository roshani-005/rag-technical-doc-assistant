# FastAPI Overview and Core Concepts

FastAPI is a modern, high-performance, web framework for building APIs with Python 3.8+ based on standard Python type hints.

## Key Features

1. **High Performance**: Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic). One of the fastest Python frameworks available.
2. **Fast to code**: Increase the speed to develop features by about 200% to 300%.
3. **Fewer bugs**: Reduce about 40% of human (developer) induced errors.
4. **Intuitive**: Great editor support. Completion everywhere. Less time debugging.
5. **Easy**: Designed to be easy to use and learn. Less time reading docs.
6. **Short**: Minimize code duplication. Multiple features from each parameter declaration.
7. **Robust**: Get production-ready code with automatic interactive documentation (OpenAPI and JSON Schema).
8. **Standards-based**: Based on and fully compatible with the open standards for APIs: OpenAPI and JSON Schema.

## Asynchronous Code with ASGI

FastAPI is built on top of Starlette for the web parts and Pydantic for the data parts. It runs on an ASGI (Asynchronous Server Gateway Interface) server like Uvicorn or Hypercorn.

To define an asynchronous path operation, use standard Python `async def`:

```python
from fastapi import FastAPI

app = FastAPI(title="Documentation API", version="1.0.0")

@app.get("/")
async def read_root():
    return {"message": "Hello World"}
```

If your code interacts with third-party libraries that communicate over network or database drivers using `async` (such as `httpx` or `asyncpg`), use `async def`. If you use synchronous/blocking libraries (like standard `requests` or `sqlalchemy` without async drivers), use regular `def`:

```python
@app.get("/sync-endpoint")
def read_sync():
    # FastAPI automatically runs synchronous path operations in an external thread pool
    # so they do not block the main asyncio event loop!
    return {"status": "executed in threadpool"}
```

## Running the Application

FastAPI applications are served using an ASGI server such as `uvicorn`:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag enables auto-reload on file changes, recommended for local development only.

## Interactive API Documentation

When you run a FastAPI application, two interactive API documentations are automatically generated at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
