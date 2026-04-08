"""Deployment entry point for the MSME OpenEnv app."""

from env.server import app


def main() -> None:
    """Run the FastAPI app with uvicorn."""
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
