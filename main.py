import uvicorn

if __name__ == "__main__":
    pass
    app = "app.server:app"
    uvicorn.run(
        app,
        port=3000,
        host="0.0.0.0",
        workers=1,
    )
