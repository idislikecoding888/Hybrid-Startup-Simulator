from fastapi import FastAPI

from backend.api.routes import simulation, state, metrics, logs

app = FastAPI(title="Startup Simulator API")

# Register routes
app.include_router(simulation.router, prefix="/simulation", tags=["Simulation"])
app.include_router(state.router, prefix="/state", tags=["State"])
app.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])
app.include_router(logs.router, prefix="/logs", tags=["Logs"])


@app.get("/")
def root():
    return {"message": "Startup Simulator Running 🚀"}