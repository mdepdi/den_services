from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.exceptions import HTTPException
from routes import intersite
from routes import fwa, utils
from core.config import settings
from pathlib import Path

import os

# SETUP FASTAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    debug=True,
    description="Design and Engineering API services for planning and design tools.",
    openapi_tags=[
        {
            "name": "Intersite",
            "description": "Endpoints related to intersite fiberization services."
        }
    ]
)

# MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# STATIC FILES
data_dir = settings.DATA_DIR
export_dir = settings.EXPORT_DIR
template_dir = os.path.join(data_dir, "template")

app.mount("/template", StaticFiles(directory=template_dir), name="template")
app.mount("/exports", StaticFiles(directory=export_dir), name="exports")

@app.get("/download-template/{fname}", tags=["Utils"])
async def download_template(fname: str):
    
    fpath = Path(f"{template_dir}/{fname}")
    if not fpath.is_file():
        raise HTTPException(404, "Template not found")

    return FileResponse(
        path=fpath,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=fpath.name,
    )

# ROUTERS
@app.get("/")
async def root():
    return {"message": "Welcome to the Design and Engineering API Services!"}

# INCLUDE ROUTERS
app.include_router(intersite.router, prefix="/intersite", tags=["Intersite"])
app.include_router(fwa.router, prefix="/fwa", tags=["Fixed Wireless Access (FWA)"])
app.include_router(utils.router, prefix="/utils", tags=["Utils"])

