from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ventilacia_ai.api.controllers.training_controller import save_user_correction
from ventilacia_ai.models.schemas import CorrectionPayload
from ventilacia_ai.services import training_store, nomenclature_service
from ventilacia_ai.services.embeddings_service import nomenclature_index_ready

router = APIRouter()


@router.post("/api/correct")
async def correct(payload: CorrectionPayload) -> JSONResponse:
    return JSONResponse(save_user_correction(payload))


@router.get("/api/training/stats")
async def training_stats() -> JSONResponse:
    training_data = training_store.load_training_data()
    corrections = training_data.get("corrections", [])
    total_corrections = len(corrections)

    return JSONResponse(
        {
            "total_corrections": total_corrections,
            "total_examples": total_corrections,
            "corrected_examples": total_corrections,
            "average_confidence": 1.0 if total_corrections > 0 else 0.0,
            "vector_search_enabled": nomenclature_index_ready(),
        }
    )


@router.get("/api/status")
async def status() -> JSONResponse:
    training_data = training_store.load_training_data()
    corrections = training_data.get("corrections", [])

    return JSONResponse(
        {
            "nomenclature_loaded": nomenclature_service.nomenclature_df is not None,
            "items_count": len(nomenclature_service.nomenclature_df)
            if nomenclature_service.nomenclature_df is not None
            else 0,
            "training_examples": len(corrections),
            "corrections": len(corrections),
            "vector_search_enabled": nomenclature_index_ready(),
        }
    )
