import os

UPLOAD_FOLDER = "uploads"
REPORTS_FOLDER = "reports"
TRAINING_DATA_FILE = "training_data.json"
NOMENCLATURE_INDEX_FILE = "nomenclature_embeddings.npz"
ALLOWED_EXTENSIONS = {"xlsx", "xls"}
ALLOWED_EXTENSIONS_SMART = {"xlsx", "xls", "docx", "pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

