import os
from typing import Literal

import numpy as np
import pandas as pd

from ventilacia_ai.services.config_service import NOMENCLATURE_INDEX_FILE
from ventilacia_ai.services.env_loader import ensure_env_loaded
from ventilacia_ai.services.text_utils import normalize_name

ensure_env_loaded()

_embedding_model = None
_embeddings_cache: dict[str, np.ndarray] = {}

_nomenclature_matrix: np.ndarray | None = None
_nomenclature_norms: np.ndarray | None = None
_nomenclature_indices: list[int] = []


def _embeddings_provider() -> Literal["local", "gigachat"]:
    explicit = os.getenv("EMBEDDINGS_PROVIDER", "").strip().lower()
    if explicit in ("local", "sentence-transformers", "offline"):
        return "local"
    if explicit in ("gigachat", "giga", "api"):
        return "gigachat"
    if os.getenv("GIGACHAT_CREDENTIALS") or (
        os.getenv("GIGACHAT_CLIENT_ID") and os.getenv("GIGACHAT_AUTH_KEY")
    ):
        return "gigachat"
    return "local"


def _gigachat_embeddings_model() -> str:
    return (os.getenv("GIGACHAT_EMBEDDINGS_MODEL") or "Embeddings").strip() or "Embeddings"


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print("[EMBEDDINGS] Загрузка локальной модели sentence-transformers...")
        _embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        print("[EMBEDDINGS] ✅ Локальная модель загружена")
    return _embedding_model


def _embeddings_batch_size() -> int:
    try:
        n = int(os.getenv("EMBEDDINGS_BATCH_SIZE", "32"))
    except ValueError:
        n = 32
    return max(1, min(n, 100))


def prefill_embeddings_cache(texts: list[str]) -> None:
    """Заполняет кэш эмбеддингов батчами (для коррекций и входных позиций)."""
    norms_to_fetch: list[str] = []
    seen: set[str] = set()
    for t in texts:
        if t is None:
            continue
        raw = str(t).strip()
        if not raw:
            continue
        n = normalize_name(raw)
        if not n or n in _embeddings_cache or n in seen:
            continue
        seen.add(n)
        norms_to_fetch.append(n)

    if not norms_to_fetch:
        return

    if _embeddings_provider() == "gigachat":
        from ventilacia_ai.clients.gigachat_client import get_gigachat_client
        client = get_gigachat_client()
        model_name = _gigachat_embeddings_model()
        bs = _embeddings_batch_size()
        total = len(norms_to_fetch)
        for i in range(0, total, bs):
            chunk = norms_to_fetch[i : i + bs]
            resp = client.embeddings(chunk, model=model_name)
            if not resp.data or len(resp.data) != len(chunk):
                raise RuntimeError(
                    f"GigaChat embeddings: ожидалось {len(chunk)} векторов, пришло {len(resp.data or [])}"
                )
            for emb_obj in resp.data:
                j = emb_obj.index
                if j < 0 or j >= len(chunk):
                    raise RuntimeError(
                        f"GigaChat embeddings: неверный index={j} для батча длины {len(chunk)}"
                    )
                _embeddings_cache[chunk[j]] = np.asarray(emb_obj.embedding, dtype=np.float32)
        print(f"[EMBEDDINGS] Prefill: в кэш добавлено {total} уникальных строк (батчи по {bs})")
        return

    model = get_embedding_model()
    bs = max(_embeddings_batch_size(), 64)
    for i in range(0, len(norms_to_fetch), bs):
        chunk = norms_to_fetch[i : i + bs]
        vecs = model.encode(chunk, convert_to_numpy=True)
        for j, n in enumerate(chunk):
            _embeddings_cache[n] = vecs[j].astype(np.float32)
    print(f"[EMBEDDINGS] Prefill: локально {len(norms_to_fetch)} уникальных строк")


def get_text_embedding(text: str) -> np.ndarray:
    """Получает (и кэширует) вектор для текста."""
    text_normalized = normalize_name(text)
    if text_normalized in _embeddings_cache:
        return _embeddings_cache[text_normalized]

    if _embeddings_provider() == "gigachat":
        from ventilacia_ai.clients.gigachat_client import get_gigachat_client
        client = get_gigachat_client()
        model_name = _gigachat_embeddings_model()
        resp = client.embeddings([text_normalized], model=model_name)
        if not resp.data:
            raise RuntimeError("GigaChat embeddings: пустой ответ data")
        vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
        _embeddings_cache[text_normalized] = vec
        return vec

    model = get_embedding_model()
    embedding = model.encode(text_normalized, convert_to_numpy=True).astype(np.float32)
    _embeddings_cache[text_normalized] = embedding
    return embedding


# ---------------------------------------------------------------------------
# Индекс номенклатуры для семантического поиска Top-K кандидатов
# ---------------------------------------------------------------------------

def build_nomenclature_index(nomenclature_df: pd.DataFrame) -> None:
    """
    Строит матрицу эмбеддингов для всей номенклатуры.
    Сохраняет на диск для ускорения последующих запусков.
    """
    global _nomenclature_matrix, _nomenclature_norms, _nomenclature_indices

    if _try_load_index(nomenclature_df):
        return

    print("[EMBEDDINGS INDEX] Построение индекса номенклатуры...")
    names = nomenclature_df["name_normalized"].tolist()
    indices = nomenclature_df.index.tolist()

    if _embeddings_provider() == "gigachat":
        vectors = _build_index_gigachat(names)
    else:
        vectors = _build_index_local(names)

    _nomenclature_matrix = np.vstack(vectors).astype(np.float32)
    _nomenclature_norms = np.linalg.norm(_nomenclature_matrix, axis=1, keepdims=True)
    _nomenclature_norms = np.where(_nomenclature_norms == 0, 1e-10, _nomenclature_norms)
    _nomenclature_indices = indices

    try:
        np.savez_compressed(
            NOMENCLATURE_INDEX_FILE,
            matrix=_nomenclature_matrix,
            indices=np.array(indices, dtype=np.int64),
            count=np.array([len(nomenclature_df)]),
        )
        print(f"[EMBEDDINGS INDEX] ✅ Индекс сохранён ({len(indices)} позиций)")
    except Exception as e:
        print(f"[EMBEDDINGS INDEX] ⚠️ Не удалось сохранить индекс: {e}")


def _try_load_index(nomenclature_df: pd.DataFrame) -> bool:
    """Пытается загрузить ранее сохранённый индекс, если размер совпадает."""
    global _nomenclature_matrix, _nomenclature_norms, _nomenclature_indices
    try:
        if not os.path.exists(NOMENCLATURE_INDEX_FILE):
            return False
        data = np.load(NOMENCLATURE_INDEX_FILE)
        saved_count = int(data["count"][0])
        if saved_count != len(nomenclature_df):
            print(
                f"[EMBEDDINGS INDEX] Размер номенклатуры изменился "
                f"({saved_count} → {len(nomenclature_df)}), пересоздаём индекс"
            )
            return False
        _nomenclature_matrix = data["matrix"].astype(np.float32)
        _nomenclature_indices = data["indices"].tolist()
        _nomenclature_norms = np.linalg.norm(_nomenclature_matrix, axis=1, keepdims=True)
        _nomenclature_norms = np.where(_nomenclature_norms == 0, 1e-10, _nomenclature_norms)
        print(f"[EMBEDDINGS INDEX] ✅ Индекс загружен с диска ({saved_count} позиций)")
        return True
    except Exception as e:
        print(f"[EMBEDDINGS INDEX] Не удалось загрузить индекс: {e}")
        return False


def _build_index_gigachat(names: list[str]) -> list[np.ndarray]:
    from ventilacia_ai.clients.gigachat_client import get_gigachat_client
    client = get_gigachat_client()
    model_name = _gigachat_embeddings_model()
    bs = _embeddings_batch_size()
    vectors: list[np.ndarray] = []
    total = len(names)
    for i in range(0, total, bs):
        chunk = names[i : i + bs]
        resp = client.embeddings(chunk, model=model_name)
        if not resp.data or len(resp.data) != len(chunk):
            raise RuntimeError(f"GigaChat index: ожидалось {len(chunk)} векторов")
        batch_vecs = [None] * len(chunk)
        for emb_obj in resp.data:
            batch_vecs[emb_obj.index] = np.asarray(emb_obj.embedding, dtype=np.float32)
        vectors.extend(batch_vecs)
        if (i + bs) % 1000 < bs:
            print(f"[EMBEDDINGS INDEX] GigaChat: {min(i + bs, total)}/{total}")
    return vectors


def _build_index_local(names: list[str]) -> list[np.ndarray]:
    model = get_embedding_model()
    bs = 256
    vectors: list[np.ndarray] = []
    total = len(names)
    for i in range(0, total, bs):
        chunk = names[i : i + bs]
        vecs = model.encode(chunk, convert_to_numpy=True, show_progress_bar=False)
        vectors.extend([v.astype(np.float32) for v in vecs])
        if (i + bs) % 5000 < bs:
            print(f"[EMBEDDINGS INDEX] Local: {min(i + bs, total)}/{total}")
    return vectors


def find_top_k_candidates(
    query: str,
    nomenclature_df: pd.DataFrame,
    top_k: int = 30,
) -> pd.DataFrame:
    """
    Находит top_k ближайших позиций номенклатуры по cosine similarity
    к эмбеддингу запроса. Возвращает срез DataFrame.
    """
    if _nomenclature_matrix is None:
        return nomenclature_df.head(top_k)

    query_vec = get_text_embedding(query).reshape(1, -1).astype(np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return nomenclature_df.head(top_k)

    similarities = (_nomenclature_matrix @ query_vec.T).flatten() / (
        _nomenclature_norms.flatten() * query_norm
    )

    k = min(top_k, len(similarities))
    top_indices_pos = np.argpartition(-similarities, k)[:k]
    top_indices_pos = top_indices_pos[np.argsort(-similarities[top_indices_pos])]

    df_indices = [_nomenclature_indices[i] for i in top_indices_pos]
    valid_indices = [i for i in df_indices if i in nomenclature_df.index]
    if not valid_indices:
        return nomenclature_df.head(top_k)
    return nomenclature_df.loc[valid_indices]


def nomenclature_index_ready() -> bool:
    return _nomenclature_matrix is not None


def initialize_embeddings() -> None:
    """Проверка доступности эмбеддингов при старте."""
    try:
        if _embeddings_provider() == "gigachat":
            from ventilacia_ai.clients.gigachat_client import get_gigachat_client
            client = get_gigachat_client()
            model_name = _gigachat_embeddings_model()
            client.embeddings(["__startup__"], model=model_name)
            print(f"[EMBEDDINGS] ✅ GigaChat API, модель «{model_name}» готова к работе")
        else:
            _ = get_embedding_model()
            print("[EMBEDDINGS] ✅ Локальная система векторного поиска готова")
    except Exception as e:
        print(f"[EMBEDDINGS] ⚠️ Не удалось инициализировать векторный поиск: {e}")
        print(
            "[EMBEDDINGS] Продолжаем работу БЕЗ эмбеддингов "
            "(будут доступны только локальные правила и GigaChat для чата)."
        )
