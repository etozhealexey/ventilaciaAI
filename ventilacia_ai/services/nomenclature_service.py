import pandas as pd

from ventilacia_ai.services.text_utils import normalize_name

nomenclature_df: pd.DataFrame | None = None


def load_nomenclature() -> bool:
    """Загружает номенклатуру из CSV и предрассчитывает нормализованные имена."""
    global nomenclature_df
    try:
        for encoding in ("utf-8", "cp1251", "utf-8-sig"):
            try:
                nomenclature_df = pd.read_csv(
                    "номенклатура.csv",
                    encoding=encoding,
                    quotechar='"',
                    on_bad_lines="skip",
                )
                break
            except Exception:
                continue
        else:
            raise RuntimeError("Не удалось прочитать CSV ни в одной кодировке")

        nomenclature_df = nomenclature_df.dropna(subset=["Наименование полное"])
        nomenclature_df["Наименование полное"] = (
            nomenclature_df["Наименование полное"].astype(str).str.strip()
        )
        nomenclature_df["name_lower"] = (
            nomenclature_df["Наименование полное"].str.lower().str.strip()
        )
        nomenclature_df["name_normalized"] = nomenclature_df[
            "Наименование полное"
        ].apply(normalize_name)

        print(f"Загружено {len(nomenclature_df)} позиций номенклатуры")
        return True
    except Exception as e:
        print(f"Ошибка загрузки номенклатуры: {e}")
        return False
