from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from data_loading import load_records, records_to_frame

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
FILTER_PREFIX = "filter_"
LABELS = {"region": "Территория", "category": "Тип ДТП", "severity": "Последствия"}

st.set_page_config(page_title="Атлас аварийности", page_icon="🗺️", layout="wide")


@st.cache_data(show_spinner="Читаем выгрузку…")
def read_data(path: str, modified_ns: int, size: int) -> tuple[list[dict], pd.DataFrame]:
    records = load_records(Path(path))
    return records, records_to_frame(records)


def reset_filters() -> None:
    for key in list(st.session_state):
        if str(key).startswith(FILTER_PREFIX):
            del st.session_state[key]


def main() -> None:
    st.title("Атлас аварийности")
    st.caption("Каркас ДЗ 2 · данные вашего региона")
    st.info(
        "Здесь можно посмотреть выгрузку и выбрать данные для анализа. Добавьте свои графики, "
        "проверки гипотез и выводы по заданию."
    )
    files = sorted(DATA_DIR.glob("*.geojson")) + sorted(DATA_DIR.glob("*.json"))
    files = [p for p in files if not p.name.endswith(".source.json")]
    if not files:
        st.warning("В data/raw пока нет выгрузки. Скачайте файл своего региона.")
        st.code(
            "python data/assign_region.py YOUR_GITHUB_USERNAME\n"
            "python data/import_data.py /path/to/region.geojson.zip",
            language="bash",
        )
        st.link_button("Открыть данные «Карты ДТП»", "https://dtp-stat.ru/opendata/")
        st.markdown("Инструкция находится в `data/README.md` репозитория.")
        return

    selected_path = st.sidebar.selectbox(
        "Выгрузка", files, format_func=lambda p: p.name,
        key="source_file", on_change=reset_filters,
    )
    try:
        stat = selected_path.stat()
        records, frame = read_data(str(selected_path), stat.st_mtime_ns, stat.st_size)
    except (ValueError, OSError, TypeError) as error:
        st.error(f"Не удалось прочитать выгрузку: {error}")
        return
    if frame.empty:
        st.warning("Файл содержит пустой список записей.")
        return

    invalid_dates = int(frame["datetime"].isna().sum())
    duplicate_ids = int(frame.loc[frame["id"].notna(), "id"].duplicated().sum())
    st.sidebar.caption(f"Исходных записей: {len(frame):,}")
    st.sidebar.caption(
        f"Без распознанной даты: {invalid_dates:,}. "
        f"Повторных непустых ID: {duplicate_ids:,}. Дубликаты не удалены."
    )
    dated = frame[frame["datetime"].notna()]
    if dated.empty:
        st.error("Нет распознанных дат. Проверьте исходный формат и загрузчик.")
        st.dataframe(frame.head(20), hide_index=True)
        return

    first, last = dated["datetime"].min().date(), dated["datetime"].max().date()
    st.sidebar.button("Сбросить фильтры", on_click=reset_filters)
    period = st.sidebar.date_input(
        "Период", value=(first, last), min_value=first, max_value=last,
        key="filter_period",
    )
    if len(period) != 2:
        st.info("Выберите начало и конец периода.")
        return
    start, end = period
    current = dated[
        (dated["datetime"] >= pd.Timestamp(start))
        & (dated["datetime"] < pd.Timestamp(end) + pd.Timedelta(days=1))
    ].copy()
    for column, title in list(LABELS.items())[:2]:
        include_all = st.sidebar.checkbox(
            "Все территории" if column == "region" else "Все типы ДТП",
            value=True, key=f"filter_all_{column}",
        )
        if include_all:
            continue
        values = frame[column]
        options = sorted(values.dropna().astype(str).unique().tolist())
        if values.isna().any():
            options.append(None)
        selected = st.sidebar.multiselect(
            title, options, default=[],
            format_func=lambda value: "〈значение отсутствует〉" if value is None else value,
            key=f"filter_{column}",
        )
        mask = current[column].astype("string").isin([v for v in selected if v is not None])
        if None in selected:
            mask |= current[column].isna()
        current = current[mask].copy()

    st.caption(
        f"Период: {start:%d.%m.%Y} - {end:%d.%m.%Y}. "
        "Ниже показаны данные с учётом фильтров. Исходные записи посчитаны "
        "без автоматической очистки."
    )
    if invalid_dates:
        st.warning(f"Из фильтра по времени исключено записей без даты: {invalid_dates:,}.")
    if current.empty:
        st.warning("В выбранном срезе нет записей. Измените или сбросьте фильтры.")
        return
    a, b, c = st.columns(3)
    a.metric("Записей в срезе", f"{len(current):,}")
    b.metric("Уникальных непустых ID", f"{current['id'].nunique():,}")
    c.metric("Без пригодных для карты координат", f"{int((~current['map_valid']).sum()):,}")
    if len(current) < 30:
        st.warning(
            "В выборке меньше 30 записей. Посмотрите, как отдельные ДТП влияют на результат. "
            "Число 30 здесь выбрано для напоминания, а не как критерий надёжности."
        )

    overview, geography, source = st.tabs(["Динамика", "Карта", "Исходные записи"])
    with overview:
        counts = current.set_index("datetime").resample("MS").size().rename("records")
        months = pd.date_range(pd.Timestamp(start).to_period("M").start_time,
                               pd.Timestamp(end).to_period("M").start_time, freq="MS")
        counts = counts.reindex(months, fill_value=0).rename_axis("month").reset_index()
        fig = px.line(counts, x="month", y="records", markers=True,
                      labels={"month": "Месяц", "records": "Число исходных записей"})
        st.subheader("Число записей по месяцам")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Ноль на графике означает, что за этот месяц в выбранных данных нет записей. "
            "Первый и последний месяцы могут быть неполными. "
            "Перед сравнением проверьте, за все ли месяцы данные собраны полностью."
        )
    with geography:
        points = current[current["map_valid"]].copy()
        if points.empty:
            st.info("В срезе нет пригодных для карты координат.")
        else:
            if len(points) > 5000:
                points = points.sample(5000, random_state=42)
                st.warning(
                    "Для быстрого просмотра показана случайная подвыборка 5 000 точек. "
                    "Остальные показатели и CSV используют полный срез."
                )
            backdrop = st.checkbox("Показывать подложку карты", value=True)
            points["severity_display"] = points["severity"].fillna("〈значение отсутствует〉")
            fig = px.scatter_map(
                points, lat="latitude", lon="longitude", color="severity_display",
                color_discrete_map={
                    "Легкий": "#3579B8", "Тяжёлый": "#C18A16",
                    "С погибшими": "#B33440", "〈значение отсутствует〉": "#8D939C",
                },
                category_orders={"severity_display": ["Легкий", "Тяжёлый", "С погибшими"]},
                hover_data=["source_row", "id", "datetime", "category", "region"],
                labels={"severity_display": "Последствия", **LABELS},
                opacity=0.6, zoom=5,
                center={"lat": float(points["latitude"].median()),
                        "lon": float(points["longitude"].median())},
                map_style="open-street-map" if backdrop else "white-bg", height=540,
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Подложка: © OpenStreetMap contributors, https://www.openstreetmap.org/copyright. "
                "Координаты взяты из выгрузки; принадлежность региону здесь не проверена."
            )
    with source:
        columns = ["source_row", "id", "datetime", "region", "category", "severity",
                   "participants_count", "injured_count", "dead_count", "latitude", "longitude"]
        st.dataframe(current[columns].head(500), hide_index=True, width="stretch")
        st.caption("Первые 500 записей текущего среза; CSV содержит весь срез.")
        st.download_button(
            "Скачать текущий срез CSV", current[columns].to_csv(index=False).encode("utf-8-sig"),
            file_name="selected_records.csv", mime="text/csv",
        )
        lookup = current.set_index("source_row")
        chosen = st.selectbox(
            "Открыть исходную запись", current["source_row"].tolist(),
            format_func=lambda index: f"Строка {index} · ID {lookup.at[index, 'id']}",
        )
        st.json(records[int(chosen)], expanded=False)


if __name__ == "__main__":
    main()
