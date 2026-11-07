import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from shapely.geometry import Polygon
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from data_loading import load_records, records_to_frame


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
AMUR_BOUNDS = {"lon_min": 119, "lon_max": 135, "lat_min": 48, "lat_max": 58}
SEVERITY_COLORS = {"Легкий": "#4C88B7", "Тяжёлый": "#D49728", "С погибшими": "#B84242"}
MONTH_NAMES = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]

st.set_page_config(page_title="Атлас ДТП Амурской области", page_icon="◈", layout="wide")


@st.cache_data(show_spinner="Читаем выгрузку…")
def read_data(path: str, modified_ns: int, size: int) -> tuple[list[dict], pd.DataFrame]:
    records = load_records(Path(path))
    return records, records_to_frame(records)


def strict_map_points(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame["map_valid"]
        & frame["longitude"].between(AMUR_BOUNDS["lon_min"], AMUR_BOUNDS["lon_max"])
        & frame["latitude"].between(AMUR_BOUNDS["lat_min"], AMUR_BOUNDS["lat_max"])
    ].copy()


@st.cache_data(show_spinner="Собираем гексагональную карту…")
def hex_aggregate(points: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Aggregate valid points into 1 km-radius hexagons in a metric CRS."""
    radius_m = 1_000
    metric_crs = "EPSG:3576"
    point_gdf = gpd.GeoDataFrame(
        points[["id", "severity"]].copy(),
        geometry=gpd.points_from_xy(points["longitude"], points["latitude"]),
        crs="EPSG:4326",
    ).to_crs(metric_crs)
    min_x, min_y, max_x, max_y = point_gdf.total_bounds
    x_step, y_step = 1.5 * radius_m, np.sqrt(3) * radius_m
    angles = np.radians(np.arange(0, 360, 60))
    hexagons = []
    for column, center_x in enumerate(np.arange(min_x - radius_m, max_x + radius_m, x_step)):
        y_offset = (column % 2) * y_step / 2
        for center_y in np.arange(min_y - radius_m + y_offset, max_y + radius_m, y_step):
            hexagons.append(
                Polygon(
                    [
                        (center_x + radius_m * np.cos(angle), center_y + radius_m * np.sin(angle))
                        for angle in angles
                    ]
                )
            )
    hex_grid = gpd.GeoDataFrame(geometry=hexagons, crs=metric_crs)
    joined = gpd.sjoin(point_gdf, hex_grid, how="inner", predicate="within")
    summary = joined.groupby("index_right").agg(
        ДТП=("id", "size"),
        ДТП_с_погибшими=("severity", lambda values: (values == "С погибшими").sum()),
    )
    hex_map = hex_grid.join(summary, how="inner").to_crs("EPSG:4326").reset_index(drop=True)
    return json.loads(hex_map.to_json()), hex_map


@st.cache_data(show_spinner="Считаем профили территорий…")
def territory_projections(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, np.ndarray]]:
    unlit = "В темное время суток, освещение отсутствует"
    grouped = frame.groupby("region")
    profiles = pd.DataFrame(index=sorted(frame["region"].unique()))
    profiles["log_n"] = np.log1p(grouped.size())
    profiles["fatal_share"] = frame.assign(x=frame["severity"].eq("С погибшими")).groupby("region")["x"].mean()
    profiles["heavy_share"] = frame.assign(x=frame["severity"].eq("Тяжёлый")).groupby("region")["x"].mean()
    profiles["pedestrian_share"] = frame.assign(x=frame["category"].eq("Наезд на пешехода")).groupby("region")["x"].mean()
    profiles["collision_share"] = frame.assign(x=frame["category"].eq("Столкновение")).groupby("region")["x"].mean()
    profiles["runoff_share"] = frame.assign(x=frame["category"].eq("Съезд с дороги")).groupby("region")["x"].mean()
    profiles["rollover_share"] = frame.assign(x=frame["category"].eq("Опрокидывание")).groupby("region")["x"].mean()
    profiles["night_unlit_share"] = frame.assign(x=frame["light"].eq(unlit)).groupby("region")["x"].mean()
    profiles["mean_participants"] = grouped["participants_count"].mean()

    scaled = StandardScaler().fit_transform(profiles)
    pca = PCA(n_components=2)
    pca_xy = pca.fit_transform(scaled)
    pca_frame = pd.DataFrame(pca_xy, columns=["PC1", "PC2"])
    pca_frame["Территория"] = profiles.index
    pca_frame["Число ДТП"] = grouped.size().reindex(profiles.index).to_numpy()
    pca_frame["Доля с погибшими, %"] = 100 * profiles["fatal_share"].to_numpy()
    loadings = pd.DataFrame(pca.components_.T, index=profiles.columns, columns=["PC1", "PC2"])
    tsne = {
        perplexity: TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=67,
            init="pca",
            learning_rate="auto",
            max_iter=1_000,
        ).fit_transform(scaled)
        for perplexity in (5, 10)
    }
    return pca_frame, loadings, tsne


def plotly_defaults(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        font={"family": "Arial, sans-serif", "color": "#1E2B3A"},
        title={"x": 0.02, "xanchor": "left"},
    )
    return fig


def main() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #F7F8F6; }
        h1, h2, h3 { color: #172B3A; }
        [data-testid="stMetric"] { background: #FFFFFF; border-left: 4px solid #D49728;
            padding: 0.7rem 0.85rem; border-radius: 0.35rem; }
        .insight { background: #EAF1F4; border-left: 4px solid #4C88B7; padding: 0.8rem 1rem;
            border-radius: 0.25rem; margin: 0.4rem 0 1rem; color: #1E2B3A; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Атлас ДТП Амурской области")
    st.caption("Исследование зарегистрированных ДТП: динамика, сценарии, география и профили территорий")

    files = sorted(DATA_DIR.glob("*.geojson")) + sorted(DATA_DIR.glob("*.json"))
    files = [path for path in files if not path.name.endswith(".source.json")]
    if not files:
        st.warning("В data/raw пока нет выгрузки. Импортируйте GeoJSON региона по инструкции в data/README.md.")
        return

    with st.sidebar:
        st.header("Срез данных")
        selected_path = st.selectbox("Выгрузка", files, format_func=lambda path: path.name)
    try:
        stat = selected_path.stat()
        records, frame = read_data(str(selected_path), stat.st_mtime_ns, stat.st_size)
    except (OSError, TypeError, ValueError) as error:
        st.error(f"Не удалось прочитать выгрузку: {error}")
        return
    if frame.empty:
        st.warning("Выгрузка не содержит записей.")
        return

    dated = frame[frame["datetime"].notna()].copy()
    first, last = dated["datetime"].min().date(), dated["datetime"].max().date()
    with st.sidebar:
        period = st.date_input("Период", value=(first, last), min_value=first, max_value=last)
        regions = sorted(frame["region"].dropna().astype(str).unique())
        categories = sorted(frame["category"].dropna().astype(str).unique())
        selected_regions = st.multiselect("Территории", regions, placeholder="Все территории")
        selected_categories = st.multiselect("Типы ДТП", categories, placeholder="Все типы")
        st.caption("Период и оба фильтра меняют основные графики и карту.")
    if len(period) != 2:
        st.info("Выберите начальную и конечную дату.")
        return
    start, end = period
    current = dated[(dated["datetime"] >= pd.Timestamp(start)) & (dated["datetime"] < pd.Timestamp(end) + pd.Timedelta(days=1))].copy()
    if selected_regions:
        current = current[current["region"].isin(selected_regions)]
    if selected_categories:
        current = current[current["category"].isin(selected_categories)]
    if current.empty:
        st.warning("В выбранном срезе нет ДТП. Измените фильтры.")
        return

    current["year"] = current["datetime"].dt.year
    current["month"] = current["datetime"].dt.month
    current["hour"] = current["datetime"].dt.hour
    current["fatal"] = current["severity"].eq("С погибшими")
    map_points = strict_map_points(current)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ДТП в срезе", f"{len(current):,}")
    m2.metric("ДТП с погибшими", f"{int(current['fatal'].sum()):,}")
    m3.metric("Доля с погибшими", f"{100 * current['fatal'].mean():.1f}%")
    m4.metric("Точек для карты", f"{len(map_points):,}")
    st.caption(f"Срез: {start:%d.%m.%Y} — {end:%d.%m.%Y}. Для карты исключаются точки вне приблизительной рамки области.")

    overview, conditions, geography, projections, before_after, source = st.tabs(
        ["Динамика и состав", "Условия и гипотезы", "Карта", "PCA и t-SNE", "До и после", "Исходные записи"]
    )

    with overview:
        st.subheader("Когда и какие ДТП регистрируются")
        complete = current[current["year"] < 2026]
        yearly = complete.groupby("year").size().rename("ДТП").reset_index().rename(columns={"year": "Год"})
        fig = px.bar(yearly, x="Год", y="ДТП", text="ДТП", title="Число ДТП по полным годам")
        fig.update_traces(textposition="outside")
        st.plotly_chart(plotly_defaults(fig), width="stretch")

        monthly = (
            current.set_index("datetime").resample("MS").size().rename("ДТП").reset_index()
            .rename(columns={"datetime": "Месяц"})
        )
        fig = px.line(monthly, x="Месяц", y="ДТП", markers=True, title="Число ДТП по месяцам")
        st.plotly_chart(plotly_defaults(fig), width="stretch")

        left, right = st.columns(2)
        with left:
            hours = (
                current.groupby("hour").size().reindex(range(24), fill_value=0).rename("ДТП").reset_index()
                .rename(columns={"hour": "Час"})
            )
            fig = px.bar(hours, x="Час", y="ДТП", title="Распределение ДТП по часам")
            fig.update_xaxes(dtick=1)
            st.plotly_chart(plotly_defaults(fig), width="stretch")
        with right:
            season = complete.groupby("month").size().div(max(complete["year"].nunique(), 1)).reindex(range(1, 13), fill_value=0)
            season = season.rename("Среднее число ДТП").reset_index().rename(columns={"month": "Номер месяца"})
            season["Месяц"] = MONTH_NAMES
            fig = px.bar(season, x="Месяц", y="Среднее число ДТП", title="Среднее число ДТП по месяцам")
            st.plotly_chart(plotly_defaults(fig), width="stretch")

        left, right = st.columns(2)
        with left:
            types = current["category"].value_counts().head(10).sort_values().rename_axis("Тип ДТП").reset_index(name="ДТП")
            fig = px.bar(types, x="ДТП", y="Тип ДТП", orientation="h", title="Наиболее частые типы ДТП")
            st.plotly_chart(plotly_defaults(fig, 500), width="stretch")
        with right:
            severity = current["severity"].value_counts().rename_axis("Последствия").reset_index(name="ДТП")
            fig = px.bar(
                severity, x="Последствия", y="ДТП", color="Последствия", text="ДТП",
                color_discrete_map=SEVERITY_COLORS, title="Тяжесть последствий",
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(plotly_defaults(fig, 500), width="stretch")

        regions_frame = current["region"].value_counts().head(12).sort_values().rename_axis("Территория").reset_index(name="ДТП")
        fig = px.bar(regions_frame, x="ДТП", y="Территория", orientation="h", title="Территории с наибольшим числом ДТП")
        st.plotly_chart(plotly_defaults(fig, 520), width="stretch")

        st.markdown(
            "<div class='insight'>Количество ДТП — не оценка риска: для сравнения риска между территориями нужны данные о населении или транспортном потоке.</div>",
            unsafe_allow_html=True,
        )

    with conditions:
        st.subheader("Наезды на пешеходов: освещение и тяжесть")
        light_on = "В темное время суток, освещение включено"
        light_off = "В темное время суток, освещение отсутствует"
        severity_light = (
            pd.crosstab(current["light"], current["severity"], normalize="index")
            .mul(100).reset_index().melt(id_vars="light", var_name="Последствия", value_name="Доля, %")
        )
        fig = px.bar(
            severity_light, x="light", y="Доля, %", color="Последствия", barmode="stack",
            color_discrete_map=SEVERITY_COLORS, title="Структура тяжести ДТП при разном освещении",
            labels={"light": "Условия освещения"},
        )
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(plotly_defaults(fig, 560), width="stretch")

        pedestrians = current[current["category"].eq("Наезд на пешехода") & current["light"].isin([light_on, light_off])]
        if pedestrians.empty:
            st.info("В выбранном срезе нет подходящих ночных наездов на пешеходов.")
        else:
            rate = pedestrians.groupby("light")["fatal"].agg(["mean", "size"]).reindex([light_on, light_off]).dropna().reset_index()
            rate["Освещение"] = rate["light"].map({light_on: "Включено", light_off: "Отсутствует"})
            rate["Доля ДТП с погибшими, %"] = 100 * rate["mean"]
            fig = px.bar(
                rate, x="Освещение", y="Доля ДТП с погибшими, %", color="Освещение", text="Доля ДТП с погибшими, %",
                hover_data={"size": True, "light": False, "mean": False},
                color_discrete_map={"Включено": "#4C88B7", "Отсутствует": "#B84242"},
                title="Доля ДТП с погибшими при ночных наездах на пешеходов",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(plotly_defaults(fig), width="stretch")

        type_light = current[current["light"].isin([light_on, light_off])].copy()
        type_light = type_light[type_light["category"].isin(["Столкновение", "Наезд на пешехода", "Съезд с дороги"])]
        if not type_light.empty:
            by_type = type_light.groupby(["category", "light"])["fatal"].mean().mul(100).reset_index(name="Доля ДТП с погибшими, %")
            by_type["Освещение"] = by_type["light"].map({light_on: "Включено", light_off: "Отсутствует"})
            fig = px.bar(
                by_type, x="category", y="Доля ДТП с погибшими, %", color="Освещение", barmode="group",
                color_discrete_map={"Включено": "#4C88B7", "Отсутствует": "#B84242"},
                title="Доля ДТП с погибшими ночью: сравнение внутри типа ДТП",
                labels={"category": "Тип ДТП"},
            )
            st.plotly_chart(plotly_defaults(fig), width="stretch")

        st.subheader("Проверенные результаты по полной выгрузке")
        one, two, three = st.columns(3)
        one.info("Наезды на пешеходов: 30,5% без освещения против 10,5% с освещением. Разность 19,9 п.п.; p < 0,001.")
        two.info("Съезды с дороги: 11,4% с погибшими против 7,0% у столкновений. Разность 4,4 п.п.; p < 0,001.")
        three.info("Съезды с дороги: 15,8% вне Благовещенска против 1,7% в городе. Разность 14,0 п.п.; p < 0,001.")
        st.caption("Это связи в полной выгрузке, а не доказанные причинные эффекты. Доверительные интервалы и расчёты — в ноутбуке.")

        full_years = current[current["year"] < 2026]
        runoff = full_years[full_years["category"].eq("Съезд с дороги")].copy()
        runoff["Период"] = np.where(runoff["month"].isin([12, 1, 2]), "Зима (дек–фев)", "Не зима (мар–ноя)")
        if not runoff.empty:
            month_counts = {"Зима (дек–фев)": max(full_years[full_years["month"].isin([12, 1, 2])]["year"].nunique() * 3, 1), "Не зима (мар–ноя)": max(full_years[~full_years["month"].isin([12, 1, 2])]["year"].nunique() * 9, 1)}
            season_runoff = runoff["Период"].value_counts().rename_axis("Период").reset_index(name="Всего съездов")
            season_runoff["Среднее съездов в месяц"] = season_runoff.apply(lambda row: row["Всего съездов"] / month_counts[row["Период"]], axis=1)
            fig = px.bar(season_runoff, x="Период", y="Среднее съездов в месяц", text="Среднее съездов в месяц", color="Период", title="Съезды с дороги зимой и в остальные месяцы (2015–2025)")
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            st.plotly_chart(plotly_defaults(fig), width="stretch")
            st.caption("Сравнение нормировано на число месяцев; по нему нельзя делать причинный вывод о гололёде без дорожных условий и транспортной экспозиции.")

        left, right = st.columns(2)
        with left:
            type_fatal = current[current["category"].isin(["Столкновение", "Съезд с дороги"])]
            type_fatal = type_fatal.groupby("category")["fatal"].agg(["mean", "size"]).reset_index()
            type_fatal["Доля ДТП с погибшими, %"] = 100 * type_fatal["mean"]
            fig = px.bar(type_fatal, x="category", y="Доля ДТП с погибшими, %", color="category", hover_data={"size": True, "mean": False}, title="Доля ДТП с погибшими: столкновения и съезды")
            st.plotly_chart(plotly_defaults(fig, 460), width="stretch")
        with right:
            city = current.assign(Территория=np.where(current["region"].eq("Благовещенск"), "Благовещенск", "Остальная область"))
            city["Съезд с дороги"] = city["category"].eq("Съезд с дороги")
            city_rate = city.groupby("Территория")["Съезд с дороги"].agg(["mean", "size"]).reset_index()
            city_rate["Доля съездов с дороги, %"] = 100 * city_rate["mean"]
            fig = px.bar(city_rate, x="Территория", y="Доля съездов с дороги, %", color="Территория", hover_data={"size": True, "mean": False}, title="Доля съездов: Благовещенск и область")
            st.plotly_chart(plotly_defaults(fig, 460), width="stretch")

    with geography:
        st.subheader("Где концентрируются ДТП")
        if map_points.empty:
            st.info("В этом срезе нет точек с пригодными координатами внутри рамки Амурской области.")
        else:
            map_mode = st.radio("Представление", ["Точки", "Гексагоны (радиус 1 км)"], horizontal=True)
            if map_mode == "Точки":
                points = map_points.sample(min(len(map_points), 5_000), random_state=67).copy()
                fig = px.scatter_map(
                    points, lat="latitude", lon="longitude", color="severity",
                    color_discrete_map=SEVERITY_COLORS,
                    hover_data=["datetime", "region", "category", "participants_count"],
                    center={"lat": float(map_points["latitude"].median()), "lon": float(map_points["longitude"].median())},
                    zoom=5, height=650, opacity=0.58, map_style="open-street-map",
                    title="Точечная карта ДТП",
                )
                st.plotly_chart(plotly_defaults(fig, 650), width="stretch")
                if len(map_points) > 5_000:
                    st.caption("Для скорости отображается случайная подвыборка 5 000 точек; показатели и гексагоны используют полный срез.")
            else:
                geojson, hex_map = hex_aggregate(map_points[["id", "severity", "longitude", "latitude"]])
                fig = px.choropleth_map(
                    hex_map, geojson=geojson, locations=hex_map.index.astype(str), featureidkey="id",
                    color="ДТП", hover_data=["ДТП_с_погибшими"], color_continuous_scale="YlOrRd",
                    center={"lat": float(map_points["latitude"].median()), "lon": float(map_points["longitude"].median())},
                    zoom=5, height=650, opacity=0.72, map_style="open-street-map",
                    title="Концентрация ДТП по гексагонам радиусом 1 км",
                    labels={"ДТП": "Число ДТП", "ДТП_с_погибшими": "С погибшими"},
                )
                st.plotly_chart(plotly_defaults(fig, 650), width="stretch")
                st.caption(f"Заполнено гексагонов: {len(hex_map):,}. Подложка: © OpenStreetMap contributors.")

    with projections:
        st.subheader("Профили территорий по полной выгрузке")
        st.caption("Проекции рассчитаны заранее для всех 24 территорий и не меняются от фильтров выше.")
        pca_frame, loadings, tsne_results = territory_projections(frame)
        left, right = st.columns([3, 2])
        with left:
            fig = px.scatter(
                pca_frame, x="PC1", y="PC2", size="Число ДТП", color="Доля с погибшими, %",
                color_continuous_scale="YlOrRd", hover_name="Территория", size_max=34,
                title="PCA: профили территорий (72,3% сохранённой дисперсии)",
            )
            fig.add_hline(y=0, line_width=1, line_color="#D1D5DB")
            fig.add_vline(x=0, line_width=1, line_color="#D1D5DB")
            st.plotly_chart(plotly_defaults(fig, 650), width="stretch")
        with right:
            readable_loadings = loadings.rename(
                index={
                    "log_n": "Число ДТП", "fatal_share": "Доля с погибшими", "heavy_share": "Тяжёлые ДТП",
                    "pedestrian_share": "Наезды на пешеходов", "collision_share": "Столкновения",
                    "runoff_share": "Съезды с дороги", "rollover_share": "Опрокидывания",
                    "night_unlit_share": "Ночь без освещения", "mean_participants": "Участники",
                }
            )
            st.markdown("PC1 отделяет территории со съездами, опрокидываниями и ночными ДТП без освещения от территорий со столкновениями и наездами на пешеходов.")
            st.dataframe(readable_loadings.round(2), width="stretch")

        fig = make_subplots(rows=1, cols=2, subplot_titles=["perplexity = 5", "perplexity = 10"])
        for column, perplexity in enumerate((5, 10), start=1):
            coords = tsne_results[perplexity]
            fig.add_trace(
                go.Scatter(
                    x=coords[:, 0], y=coords[:, 1], mode="markers", text=pca_frame["Территория"],
                    marker={"size": 10, "color": pca_frame["Доля с погибшими, %"], "colorscale": "YlOrRd", "showscale": column == 2},
                    hovertemplate="%{text}<extra></extra>", showlegend=False,
                ),
                row=1, col=column,
            )
        fig.update_layout(title="t-SNE: похожие локальные профили территорий, seed = 67", template="plotly_white", height=650)
        st.plotly_chart(fig, width="stretch")
        st.caption("У t-SNE оси не имеют содержательной интерпретации: смотрим только на устойчивые соседства между двумя запусками.")

    with before_after:
        st.subheader("Как улучшение графика меняет чтение данных")
        st.caption("Сравнение построено на выбранном срезе и обновляется вместе с фильтрами.")
        all_types = current["category"].value_counts().rename_axis("Тип ДТП").reset_index(name="ДТП")
        before = px.bar(
            all_types, x="Тип ДТП", y="ДТП",
            title="До: все категории на вертикальной оси",
            labels={"Тип ДТП": "Тип ДТП", "ДТП": "Количество ДТП"},
        )
        before.update_xaxes(tickangle=-45)
        before.update_layout(showlegend=False)

        main_types = current["category"].value_counts().head(8).index
        improved_types = (
            current.assign(Тип=np.where(current["category"].isin(main_types), current["category"], "Прочие"))["Тип"]
            .value_counts().sort_values().rename_axis("Тип ДТП").reset_index(name="ДТП")
        )
        improved_types["Доля, %"] = 100 * improved_types["ДТП"] / max(len(current), 1)
        improved_types["Подпись"] = improved_types.apply(
            lambda row: f"{row['ДТП']:,} ({row['Доля, %']:.1f}%)", axis=1
        )
        after = px.bar(
            improved_types, x="ДТП", y="Тип ДТП", orientation="h", text="Подпись", color="ДТП",
            color_continuous_scale=["#B9D7EA", "#2E6F9E"],
            title="После: горизонтальные бары, основные категории и доли",
            labels={"Тип ДТП": "Тип ДТП", "ДТП": "Количество ДТП"},
        )
        after.update_traces(textposition="outside")
        after.update_layout(coloraxis_showscale=False)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(plotly_defaults(before, 520), width="stretch")
        with right:
            st.plotly_chart(plotly_defaults(after, 520), width="stretch")
        st.markdown(
            "<div class='insight'><b>Что стало лучше видно:</b> горизонтальная шкала не обрезает длинные названия категорий; "
            "сортировка сразу показывает порядок частот; объединение редких категорий в «Прочие» снижает визуальный шум; "
            "подписи одновременно показывают абсолютное число ДТП и долю от выбранного среза.</div>",
            unsafe_allow_html=True,
        )

    with source:
        st.subheader("Исходные записи выбранного среза")
        columns = ["source_row", "id", "datetime", "region", "category", "severity", "light", "participants_count", "injured_count", "dead_count", "latitude", "longitude"]
        st.dataframe(current[columns].head(500), hide_index=True, width="stretch")
        st.download_button(
            "Скачать выбранный срез в CSV",
            current[columns].to_csv(index=False).encode("utf-8-sig"),
            file_name="selected_crashes.csv",
            mime="text/csv",
        )
        lookup = current.set_index("source_row")
        source_row = st.selectbox(
            "Открыть исходную запись",
            current["source_row"].tolist(),
            format_func=lambda value: f"Строка {value} · ID {lookup.at[value, 'id']}",
        )
        st.json(records[int(source_row)], expanded=False)


if __name__ == "__main__":
    main()
