import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path
from typing import Iterable, List, Optional

st.set_page_config(page_title="Optimization Study Dashboard", layout="wide")

DATA_DIR = Path("optuna")
STUDY_LABELS = {
    "Max Rounds": "max_rounds",
    "Multi Objective": "multi_objective",
    "Multi Objective (Old)": "multi_objective_old",
    "Milestone": "milestone",
}
ALGORITHM_LABELS = {
    "random": "Random",
    "tpe": "TPE",
    "nsga2-blend": "NSGA-II (blend)",
    "nsga2-uniform": "NSGA-II (uniform)",
}
ALGORITHM_COLORS = {
    "Random": "#1f77b4",
    "TPE": "#ff7f0e",
    "NSGA-II (blend)": "#2ca02c",
    "NSGA-II (uniform)": "#d62728",
}
PARAM_COLUMNS = [
    "params_enemyHp",
    "params_enemyMeleeDamage",
    "params_enemyShotDamage",
    "params_playerHp",
    "params_playerMeleeDamage",
    "params_playerShotDamage",
    "params_playerShotRange",
]


def chunk_list(items: List[str], chunk_size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def nearest_to_target(series: pd.Series, target: float) -> Optional[float]:
    non_null = series.dropna()
    if non_null.empty:
        return None
    closest_index = (non_null - target).abs().idxmin()
    return float(non_null.loc[closest_index])


def final_hp_closest_to_one(values: pd.Series) -> float:
    closest_value = nearest_to_target(values, 1.0)
    return closest_value if closest_value is not None else float("nan")


def add_jitter(values: pd.Series, fraction: float = 0.02) -> pd.Series:
    if values.empty:
        return values
    span = values.max() - values.min()
    if pd.isna(span) or span == 0:
        span = 1.0
    scale = span * fraction
    noise = np.random.uniform(-scale, scale, size=len(values))
    return values + noise


def identify_pareto(df: pd.DataFrame) -> pd.Series:
    ordered = df.sort_values(["final_hp", "max_rounds"], ascending=[True, False])
    best_rounds = float("-inf")
    pareto_indices: List[int] = []
    for index, row in ordered.iterrows():
        rounds = row.get("max_rounds")
        if pd.isna(rounds):
            continue
        if rounds >= best_rounds:
            pareto_indices.append(index)
            best_rounds = rounds
    mask = pd.Series(False, index=df.index)
    if pareto_indices:
        mask.loc[pareto_indices] = True
    return mask


@st.cache_data(show_spinner=False)
def load_study_data(study_key: str, _cache_key: float = 0) -> pd.DataFrame:
    study_path = DATA_DIR / study_key
    if not study_path.exists():
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for csv_path in study_path.glob("*_study.csv"):
        df = pd.read_csv(csv_path)
        algo_key = csv_path.stem.replace("_study", "")
        algorithm_name = ALGORITHM_LABELS.get(
            algo_key, algo_key.replace("-", " ").title()
        )
        df["algorithm"] = algorithm_name
        df["study"] = study_key

        if "duration" in df.columns:
            df["duration_seconds"] = pd.to_timedelta(
                df["duration"], errors="coerce"
            ).dt.total_seconds()

        if "value" in df.columns:
            df["max_rounds"] = df["value"]
        if "values_0" in df.columns:
            df["max_rounds"] = df["values_0"]
        if "values_1" in df.columns:
            df["final_hp"] = df["values_1"]

        if "final_hp_error" in df.columns and "final_hp" not in df.columns:
            df = df.rename(columns={"final_hp_error": "final_hp"})

        # Rename std columns for easier access
        if "user_attrs_rounds_std" in df.columns:
            df = df.rename(columns={"user_attrs_rounds_std": "max_rounds_std"})
        if "user_attrs_lost_players_std" in df.columns:
            df = df.rename(columns={"user_attrs_lost_players_std": "final_hp_std"})

        drop_cols = [col for col in df.columns if ":" in col]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    for param in PARAM_COLUMNS:
        if param in data.columns:
            data[param] = pd.to_numeric(data[param], errors="coerce")
    return data


def format_duration(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.1f} s"


st.title("Optimization Study Dashboard")
st.caption("Explore Optuna trials by study, algorithm, and parameter set.")

with st.sidebar:
    st.header("Controls")
    study_labels = list(STUDY_LABELS.keys())
    default_study_index = (
        study_labels.index("Milestone") if "Milestone" in study_labels else 0
    )
    study_label = st.selectbox("Study", study_labels, index=default_study_index)
    study_key = STUDY_LABELS[study_label]

    # Create cache key based on file modification times
    study_path = DATA_DIR / study_key
    cache_key = 0
    if study_path.exists():
        csv_files = list(study_path.glob("*_study.csv"))
        if csv_files:
            cache_key = max(f.stat().st_mtime for f in csv_files)

    data = load_study_data(study_key, cache_key)

    primary_metric_label = "Max Rounds"
    if study_key == "milestone":
        primary_metric_label = "Killed Enemies"

    if data.empty:
        st.warning("No trials found for the selected study.")
        st.stop()

    available_algorithms = sorted(data["algorithm"].unique())
    selected_algorithms = st.multiselect(
        "Algorithms",
        options=available_algorithms,
        default=available_algorithms,
    )

    if not selected_algorithms:
        st.warning("Pick at least one algorithm to continue.")
        st.stop()

    filtered = data[data["algorithm"].isin(selected_algorithms)].copy()

    if study_key == "milestone":
        filtered["Killed enemies"] = filtered["max_rounds"]
        metric_hover_key = "Killed enemies"
    else:
        metric_hover_key = "max_rounds"

    if filtered.empty:
        st.warning("No trials match the selected filters.")
        st.stop()

has_final_hp = "final_hp" in filtered.columns

metric_columns = 4 if has_final_hp else 3
metric_containers = st.columns(metric_columns)
metric_containers[0].metric("Trials", len(filtered))
metric_containers[1].metric(
    f"Best {primary_metric_label}", f"{filtered['max_rounds'].max():.2f}"
)
metric_containers[2].metric(
    f"Median {primary_metric_label}", f"{filtered['max_rounds'].median():.2f}"
)
if has_final_hp:
    closest_final_hp_value = nearest_to_target(filtered["final_hp"], 1.0)
    metric_containers[3].metric(
        "Final HP Closest to 1",
        (
            f"{closest_final_hp_value:.2f}"
            if closest_final_hp_value is not None
            and not pd.isna(closest_final_hp_value)
            else "-"
        ),
    )

filtered["is_pareto"] = False
if has_final_hp:
    pareto_candidates = filtered.dropna(subset=["final_hp", "max_rounds"])
    if not pareto_candidates.empty:
        pareto_mask = identify_pareto(pareto_candidates)
        filtered.loc[pareto_mask.index, "is_pareto"] = pareto_mask

summary_agg = {
    "number": "count",
    "max_rounds": "max",
}
if has_final_hp:
    summary_agg["final_hp"] = final_hp_closest_to_one

summary = (
    filtered.groupby("algorithm", as_index=False)
    .agg(summary_agg)
    .rename(
        columns={
            "number": "trials",
            "max_rounds": "best_max_rounds",
            "final_hp": "final_hp_closest_to_one",
        }
    )
)

st.subheader("Algorithm Snapshot")
display_summary = summary.rename(
    columns={"best_max_rounds": f"best_{primary_metric_label}"}
)
summary_formatters = {
    f"best_{primary_metric_label}": "{:.2f}",
}
if has_final_hp and "final_hp_closest_to_one" in summary.columns:
    summary_formatters["final_hp_closest_to_one"] = "{:.2f}"

st.dataframe(display_summary.style.format(summary_formatters), hide_index=True)

st.subheader("Performance Overview")

sorted_trials = filtered.sort_values("number")
param_hover_cols = [col for col in PARAM_COLUMNS if col in sorted_trials.columns]

if has_final_hp:
    pareto_fig = px.scatter(
        sorted_trials,
        x="max_rounds",
        y="final_hp",
        color="algorithm",
        color_discrete_map=ALGORITHM_COLORS,
        hover_data=["number", "duration_seconds", metric_hover_key]
        + param_hover_cols,
        title="Pareto Front: Max Rounds vs Final HP",
    )
    min_final_hp = sorted_trials["final_hp"].min(skipna=True)
    y_lower_bound = min(min_final_hp if pd.notna(min_final_hp) else 0, 1)
    pareto_fig.add_hline(y=1, line_dash="dash", line_color="#666666")
    pareto_fig.add_hrect(
        y0=y_lower_bound,
        y1=1,
        x0=0,
        x1=1,
        xref="paper",
        fillcolor="rgba(128,128,128,0.15)",
        layer="below",
        line_width=0,
    )
    for trace in pareto_fig.data:
        trace_mask = sorted_trials["algorithm"] == trace.name
        pareto_flags = sorted_trials.loc[trace_mask, "is_pareto"].tolist()
        line_colors = ["#000000" if flag else "rgba(0,0,0,0)" for flag in pareto_flags]
        line_widths = [1 if flag else 0 for flag in pareto_flags]
        trace.update(
            marker={
                "line": {
                    "color": line_colors,
                    "width": line_widths,
                }
            }
        )
    pareto_fig.update_layout(legend_title_text="Algorithm")

    # Create scatter plot with error bars for max_rounds vs final_hp
    has_std_data = (
        "max_rounds_std" in sorted_trials.columns
        and "final_hp_std" in sorted_trials.columns
    )

    if has_std_data:
        # Cap error bars so they don't go below 0
        scatter_data = sorted_trials.copy()
        scatter_data["max_rounds_error_lower"] = scatter_data.apply(
            lambda row: (
                min(row["max_rounds_std"], row["max_rounds"])
                if pd.notna(row["max_rounds_std"]) and pd.notna(row["max_rounds"])
                else 0
            ),
            axis=1,
        )
        scatter_data["max_rounds_error_upper"] = scatter_data["max_rounds_std"].fillna(
            0
        )
        scatter_data["final_hp_error_lower"] = scatter_data.apply(
            lambda row: (
                min(row["final_hp_std"], row["final_hp"])
                if pd.notna(row["final_hp_std"]) and pd.notna(row["final_hp"])
                else 0
            ),
            axis=1,
        )
        scatter_data["final_hp_error_upper"] = scatter_data["final_hp_std"].fillna(0)

        rounds_scatter_fig = px.scatter(
            scatter_data,
            x="max_rounds",
            y="final_hp",
            color="algorithm",
            color_discrete_map=ALGORITHM_COLORS,
            error_x="max_rounds_error_upper",
            error_x_minus="max_rounds_error_lower",
            error_y="final_hp_error_upper",
            error_y_minus="final_hp_error_lower",
            hover_data=["number", "duration_seconds", metric_hover_key]
            + param_hover_cols,
            title="Max Rounds vs Final HP (with std)",
        )
        rounds_scatter_fig.update_traces(
            error_x=dict(thickness=1.5, color="rgba(0,0,0,0.2)"),
            error_y=dict(thickness=1.5, color="rgba(0,0,0,0.2)"),
        )
        rounds_scatter_fig.update_layout(legend_title_text="Algorithm")
    else:
        # Fallback to simple scatter if std data not available
        rounds_scatter_fig = px.scatter(
            sorted_trials,
            x="max_rounds",
            y="final_hp",
            color="algorithm",
            color_discrete_map=ALGORITHM_COLORS,
            hover_data=["number", "duration_seconds", metric_hover_key]
            + param_hover_cols,
            title="Max Rounds vs Final HP",
        )
        rounds_scatter_fig.update_layout(legend_title_text="Algorithm")

    hp_over_trials_fig = px.line(
        sorted_trials,
        x="number",
        y="final_hp",
        color="algorithm",
        color_discrete_map=ALGORITHM_COLORS,
        markers=True,
        hover_data=["duration_seconds", metric_hover_key] + param_hover_cols,
        title="Final HP vs Trial Number",
    )
    hp_over_trials_fig.update_layout(
        xaxis_title="Trial Number", legend_title_text="Algorithm"
    )

    overview_columns = st.columns(2)
    overview_columns[0].plotly_chart(pareto_fig, use_container_width=True)
    overview_columns[1].plotly_chart(rounds_scatter_fig, use_container_width=True)
    st.plotly_chart(hp_over_trials_fig, use_container_width=True)
else:
    rounds_over_trials_fig = px.line(
        sorted_trials,
        x="number",
        y="max_rounds",
        color="algorithm",
        color_discrete_map=ALGORITHM_COLORS,
        markers=True,
        hover_data=["duration_seconds", metric_hover_key] + param_hover_cols,
        title=f"{primary_metric_label} vs Trial Number",
    )
    rounds_over_trials_fig.update_layout(
        xaxis_title="Trial Number",
        yaxis_title=primary_metric_label,
        legend_title_text="Algorithm",
    )
    st.plotly_chart(rounds_over_trials_fig, use_container_width=True)

# Milestone-specific 3D view
if study_key == "milestone":
    st.subheader("3D View: Killed Enemies vs Range & Damage")
    required_cols = {
        "params_playerShotRange",
        "params_playerShotDamage",
        "max_rounds",
    }
    if required_cols.issubset(filtered.columns):
        milestone_hover = [
            "number",
            "duration_seconds",
            metric_hover_key,
            "params_playerShotRange",
            "params_playerShotDamage",
        ]
        milestone_3d_fig = px.scatter_3d(
            filtered,
            x="params_playerShotRange",
            y="params_playerShotDamage",
            z="max_rounds",
            color="algorithm",
            color_discrete_map=ALGORITHM_COLORS,
            hover_data=milestone_hover,
            opacity=0.65,
            title="Killed Enemies vs Shot Range and Damage",
        )
        milestone_3d_fig.update_traces(marker=dict(size=3, opacity=0.6))
        milestone_3d_fig.update_layout(
            legend_title_text="Algorithm",
            height=720,
            scene=dict(
                xaxis_title="Player Shot Range",
                yaxis_title="Player Shot Damage",
                zaxis_title=primary_metric_label,
            ),
        )
        st.plotly_chart(milestone_3d_fig, use_container_width=True)
    else:
        st.info(
            "3D plot unavailable: missing playerShotRange or playerShotDamage columns in milestone data."
        )

st.subheader("Parameter Effects")

metric_options = {
    primary_metric_label: "max_rounds",
    "Duration (s)": "duration_seconds",
}
if has_final_hp:
    metric_options["Final HP"] = "final_hp"

metric_label = st.selectbox("Metric", list(metric_options.keys()))
target_metric = metric_options[metric_label]

selected_params = st.multiselect(
    "Parameters",
    options=[col for col in PARAM_COLUMNS if col in filtered.columns],
    default=[col for col in PARAM_COLUMNS if col in filtered.columns],
)

if selected_params:
    for row in chunk_list(selected_params, 2):
        chart_columns = st.columns(len(row))
        for index, param_col in enumerate(row):
            plot_df = filtered.copy()
            plot_df["param_jittered"] = add_jitter(plot_df[param_col])

            param_fig = px.scatter(
                plot_df,
                x="param_jittered",
                y=target_metric,
                color="algorithm",
                color_discrete_map=ALGORITHM_COLORS,
                hover_data=["number", "duration_seconds", metric_hover_key, param_col]
                + (["final_hp"] if has_final_hp else [])
                + [
                    col
                    for col in PARAM_COLUMNS
                    if col in plot_df.columns and col != param_col
                ],
                title=f"{metric_label} vs {param_col}",
            )
            param_fig.update_layout(
                legend_title_text="Algorithm",
                xaxis_title=param_col,
                yaxis_title=metric_label,
            )
            chart_columns[index].plotly_chart(param_fig, use_container_width=True)
else:
    st.info("Select at least one parameter to see how it relates to the chosen metric.")

st.subheader("Raw Trials")

display_columns = [
    "number",
    "algorithm",
    "max_rounds",
    "final_hp",
    "duration_seconds",
    "duration",
    "datetime_start",
    "datetime_complete",
] + [col for col in PARAM_COLUMNS if col in filtered.columns]

display_columns = [col for col in display_columns if col in filtered.columns]
display_frame = filtered[display_columns].copy()
if study_key == "milestone" and "max_rounds" in display_frame.columns:
    display_frame["Killed enemies"] = display_frame["max_rounds"]
    display_frame = display_frame.drop(columns=["max_rounds"])
if "duration_seconds" in display_frame.columns:
    display_frame["duration_seconds"] = display_frame["duration_seconds"].map(
        format_duration
    )

st.dataframe(display_frame, use_container_width=True)

st.caption("Data source: Optuna study exports located in the optuna/ directory.")
