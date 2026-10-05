import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine, text
from datetime import datetime
import hashlib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Digital Twin",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown("""
<style>

/* ---------- MAIN APP ---------- */

.stApp {
    background: #07111F;
    color: #F4F7FB;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ---------- SIDEBAR ---------- */

section[data-testid="stSidebar"] {
    background: #0B1728;
    border-right: 1px solid #1B3048;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #F4F7FB;
}


/* ---------- TEXT ---------- */

h1, h2, h3 {
    color: #F4F7FB !important;
}

p, label {
    color: #A9B8CA !important;
}


/* ---------- HERO ---------- */

.hero-box {
    background: linear-gradient(
        135deg,
        #102A43 0%,
        #0B1F33 55%,
        #09253A 100%
    );

    border: 1px solid #1C405B;
    border-radius: 24px;

    padding: 38px 42px;

    margin-bottom: 30px;

    box-shadow:
        0 20px 60px rgba(0,0,0,0.28);
}

.hero-title {
    color: #FFFFFF;
    font-size: 46px;
    font-weight: 800;
    margin: 0;
}

.hero-subtitle {
    color: #9FB3C8;
    font-size: 17px;
    margin-top: 10px;
    line-height: 1.6;
}

.online {
    display: inline-block;

    margin-top: 20px;

    padding: 6px 13px;

    border-radius: 20px;

    background: #0E3B3A;

    color: #58E0B2;

    font-size: 12px;

    font-weight: 700;

    letter-spacing: 1px;
}


/* ---------- SECTION ---------- */

.section {
    color: #F4F7FB;

    font-size: 24px;

    font-weight: 750;

    margin-top: 32px;

    margin-bottom: 8px;
}

.section-info {
    color: #8093A8;

    font-size: 14px;

    margin-bottom: 18px;
}


/* ---------- METRIC BOX ---------- */

.metric-box {
    background: #0D1B2D;

    border: 1px solid #1B3048;

    border-radius: 18px;

    padding: 22px;

    min-height: 135px;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.18);
}

.metric-label {
    color: #8093A8;

    font-size: 12px;

    text-transform: uppercase;

    letter-spacing: 1px;
}

.metric-number {
    color: #FFFFFF;

    font-size: 29px;

    font-weight: 800;

    margin-top: 10px;
}

.metric-note {
    color: #61758A;

    font-size: 12px;

    margin-top: 7px;
}


/* ---------- AI PANEL ---------- */

.ai-panel {
    background: #0D1B2D;

    border: 1px solid #274A66;

    border-left: 5px solid #36C5F0;

    border-radius: 18px;

    padding: 24px;

    margin-top: 22px;

    margin-bottom: 25px;
}

.ai-heading {
    color: #36C5F0;

    font-size: 12px;

    font-weight: 800;

    letter-spacing: 1.5px;

    text-transform: uppercase;
}

.ai-state {
    color: #FFFFFF;

    font-size: 25px;

    font-weight: 750;

    margin-top: 7px;
}

.ai-message {
    color: #A9B8CA;

    font-size: 15px;

    line-height: 1.6;

    margin-top: 7px;
}


/* ---------- ID PANEL ---------- */

.id-panel {
    background: #0B1728;

    border: 1px solid #1B3048;

    border-radius: 18px;

    padding: 24px;

    text-align: center;

    margin-top: 30px;
}

.id-label {
    color: #657B91;

    font-size: 11px;

    letter-spacing: 2px;
}

.id-value {
    color: #74D7F5;

    font-size: 20px;

    font-weight: 800;

    letter-spacing: 3px;

    margin-top: 8px;
}


/* ---------- BUTTON ---------- */

.stButton > button {
    background: #147DA8;

    color: white;

    border: 1px solid #2BA9D4;

    border-radius: 10px;

    font-weight: 700;

    min-height: 42px;
}

.stButton > button:hover {
    background: #1894C5;

    border-color: #5FD9FF;

    color: white;
}


/* ---------- SLIDERS ---------- */

.stSlider [data-baseweb="slider"] {
    margin-top: 5px;
}


/* ---------- TABS ---------- */

button[data-baseweb="tab"] {
    color: #91A5B9 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #49C8F0 !important;
}


/* ---------- DATAFRAME ---------- */

[data-testid="stDataFrame"] {
    border: 1px solid #1B3048;
    border-radius: 12px;
}


/* ---------- DIVIDER ---------- */

hr {
    border-color: #1B3048 !important;
}


/* ---------- ALERTS ---------- */

div[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATABASE
# ============================================================

engine = create_engine(
    "sqlite:///digital_twin.db",
    echo=False
)

with engine.begin() as connection:
    connection.execute(
        text("""
        CREATE TABLE IF NOT EXISTS twin_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            name TEXT,
            age INTEGER,
            sleep REAL,
            activity REAL,
            stress REAL,
            productivity REAL,
            predicted_state TEXT
        )
        """)
    )


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero-box">

    <div class="hero-title">
        🧬 AI Digital Twin
    </div>

    <div class="hero-subtitle">
        An intelligent digital representation that analyzes
        behavioral signals, learns patterns and simulates
        possible future states.
    </div>

    <div class="online">
        ● TWIN ENGINE ONLINE
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧬 Twin Setup")

st.sidebar.caption(
    "Configure the current state of your digital twin."
)

user_name = st.sidebar.text_input(
    "Name",
    value="Alex"
)

user_age = st.sidebar.number_input(
    "Age",
    min_value=10,
    max_value=100,
    value=21,
    step=1
)

st.sidebar.divider()

st.sidebar.subheader("📡 Behavioral Signals")

sleep_hours = st.sidebar.slider(
    "Sleep Hours",
    min_value=0.0,
    max_value=12.0,
    value=7.0,
    step=0.5
)

activity_level = st.sidebar.slider(
    "Activity Level",
    min_value=0,
    max_value=100,
    value=65
)

stress_level = st.sidebar.slider(
    "Stress Level",
    min_value=0,
    max_value=100,
    value=35
)

productivity_level = st.sidebar.slider(
    "Productivity",
    min_value=0,
    max_value=100,
    value=75
)

st.sidebar.divider()

sync_button = st.sidebar.button(
    "🔄 Sync Digital Twin",
    use_container_width=True
)


# ============================================================
# SCORE FUNCTION
# ============================================================

def calculate_score(
    sleep,
    activity,
    stress,
    productivity
):

    sleep_score = min(
        (sleep / 8) * 100,
        100
    )

    score = (
        sleep_score * 0.25
        + activity * 0.25
        + productivity * 0.30
        + (100 - stress) * 0.20
    )

    return round(score, 2)


# ============================================================
# STATE FUNCTION
# ============================================================

def get_state(score):

    if score >= 75:
        return "Optimal"

    if score >= 55:
        return "Stable"

    if score >= 35:
        return "Needs Attention"

    return "Critical"


# ============================================================
# CREATE SYNTHETIC TRAINING DATA
# ============================================================

np.random.seed(42)

rows = []

for _ in range(1200):

    random_sleep = np.random.uniform(3, 10)

    random_activity = np.random.uniform(0, 100)

    random_stress = np.random.uniform(0, 100)

    random_productivity = np.random.uniform(0, 100)

    random_score = calculate_score(
        random_sleep,
        random_activity,
        random_stress,
        random_productivity
    )

    random_state = get_state(
        random_score
    )

    rows.append(
        {
            "sleep": random_sleep,
            "activity": random_activity,
            "stress": random_stress,
            "productivity": random_productivity,
            "state": random_state
        }
    )


training_data = pd.DataFrame(rows)


# ============================================================
# TRAIN MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    training_data[
        [
            "sleep",
            "activity",
            "stress",
            "productivity"
        ]
    ],
    training_data["state"]
)


# ============================================================
# CURRENT INPUT
# ============================================================

current_data = pd.DataFrame(
    [{
        "sleep": sleep_hours,
        "activity": activity_level,
        "stress": stress_level,
        "productivity": productivity_level
    }]
)


# ============================================================
# PREDICTION
# ============================================================

predicted_state = model.predict(
    current_data
)[0]

prediction_probability = model.predict_proba(
    current_data
)[0]

confidence = round(
    max(prediction_probability) * 100,
    1
)

current_score = calculate_score(
    sleep_hours,
    activity_level,
    stress_level,
    productivity_level
)


# ============================================================
# SYNC
# ============================================================

if sync_button:

    with engine.begin() as connection:

        connection.execute(
            text("""
            INSERT INTO twin_history
            (
                timestamp,
                name,
                age,
                sleep,
                activity,
                stress,
                productivity,
                predicted_state
            )
            VALUES
            (
                :timestamp,
                :name,
                :age,
                :sleep,
                :activity,
                :stress,
                :productivity,
                :state
            )
            """),
            {
                "timestamp":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "name":
                    user_name,

                "age":
                    int(user_age),

                "sleep":
                    sleep_hours,

                "activity":
                    activity_level,

                "stress":
                    stress_level,

                "productivity":
                    productivity_level,

                "state":
                    predicted_state
            }
        )

    st.sidebar.success(
        "Twin synchronized successfully."
    )


# ============================================================
# TWIN INTELLIGENCE
# ============================================================

st.markdown(
    '<div class="section">🧠 Twin Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-info">'
    'Current interpretation generated by the machine-learning model.'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="metric-box">

            <div class="metric-label">
                Twin Score
            </div>

            <div class="metric-number">
                {current_score}
            </div>

            <div class="metric-note">
                Overall simulated score
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="metric-box">

            <div class="metric-label">
                Current State
            </div>

            <div class="metric-number">
                {predicted_state}
            </div>

            <div class="metric-note">
                AI classification
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="metric-box">

            <div class="metric-label">
                Confidence
            </div>

            <div class="metric-number">
                {confidence}%
            </div>

            <div class="metric-note">
                Model confidence
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="metric-box">

            <div class="metric-label">
                Profile Age
            </div>

            <div class="metric-number">
                {int(user_age)} yrs
            </div>

            <div class="metric-note">
                Digital profile
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# AI ANALYSIS
# ============================================================

if predicted_state == "Optimal":

    ai_message = (
        "The current simulated signals indicate a balanced "
        "and strong digital-twin state."
    )

elif predicted_state == "Stable":

    ai_message = (
        "The twin is currently stable. Changes in the "
        "input signals may influence its future state."
    )

elif predicted_state == "Needs Attention":

    ai_message = (
        "The twin has identified an imbalance in the "
        "current simulated behavioral signals."
    )

else:

    ai_message = (
        "The current simulated signals indicate a significant "
        "imbalance in the twin state."
    )


st.markdown(
    f"""
    <div class="ai-panel">

        <div class="ai-heading">
            AI STATE ANALYSIS
        </div>

        <div class="ai-state">
            {predicted_state}
        </div>

        <div class="ai-message">
            {ai_message}
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIGNAL DASHBOARD
# ============================================================

st.markdown(
    '<div class="section">📡 Twin Signal Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-info">'
    'A normalized view of the signals currently influencing the twin.'
    '</div>',
    unsafe_allow_html=True
)


sleep_normalized = min(
    (sleep_hours / 8) * 100,
    100
)


signal_data = pd.DataFrame(
    {
        "Signal": [
            "Sleep",
            "Activity",
            "Stress",
            "Productivity"
        ],

        "Value": [
            sleep_normalized,
            activity_level,
            stress_level,
            productivity_level
        ]
    }
)


chart_col1, chart_col2 = st.columns(2)


# ============================================================
# SIGNAL BAR CHART
# ============================================================

with chart_col1:

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.bar(
        signal_data["Signal"],
        signal_data["Value"]
    )

    ax.set_ylim(
        0,
        100
    )

    ax.set_ylabel(
        "Signal Level"
    )

    ax.set_title(
        "Current Twin Signals"
    )

    ax.grid(
        axis="y",
        alpha=0.15
    )

    fig.patch.set_alpha(0)

    ax.set_facecolor(
        "#07111F"
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    plt.close(fig)


# ============================================================
# SIGNAL TABLE
# ============================================================

with chart_col2:

    display_data = signal_data.copy()

    display_data["Value"] = (
        display_data["Value"].round(1)
    )

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Signal values are normalized to a 0–100 simulation scale."
    )


# ============================================================
# FUTURE SIMULATION
# ============================================================

st.markdown(
    '<div class="section">🔮 Future Twin Simulation</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-info">'
    'Change hypothetical inputs and observe how the digital twin responds.'
    '</div>',
    unsafe_allow_html=True
)


sim_col1, sim_col2 = st.columns(2)


with sim_col1:

    future_sleep = st.slider(
        "Future Sleep",
        0.0,
        12.0,
        float(sleep_hours),
        0.5,
        key="future_sleep"
    )

    future_activity = st.slider(
        "Future Activity",
        0,
        100,
        int(activity_level),
        key="future_activity"
    )


with sim_col2:

    future_stress = st.slider(
        "Future Stress",
        0,
        100,
        int(stress_level),
        key="future_stress"
    )

    future_productivity = st.slider(
        "Future Productivity",
        0,
        100,
        int(productivity_level),
        key="future_productivity"
    )


future_data = pd.DataFrame(
    [{
        "sleep": future_sleep,
        "activity": future_activity,
        "stress": future_stress,
        "productivity": future_productivity
    }]
)


future_state = model.predict(
    future_data
)[0]


future_score = calculate_score(
    future_sleep,
    future_activity,
    future_stress,
    future_productivity
)


score_change = round(
    future_score - current_score,
    2
)


st.write("")


future_col1, future_col2, future_col3 = st.columns(3)


with future_col1:

    st.metric(
        "Current Score",
        current_score
    )


with future_col2:

    st.metric(
        "Simulated Score",
        future_score,
        delta=score_change
    )


with future_col3:

    st.metric(
        "Predicted Future State",
        future_state
    )


if score_change > 0:

    st.success(
        f"Simulation indicates an improvement of "
        f"{score_change} points."
    )

elif score_change < 0:

    st.warning(
        f"Simulation indicates a decrease of "
        f"{abs(score_change)} points."
    )

else:

    st.info(
        "The simulated conditions produce the same score."
    )


# ============================================================
# TWIN EVOLUTION
# ============================================================

st.markdown(
    '<div class="section">📈 Twin Evolution</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-info">'
    'Every synchronized snapshot becomes part of the twin history.'
    '</div>',
    unsafe_allow_html=True
)


with engine.connect() as connection:

    history = pd.read_sql(
        text("""
        SELECT
            timestamp,
            name,
            age,
            sleep,
            activity,
            stress,
            productivity,
            predicted_state
        FROM twin_history
        ORDER BY id ASC
        """),
        connection
    )


if len(history) > 0:

    history["timestamp"] = pd.to_datetime(
        history["timestamp"]
    )

    history["score"] = history.apply(
        lambda row: calculate_score(
            row["sleep"],
            row["activity"],
            row["stress"],
            row["productivity"]
        ),
        axis=1
    )

    st.line_chart(
        history.set_index(
            "timestamp"
        )["score"]
    )

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No historical snapshots yet. "
        "Click 'Sync Digital Twin' to create the first snapshot."
    )


# ============================================================
# DIGITAL TWIN ID
# ============================================================

digital_twin_id = hashlib.sha256(
    f"{user_name}-{user_age}".encode()
).hexdigest()[:12].upper()


st.markdown(
    f"""
    <div class="id-panel">

        <div class="id-label">
            DIGITAL TWIN IDENTIFIER
        </div>

        <div class="id-value">
            {digital_twin_id}
        </div>

        <div style="
            color:#657B91;
            font-size:12px;
            margin-top:8px;
        ">
            Persistent identity for this simulation
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Digital Twin • Machine Learning • Behavioral Simulation"
)
