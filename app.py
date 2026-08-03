import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google import genai
from google.genai import types
import os
import urllib.parse
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="Life-OS Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 10px; }
    [data-testid="stSidebar"] { background-color: #000000; color: white;}
    [data-testid="stSidebar"] h1 { color: #ffffff !important; background-color: #000000; }
    .science-box {
        background-color: #fff8e1;
        border-left: 5px solid #f0a500;
        padding: 14px 18px;
        border-radius: 6px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("screentime.csv")
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

try:
    data = load_data()
except FileNotFoundError:
    st.error("Error: screentime.csv not found. Please create the dataset.")
    st.stop()

st.sidebar.title("⚡ Life-OS Settings")
st.sidebar.markdown("---")

available_dates = sorted(data['Date'].unique(), reverse=True)
selected_date = st.sidebar.selectbox("📅 Select Day to Analyze", available_dates)
daily_goal = st.sidebar.slider("🎯 Daily Screen Time Limit (mins)", min_value=60, max_value=600, value=240, step=30)

day_data = data[data['Date'] == selected_date]

st.markdown(":rainbow[# 🧠 Life-OS: Digital Wellbeing Coach]")
st.markdown(f"### Analyzing data for: {selected_date}")

total_minutes = day_data['Minutes_Used'].sum()
top_app = day_data.loc[day_data['Minutes_Used'].idxmax()]['App_Name'] if not day_data.empty else "None"
goal_delta = daily_goal - total_minutes

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Screen Time", f"{total_minutes} mins", delta=f"{goal_delta} mins to goal", delta_color="normal")
with col2:
    st.metric("Daily Limit", f"{daily_goal} mins")
with col3:
    st.metric("Most Used App", top_app)

st.markdown("---")

st.markdown(":rainbow[# 📊 Trends & Breakdown]")

trend_data = data.groupby('Date')['Minutes_Used'].sum().reset_index()
trend_data['Over_Goal'] = trend_data['Minutes_Used'] > daily_goal
trend_data['Status'] = trend_data['Over_Goal'].map({True: "Over Goal", False: "Under Goal"})

chart_col1, chart_col2 = st.columns([2, 1])

with chart_col1:
    fig_trend = px.bar(
        trend_data, x='Date', y='Minutes_Used', color='Status',
        color_discrete_map={"Over Goal": "#e74c3c", "Under Goal": "#2ecc71"},
        title="14-Day Screen Time vs Goal",
        labels={"Minutes_Used": "Minutes Used"}
    )
    fig_trend.add_hline(
        y=daily_goal, line_dash="dash", line_color="#f0a500",
        annotation_text=f"Goal: {daily_goal} min", annotation_position="top left"
    )
    fig_trend.update_layout(height=420, legend_title_text="")
    st.plotly_chart(fig_trend, use_container_width=True)

with chart_col2:
    cat_data = day_data.groupby('Category')['Minutes_Used'].sum().reset_index()
    fig_donut = px.pie(
        cat_data, names='Category', values='Minutes_Used', hole=0.55,
        title=f"Category Split — {selected_date}"
    )
    fig_donut.update_traces(textinfo='percent+label')
    fig_donut.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig_donut, use_container_width=True)

app_data = day_data.groupby('App_Name')['Minutes_Used'].sum().reset_index().sort_values('Minutes_Used')
fig_apps = px.bar(
    app_data, x='Minutes_Used', y='App_Name', orientation='h',
    title=f"App Usage Ranking — {selected_date}",
    color='Minutes_Used', color_continuous_scale='sunset'
)
fig_apps.update_layout(height=max(300, 40 * len(app_data)), coloraxis_showscale=False)
st.plotly_chart(fig_apps, use_container_width=True)

st.markdown("---")

st.markdown(":rainbow[# 🤖 AI Analysis & The Guilt-Trip Avatar]")

if not API_KEY:
    st.warning("⚠️ Please add your GEMINI_API_KEY to your .env file to enable AI coaching.")
else:
    if st.button("Generate Today's Actionable Intel", type="primary"):
        with st.spinner("Gemini is analyzing your behavior..."):

            category_summary = day_data.groupby('Category')['Minutes_Used'].sum().to_dict()
            app_summary = day_data.groupby('App_Name')['Minutes_Used'].sum().to_dict()

            summary_string = f"""
            Total Time: {total_minutes} minutes (Goal: {daily_goal} minutes)
            Category Breakdown: {category_summary}
            App Breakdown: {app_summary}
            """

            prompt = f"""
            You are a brutal-but-fair, personalized lifestyle and productivity coach who is
            also fluent in behavioral science and public health research on screen time.

            Review the user's screen time data below.

            Data:
            {summary_string}

            Rules:
            1. Do NOT just say "use your phone less".
            2. If they spent excessive time on Entertainment/Social Media, be blunt. Suggest specific, physical real-world replacements (e.g., meal prep, lifting weights, reading physical books).
            3. If they spent a lot of time on Coding/Productivity, praise their focus but remind them to stretch and protect their posture.
            4. For the science section: name the 1-2 categories they overused most, and explain what
               generally-accepted research says about that specific pattern (e.g., late-evening screen
               use and blue light's effect on melatonin/sleep onset; short-form video and dopamine/attention-span
               research; prolonged sitting and musculoskeletal/eye-strain risk). Keep claims general and
               well-established (no invented statistics or studies), and explicitly note this is general
               research context, not medical advice.

            OUTPUT FORMAT:
            You must output your response in exactly three parts separated by the characters "|||".
            Part 1: Your brutal-but-fair coaching advice (formatted nicely in markdown).
            Part 2: A single-sentence visual prompt for an AI image generator representing their day.
            Example for a bad day: "A lazy zombie slouching on a couch staring into a glowing smartphone in a dark messy room, digital art style."
            Example for a good day: "A focused cyber-monk meditating on a mountaintop surrounded by glowing code holograms, futuristic art style."
            Part 3: The scientific warning section described in Rule 4, formatted in markdown, 3-5 sentences max.
            """

            try:
                client = genai.Client(api_key=API_KEY)
                response = client.models.generate_content(
                    model='gemini-3.1-flash-lite',
                    contents=prompt
                )

                output_parts = response.text.split("|||")
                coaching_text = output_parts[0].strip()
                image_prompt = output_parts[1].strip() if len(output_parts) > 1 else "A confused robot."
                science_text = output_parts[2].strip() if len(output_parts) > 2 else (
                    "No specific research note was generated this time — try again."
                )

                report_col, avatar_col = st.columns([2, 1])

                with report_col:
                    if total_minutes > daily_goal:
                        st.warning(coaching_text)
                    else:
                        st.info(coaching_text)

                with avatar_col:
                    st.markdown("**Your Daily Avatar:**")
                    encoded_prompt = urllib.parse.quote(image_prompt)
                    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=400&height=400&nologo=true"
                    st.image(image_url, caption=f"Prompt: {image_prompt}")

                st.markdown("## 🔬 Science Says")
                st.markdown(f'<div class="science-box">{science_text}</div>', unsafe_allow_html=True)
                st.caption("General research context, not a medical diagnosis or advice.")

            except Exception as e:
                st.error(f"Failed to generate analysis. Error: {e}")
