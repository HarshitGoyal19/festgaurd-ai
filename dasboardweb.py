import streamlit as st
import cv2
import numpy as np
import pandas as pd
import requests
import threading
import time
from datetime import datetime
from ultralytics import YOLO

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="FestGuard AI | Campus Crowd Intelligence",
    page_icon="🛡️",
    layout="wide"
)

# --- TELEGRAM KEYS ---
BOT_TOKEN = st.secrets["BOT_TOKEN"]
CHAT_ID = st.secrets["CHAT_ID"]
ALERT_COOLDOWN = 25

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []
if "alert_logs" not in st.session_state:
    st.session_state.alert_logs = []
if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = 0

def send_telegram_alert(zone_count, max_cap, snapshot_frame):
    try:
        cv2.imwrite("alert_snapshot.jpg", snapshot_frame)
        caption = (
            f"🚨 CRITICAL CROWD ALERT!\n"
            f"📍 Zone: Main Stage Barricade\n"
            f"👥 Headcount: {zone_count} / {max_cap} (Overcrowded)\n"
            f"⚡ Action: Dispatch security & activate Student Redirect."
        )
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        with open("alert_snapshot.jpg", "rb") as photo:
            requests.post(
                url,
                data={"chat_id": CHAT_ID, "caption": caption},
                files={"photo": photo},
                timeout=10
            )
    except Exception as e:
        print(f"Telegram alert error: {e}")

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🎛️ FestGuard Controls")
view_mode = st.sidebar.radio(
    "Select Portal View",
    ["🛡️ Organizer Command Center", "🎒 Student Live Crowd Guide"]
)

st.sidebar.divider()
source_choice = st.sidebar.radio(
    "Camera Feed Source",
    ["Sample Video (crowd.mp4)", "Live Webcam (0)", "Phone IP Webcam (Wi-Fi)"]
)

if source_choice == "Sample Video (crowd.mp4)":
    video_source = "crowd.mp4"
elif source_choice == "Live Webcam (0)":
    video_source = 0
else:
    ip_url = st.sidebar.text_input("Phone Stream URL", value="http://172.20.10.2:8080/video")
    video_source = ip_url.strip()

warning_cap = st.sidebar.slider("Warning Threshold (Yellow)", 2, 30, 8)
max_cap = st.sidebar.slider("Critical Threshold (Red)", 5, 50, 14)
show_heatmap = st.sidebar.checkbox("Enable Thermal Heatmap", value=True)
enable_telegram = st.sidebar.checkbox("Enable Telegram Alerts", value=True)
run_monitor = st.sidebar.toggle("🟢 Start Live AI Engine", value=True)


# VIEW 1: ORGANIZER SECURITY COMMAND CENTER

if view_mode == "🛡️ Organizer Command Center":
    st.title("🛡️ FestGuard AI — Organizer Command Center")
    st.caption("Real-time YOLOv8 spatial density analytics, thermal bottleneck detection & automated security dispatch")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi_zone = kpi1.empty()
    kpi_pct = kpi2.empty()
    kpi_total = kpi3.empty()
    kpi_status = kpi4.empty()

    st.divider()

    col_video, col_stats = st.columns([1.45, 1])
    with col_video:
        st.subheader("📹 Zone 01: Main Stage Barricade (Live Feed)")
        video_placeholder = st.empty()

    with col_stats:
        st.subheader("📈 Live Crowd Density Trend")
        chart_placeholder = st.empty()

        st.subheader("🏫 Campus-Wide Zone Status")
        campus_placeholder = st.empty()

    st.subheader("🚨 Automated Incident & Alert Log")
    log_placeholder = st.empty()

    if run_monitor:
        cap = cv2.VideoCapture(video_source)
        heatmap = np.zeros((500, 800), dtype=np.float32)
        zone_polygon = np.array([[130, 160], [670, 160], [740, 470], [60, 470]], np.int32)
        frame_idx = 0

        while cap.isOpened() and run_monitor:
            ret, frame = cap.read()
            if not ret:
                if source_choice == "Sample Video (crowd.mp4)":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    heatmap[:] = 0
                    continue
                else:
                    st.error("⚠️ Could not read from camera stream. Check your URL/hotspot or switch back to Sample Video in the sidebar.")
                    break

            frame_idx += 1
            frame = cv2.resize(frame, (800, 500))
            heatmap *= 0.90

            results = model(frame, classes=[0], conf=0.35, verbose=False)

            zone_count = 0
            total_count = 0

            for box in results[0].boxes:
                total_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                feet_x, feet_y = (x1 + x2) // 2, y2

                inside_zone = cv2.pointPolygonTest(zone_polygon, (feet_x, feet_y), False) >= 0

                if inside_zone:
                    zone_count += 1
                    cv2.circle(heatmap, (feet_x, feet_y), 38, 0.65, -1)
                    c = (0, 215, 255)
                    l = 12
                    cv2.line(frame, (x1, y1), (x1 + l, y1), c, 2)
                    cv2.line(frame, (x1, y1), (x1, y1 + l), c, 2)
                    cv2.line(frame, (x2, y1), (x2 - l, y1), c, 2)
                    cv2.line(frame, (x2, y1), (x2, y1 + l), c, 2)
                    cv2.line(frame, (x1, y2), (x1 + l, y2), c, 2)
                    cv2.line(frame, (x1, y2), (x1, y2 - l), c, 2)
                    cv2.line(frame, (x2, y2), (x2 - l, y2), c, 2)
                    cv2.line(frame, (x2, y2), (x2, y2 - l), c, 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

            if show_heatmap:
                heatmap_blur = cv2.GaussianBlur(heatmap, (65, 65), 0)
                heatmap_norm = np.clip((heatmap_blur / 1.8) * 255, 0, 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
                mask = heatmap_norm > 15
                blended = cv2.addWeighted(frame, 0.50, heatmap_color, 0.50, 0)
                frame[mask] = blended[mask]

            if zone_count >= max_cap:
                status_label = "🔴 CRITICAL"
                status_color = (0, 0, 255)
            elif zone_count >= warning_cap:
                status_label = "🟡 WARNING"
                status_color = (0, 215, 255)
            else:
                status_label = "🟢 SAFE"
                status_color = (0, 255, 100)

            zone_overlay = frame.copy()
            cv2.fillPoly(zone_overlay, [zone_polygon], status_color)
            frame = cv2.addWeighted(zone_overlay, 0.12, frame, 0.88, 0)
            cv2.polylines(frame, [zone_polygon], isClosed=True, color=status_color, thickness=2)

            now = time.time()
            if zone_count >= max_cap and (now - st.session_state.last_alert_time > ALERT_COOLDOWN):
                st.session_state.last_alert_time = now
                timestamp_str = datetime.now().strftime("%H:%M:%S")
                st.session_state.alert_logs.insert(0, {
                    "Time": timestamp_str,
                    "Zone": "Zone 01 - Main Stage",
                    "Headcount": f"{zone_count} / {max_cap}",
                    "Severity": "CRITICAL",
                    "Action Taken": "Telegram Photo Alert Dispatched"
                })
                if enable_telegram and BOT_TOKEN != "YOUR_BOT_TOKEN_HERE":
                    threading.Thread(
                        target=send_telegram_alert,
                        args=(zone_count, max_cap, frame.copy()),
                        daemon=True
                    ).start()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Save live count to session state so Student View can read it too!
            st.session_state.live_stage_count = zone_count
            st.session_state.live_stage_max = max_cap

            capacity_pct = int((zone_count / max_cap) * 100)
            kpi_zone.metric("Stage Zone Count", f"{zone_count} / {max_cap}")
            kpi_pct.metric("Zone Capacity", f"{capacity_pct}%", delta=f"{zone_count - warning_cap} vs warn", delta_color="inverse")
            kpi_total.metric("Total Frame Detections", total_count)
            kpi_status.metric("Alert Status", status_label)

            if frame_idx % 5 == 0:
                st.session_state.history.append({"Time": datetime.now().strftime("%H:%M:%S"), "Stage Zone": zone_count})
                if len(st.session_state.history) > 40:
                    st.session_state.history.pop(0)
                df_hist = pd.DataFrame(st.session_state.history).set_index("Time")
                chart_placeholder.line_chart(df_hist, height=200)

                campus_df = pd.DataFrame([
                    {"Zone": "01. Main Stage (LIVE CAM)", "Count": f"{zone_count}/{max_cap}", "Status": status_label},
                    {"Zone": "02. Central Food Court", "Count": "42/60", "Status": "🟡 WARNING"},
                    {"Zone": "03. North Entry Gate", "Count": "18/50", "Status": "🟢 SAFE"},
                    {"Zone": "04. Tech Expo Hall", "Count": "29/80", "Status": "🟢 SAFE"},
                ])
                campus_placeholder.dataframe(campus_df, hide_index=True, use_container_width=True)

                if st.session_state.alert_logs:
                    log_placeholder.dataframe(pd.DataFrame(st.session_state.alert_logs), hide_index=True, use_container_width=True)
                else:
                    log_placeholder.info("No critical threshold breaches recorded yet.")

        cap.release()

# ==============================================================================
# VIEW 2: STUDENT LIVE CROWD GUIDE (PREVENTS BOTTLENECKS BEFORE THEY HAPPEN)
# ==============================================================================
else:
    st.title("🎒 FestPulse — Live Student Crowd & Queue Guide")
    st.caption("Skip the queues! Check live campus zone crowd levels and find open hangout spots in real time.")

    live_count = st.session_state.get("live_stage_count", 12)
    live_max = st.session_state.get("live_stage_max", 14)
    stage_pct = min(int((live_count / live_max) * 100), 100)

    if stage_pct >= 85:
        st.error(f"🚨 **Main Stage Barricade is currently PACKED ({stage_pct}% full)!** Recommended alternative: Head to **Amphitheater Chill Zone** (28% full) or **Tech Expo Arena** (36% full).")
    elif stage_pct >= 55:
        st.warning(f"⚠️ **Main Stage is filling up fast ({stage_pct}% full).** Arrive within 10 minutes or watch from the **North Lawn Screen**.")
    else:
        st.success(f"✅ **Main Stage has plenty of space ({stage_pct}% full)!** Great time to head over.")

    st.divider()
    st.subheader("📍 Live Campus Zones & Wait Times")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🎸 Main Stage Arena (Live AI Feed)")
        st.progress(stage_pct / 100)
        st.write(f"**Occupancy:** {stage_pct}% ({live_count}/{live_max} zone capacity) | **Entry Wait:** ~12 mins")

        st.markdown("### 🍔 Central Food Court")
        st.progress(0.75)
        st.write("**Occupancy:** 75% (Busy) | **Avg Stall Wait:** ~15 mins")
        st.info("💡 **Smart Tip:** Try **South Block Food Trucks** (only 25% full, <3 min wait).")

    with c2:
        st.markdown("### 💻 Tech Expo & Gaming Hall")
        st.progress(0.36)
        st.write("**Occupancy:** 36% (Chill) | **Entry Wait:** No queue")

        st.markdown("### 🚪 North vs. East Entry Gates")
        st.progress(0.22)
        st.write("**North Gate:** 22% (Fast Track) | **East Gate:** 88% (Congested)")
        st.success("✅ **Fastest Entry:** Use **North Entry Gate** right now.")
