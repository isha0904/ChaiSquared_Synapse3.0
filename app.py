# app.py
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from streamlit_autorefresh import st_autorefresh
import ric

# ── MobileNet classes ─────────────────────────────────────────────────────────
MOBILENET_CLASSES = [
    'Hazardous_batteries',
    'Hazardous_e-waste',
    'Hazardous_paints',
    'Hazardous_pesticides',
    'Non-Recyclable_ceramic_product',
    'Non-Recyclable_diapers',
    'Non-Recyclable_platics_bags_wrappers',
    'Non-Recyclable_sanitary_napkin',
    'Non-Recyclable_stroform_product',
    'Organic_coffee_tea_bags',
    'Organic_egg_shells',
    'Organic_food_scraps',
    'Organic_kitchen_waste',
    'Organic_yard_trimmings',
    'Recyclable_cans_all_type',
    'Recyclable_glass_containers',
    'Recyclable_paper_products',
    'Recyclable_plastic_bottles',
]
NUM_CLASSES = len(MOBILENET_CLASSES)

YOLO_CLASSES = [
    "ceramic", "e_waste", "glass", "hazardous",
    "metal", "organic", "paper", "plastic", "sanitary", "styrofoam"
]

DISPOSAL_RULES = {
    "Hazardous": (
        "⚠️ Hazardous Waste Facility",
        "Do NOT bin this. Take to a certified hazardous waste collection point.",
        "error"
    ),
    "Non-Recyclable_ceramic_product": (
        "🗑️ Black Bin — General Waste",
        "Ceramic is not recyclable. Place in general waste.",
        "error"
    ),
    "Non-Recyclable_diapers": (
        "🗑️ Black Bin — General Waste",
        "Seal in a bag and dispose in general waste.",
        "error"
    ),
    "Non-Recyclable_platics_bags_wrappers": (
        "🗑️ Black Bin — General Waste",
        "Soft plastics are not kerbside recyclable. Check for soft-plastic drop-off points.",
        "warning"
    ),
    "Non-Recyclable_sanitary_napkin": (
        "🗑️ Black Bin — General Waste",
        "Seal in a bag and dispose in general waste.",
        "error"
    ),
    "Non-Recyclable_stroform_product": (
        "🗑️ Black Bin — General Waste",
        "Styrofoam/EPS is generally not kerbside recyclable.",
        "warning"
    ),
    "Organic": (
        "🌱 Green/Compost Bin",
        "Place in your compost or organic waste bin.",
        "success"
    ),
    "Recyclable_cans_all_type": (
        "♻️ Blue Bin — Dry Recycling",
        "Rinse the can and place in the blue recycling bin.",
        "info"
    ),
    "Recyclable_glass_containers": (
        "🫙 Brown/Glass Bin",
        "Rinse and place in the glass/brown recycling bin.",
        "info"
    ),
    "Recyclable_paper_products": (
        "📄 Green/Paper Bin",
        "Flatten and keep dry. Place in the paper recycling bin.",
        "success"
    ),
    "Recyclable_plastic_bottles": (
        "♻️ Blue Bin — Dry Recycling",
        "Rinse, remove cap, and place in the blue recycling bin.",
        "info"
    ),
}

CARBON_SAVINGS_G = {
    "Hazardous": 0,
    "Non-Recyclable_ceramic_product": 0,
    "Non-Recyclable_diapers": 0,
    "Non-Recyclable_platics_bags_wrappers": 5,
    "Non-Recyclable_sanitary_napkin": 0,
    "Non-Recyclable_stroform_product": 5,
    "Organic": 10,
    "Recyclable_cans_all_type": 95,
    "Recyclable_glass_containers": 15,
    "Recyclable_paper_products": 20,
    "Recyclable_plastic_bottles": 30,
}

BADGES = [
    (1,   "🌱 First Sort"),
    (5,   "♻️  Getting Started"),
    (10,  "🌿 Eco Aware"),
    (25,  "💚 Green Citizen"),
    (50,  "🏅 Carbon Saver"),
    (100, "🏆 Recycling Hero"),
]

# ── Recyclers in Karvenagar ───────────────────────────────────────────────────
KARVENAGAR_RECYCLERS = [
    {
        "name": "Swachh Bharat Recycling Centre",
        "type": "General Recycling",
        "address": "Near Karvenagar Bus Stop, Karvenagar, Pune – 411052",
        "phone": "+91 98765 43210",
        "accepts": "Paper, Plastic, Metal, Glass",
        "icon": "♻️",
    },
    {
        "name": "E-Waste Collect Hub",
        "type": "E-Waste & Hazardous",
        "address": "Shop 4, Karvenagar Market, Pune – 411052",
        "phone": "+91 98234 56789",
        "accepts": "Electronics, Batteries, Paints, Chemicals",
        "icon": "⚡",
    },
    {
        "name": "Green Earth Waste Solutions",
        "type": "Organic & Compost",
        "address": "Opposite Karvenagar Garden, Pune – 411052",
        "phone": "+91 91234 00123",
        "accepts": "Organic waste, Food scraps, Yard waste",
        "icon": "🌱",
    },
    {
        "name": "Pune Municipal Corporation — Ward Office",
        "type": "Municipal Collection Point",
        "address": "Karvenagar Ward Office, Pune – 411052",
        "phone": "020-25501234",
        "accepts": "All categories (scheduled collection)",
        "icon": "🏛️",
    },
    {
        "name": "Kabadiwala Express",
        "type": "Scrap & Metal Dealer",
        "address": "Paud Road, Near Karvenagar, Pune – 411052",
        "phone": "+91 99700 88888",
        "accepts": "Metal cans, Cardboard, Newspapers, Plastic bottles",
        "icon": "🔩",
    },
]

# ── Fixed confidence threshold ────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.20

def get_rule(mobilenet_label: str):
    if mobilenet_label in DISPOSAL_RULES:
        return DISPOSAL_RULES[mobilenet_label], mobilenet_label
    parent = mobilenet_label.split("_")[0]
    if parent in DISPOSAL_RULES:
        return DISPOSAL_RULES[parent], parent
    return ("🗑️ Black Bin — General Waste", "Use general waste.", "error"), mobilenet_label

# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_yolo():
    return YOLO("YoloModel2.pt")

@st.cache_resource
def load_mobilenet():
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = torch.nn.Linear(m.last_channel, NUM_CLASSES)
    m.load_state_dict(
        torch.load("mobilenetv2.pt", map_location="cpu")
    )
    m.eval()
    return m

mobile_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_crop(mobilenet, pil_crop: Image.Image):
    tensor = mobile_transform(pil_crop).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(mobilenet(tensor), dim=1)
        conf, pred = torch.max(probs, dim=1)
    return MOBILENET_CLASSES[pred.item()], conf.item()

def get_top3(mobilenet, pil_crop: Image.Image):
    tensor = mobile_transform(pil_crop).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(mobilenet(tensor), dim=1)[0]
    top3 = torch.topk(probs, 3)
    return [(MOBILENET_CLASSES[i.item()], p.item())
            for p, i in zip(top3.values, top3.indices)]

SEVERITY_COLOURS_BGR = {
    "info":    (235, 150, 50),
    "success": (80,  175, 80),
    "warning": (50,  180, 230),
    "error":   (70,  70,  220),
}

def draw_boxes(frame_bgr, detections):
    out = frame_bgr.copy()
    for d in detections:
        colour = SEVERITY_COLOURS_BGR.get(d["severity"], (180, 180, 180))
        cv2.rectangle(out, (d["x1"], d["y1"]), (d["x2"], d["y2"]), colour, 2)
        short = (d["mobile_label"]
                 .replace("Non-Recyclable_", "").replace("Recyclable_", "")
                 .replace("_", " "))
        text = f"{short} {d['conf']:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out,
                      (d["x1"], d["y1"] - th - 8),
                      (d["x1"] + tw + 6, d["y1"]),
                      colour, -1)
        cv2.putText(out, text, (d["x1"] + 3, d["y1"] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out

# ── Session state init ────────────────────────────────────────────────────────

defaults = {
    "total_co2":         0,
    "items_sorted":      0,
    "badges":            [],   # badges already earned & dismissed
    "pending_badges":    [],   # badges earned but not yet dismissed
    "frozen":            False,
    "frozen_frame":      None,
    "frozen_detections": [],
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

def update_gamification(rule_key: str):
    st.session_state.total_co2    += CARBON_SAVINGS_G.get(rule_key, 0)
    st.session_state.items_sorted += 1
    for threshold, name in BADGES:
        if (st.session_state.items_sorted >= threshold
                and name not in st.session_state.badges
                and name not in st.session_state.pending_badges):
            st.session_state.pending_badges.append(name)

# ── Page layout ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Green Vision", page_icon="♻️", layout="wide")

# Apply dark forest green background
st.markdown("""
    <style>
    .stApp {
        background-color: #1a4d1a;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("♻️ Green Vision")
st.markdown("### Your smart assistant to cleanly dispose waste")

yolo      = load_yolo()
mobilenet = load_mobilenet()

# Auto-refresh only while camera is running (every 1.5 s)
if not st.session_state.frozen:
    st_autorefresh(interval=1500, key="cam_refresh")

# ── Badge pop-up (blocks until dismissed) ────────────────────────────────────
if st.session_state.pending_badges:
    badge = st.session_state.pending_badges[0]

    @st.dialog("🎉 Badge Unlocked!")
    def badge_dialog():
        st.markdown(
            f"<h2 style='text-align:center'>{badge}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='text-align:center;color:gray'>"
            f"You've sorted {st.session_state.items_sorted} item(s) correctly!</p>",
            unsafe_allow_html=True,
        )
        if st.button("Nice! ✅", use_container_width=True):
            st.session_state.pending_badges.remove(badge)
            st.session_state.badges.append(badge)
            st.rerun()

    badge_dialog()

# ── Tabs ──────────────────────────────────────────────────────────────────────
LEADERBOARD = [
    {"name": "Priya S.",  "avatar": "👩", "items": 87, "co2": 3420},
    {"name": "Rohan M.",  "avatar": "👦", "items": 71, "co2": 2810},
    {"name": "Ananya K.", "avatar": "👧", "items": 65, "co2": 2540},
    {"name": "You",       "avatar": "🫵", "items": None, "co2": None},  # filled dynamically
    {"name": "Arjun T.",  "avatar": "🧑", "items": 48, "co2": 1760},
    {"name": "Sneha P.",  "avatar": "👩", "items": 34, "co2": 1190},
    {"name": "Dev R.",    "avatar": "👦", "items": 21, "co2": 640},
]

tab_scan, tab_impact, tab_resin = st.tabs(["📷 Scan", "🌍 My Impact", "🔬 Resin Value Detection"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCAN
# ════════════════════════════════════════════════════════════════════════════════
with tab_scan:
    col_cam, col_result = st.columns([1.3, 1])

    with col_cam:
        st.subheader("Camera")
        annotated_ph = st.empty()

        if st.session_state.frozen:
            # Show only the annotated frame briefly in the camera slot
            # (no permanent frozen frame display below — see requirements)
            annotated_ph.image(
                cv2.cvtColor(st.session_state.frozen_frame, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )
            if st.button("🔄 Scan next item", use_container_width=True):
                st.session_state.frozen            = False
                st.session_state.frozen_frame      = None
                st.session_state.frozen_detections = []
                st.rerun()
        else:
            img_file = st.camera_input("Point camera at a waste item",
                                       label_visibility="collapsed")

        # ── Recyclers near Karvenagar ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📍 Karvenagar, Pune — Nearby Recyclers")
        for r in KARVENAGAR_RECYCLERS:
            with st.expander(f"{r['icon']}  {r['name']} — *{r['type']}*"):
                st.markdown(f"**📌 Address:** {r['address']}")
                st.markdown(f"**📞 Phone:** {r['phone']}")
                st.markdown(f"**✅ Accepts:** {r['accepts']}")

    with col_result:
        st.subheader("Results")
        result_ph = st.empty()

    # ── Inference ─────────────────────────────────────────────────────────────
    if not st.session_state.frozen and 'img_file' in dir() and img_file is not None:
        pil_img   = Image.open(img_file).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        yolo_results = yolo(frame_bgr, verbose=False, conf=CONFIDENCE_THRESHOLD)
        boxes        = yolo_results[0].boxes
        detections   = []

        if boxes is not None and len(boxes):
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue

                yolo_label                = YOLO_CLASSES[int(box.cls.item())]
                crop_pil                  = pil_img.crop((x1, y1, x2, y2))
                mobile_label, mobile_conf = classify_crop(mobilenet, crop_pil)
                top3                      = get_top3(mobilenet, crop_pil)
                (bin_label, instruction, severity), rule_key = get_rule(mobile_label)

                if mobile_conf >= CONFIDENCE_THRESHOLD:
                    detections.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "yolo_label":   yolo_label,
                        "mobile_label": mobile_label,
                        "conf":         mobile_conf,
                        "bin_label":    bin_label,
                        "instruction":  instruction,
                        "severity":     severity,
                        "rule_key":     rule_key,
                        "top3":         top3,
                    })

            if detections:
                annotated_bgr = draw_boxes(frame_bgr, detections)
                st.session_state.frozen            = True
                st.session_state.frozen_frame      = annotated_bgr
                st.session_state.frozen_detections = detections

                for d in detections:
                    update_gamification(d["rule_key"])

                st.rerun()
            else:
                annotated_ph.image(
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )

    # ── Result cards ──────────────────────────────────────────────────────────
    display_detections = (
        st.session_state.frozen_detections if st.session_state.frozen else []
    )

    with result_ph.container():
        if st.session_state.frozen and not display_detections:
            st.info("No waste detected — try better lighting or move closer.")
        elif not st.session_state.frozen and not display_detections:
            st.info("Scanning… hold the item steady in frame.")

        for i, d in enumerate(display_detections):
            short = (d["mobile_label"]
                     .replace("Non-Recyclable_", "").replace("Recyclable_", "")
                     .replace("_", " ").title())
            getattr(st, d["severity"])(
                f"**{short}** ({d['conf']:.0%} confidence)  \n"
                #f"YOLO detected: `{d['yolo_label']}`  \n"
                f"➜ **{d['bin_label']}**  \n"
                f"{d['instruction']}"
            )
            with st.expander(f"Top 3 predictions — object {i+1}"):
                for cls, prob in d["top3"]:
                    st.write(f"`{cls.replace('_', ' ')}` — {prob*100:.1f}%")
                    st.progress(prob)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — MY IMPACT
# ════════════════════════════════════════════════════════════════════════════════
with tab_impact:
    st.markdown("## 🌍 Your Environmental Impact")
    st.markdown("---")

    m1, m2 = st.columns(2)
    m1.metric("🗂️ Items Sorted",  st.session_state.items_sorted)
    m2.metric("💨 Carbon Saved",  f"{st.session_state.total_co2} g CO₂")

    with st.expander("ℹ️ How does recycling save CO₂?"):
        st.markdown("""
        When waste is **not recycled**, it typically ends up being:
        - 🔥 **Incinerated** — burning releases CO₂, methane, and toxic gases directly into the atmosphere.
        - 🏔️ **Landfilled** — organic waste decomposes anaerobically, producing **methane**, which is ~25× more potent than CO₂ as a greenhouse gas.

        Recycling breaks this chain in two ways:

        | Mechanism | How it helps |
        |---|---|
        | **Avoids burning** | No combustion = no direct CO₂ or particulate emissions |
        | **Replaces virgin production** | Making aluminium from recycled cans uses ~95% less energy than smelting raw ore |
        | **Reduces methane from landfill** | Organic waste composted properly emits far less greenhouse gas |

        **Estimates used in this app** *(per item, approximate)*:
        - 🥫 Metal can → **95 g CO₂** saved (energy-intensive smelting avoided)
        - 🧴 Plastic bottle → **30 g CO₂** saved (fossil-fuel feedstock avoided)
        - 📄 Paper → **20 g CO₂** saved (logging + pulping energy avoided)
        - 🫙 Glass → **15 g CO₂** saved (furnace energy reduced)
        - 🌱 Organic waste → **10 g CO₂eq** saved (methane from landfill avoided)

        > Sources: WRAP UK, EPA lifecycle assessments, and IPCC waste sector reports.
        """)

    st.markdown("---")
    st.markdown("### 🏅 Badges Earned")
    if st.session_state.badges:
        badge_cols = st.columns(min(len(st.session_state.badges), 4))
        for idx, badge in enumerate(st.session_state.badges):
            with badge_cols[idx % 4]:
                st.success(badge)
    else:
        st.info("No badges yet — start scanning waste items to earn your first badge!")

    st.markdown("---")
    st.markdown("### 🎯 Next Badge")
    next_badge = None
    for threshold, name in BADGES:
        if name not in st.session_state.badges and name not in st.session_state.pending_badges:
            next_badge = (threshold, name)
            break
    if next_badge:
        threshold, name = next_badge
        progress = min(st.session_state.items_sorted / threshold, 1.0)
        st.markdown(f"**{name}** — sort **{threshold}** items")
        st.progress(progress)
        remaining = max(0, threshold - st.session_state.items_sorted)
        st.caption(f"{remaining} more item(s) to go!")
    else:
        st.success("🏆 You've earned all available badges — amazing work!")

    st.markdown("---")
    if st.button("🔄 Reset Session", use_container_width=True):
        for key in defaults:
            st.session_state[key] = defaults[key]
        st.rerun()

    # ── Leaderboard ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏆 Friends Leaderboard — CO₂ Saved")

    # Inject live "You" stats
    board = []
    for entry in LEADERBOARD:
        if entry["name"] == "You":
            board.append({**entry,
                          "items": st.session_state.items_sorted,
                          "co2":   st.session_state.total_co2})
        else:
            board.append(entry)

    # Sort by CO₂ descending
    board_sorted = sorted(board, key=lambda x: x["co2"], reverse=True)

    MEDAL = {0: "🥇", 1: "🥈", 2: "🥉"}

    for rank, entry in enumerate(board_sorted):
        is_you = entry["name"] == "You"
        medal  = MEDAL.get(rank, f"#{rank+1}")
        bg     = "padding:6px 12px;"
        bar_pct = int(entry["co2"] / board_sorted[0]["co2"] * 100) if board_sorted[0]["co2"] > 0 else 0

        col_rank, col_name, col_items, col_co2, col_bar = st.columns([0.5, 2, 1.2, 1.5, 3])
        with col_rank:
            st.markdown(f"<div style='{bg}'><b>{medal}</b></div>", unsafe_allow_html=True)
        with col_name:
            st.markdown(f"<div style='{bg}'>{entry['avatar']} {'**You**' if is_you else entry['name']}</div>",
                        unsafe_allow_html=True)
        with col_items:
            st.markdown(f"<div style='{bg}'>🗂️ {entry['items']}</div>", unsafe_allow_html=True)
        with col_co2:
            st.markdown(f"<div style='{bg}'>💨 {entry['co2']} g</div>", unsafe_allow_html=True)
        with col_bar:
            st.progress(bar_pct / 100)

# ── Sidebar (clean — no impact metrics) ──────────────────────────────────────
st.sidebar.markdown("## ♻️ Eco-Label Vision")
st.sidebar.markdown("Point your camera at any waste item and the app will tell you which bin it belongs in.")
st.sidebar.markdown("---")
st.sidebar.markdown("**📍 Location:** Karvenagar, Pune, India")
st.sidebar.markdown("Switch to the **🌍 My Impact** tab to view your stats and badges.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESIN VALUE DETECTION (coming soon)
# ════════════════════════════════════════════════════════════════════════════════
with tab_resin:
    ric.run()
