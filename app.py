import customtkinter as ctk
import joblib
import pandas as pd
import os
from tkinter import messagebox
import random
import time
import re
import threading
import google.generativeai as genai
from dotenv import load_dotenv
from vehicle_knowledge import (
    extract_vin,
    vpci_decode_vin,
    carquery_search,
    parse_year_make_model,
    vpci_models_for_make_year,
    parse_make_model,
    vpci_models_for_make,
    format_vehicle_profile,
)

# --- Gemini error handling helpers ---
_RETRY_IN_SECONDS_RE = re.compile(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)
_RETRY_DELAY_SECONDS_RE = re.compile(r"retry_delay\s*\{\s*seconds:\s*([0-9]+)\s*\}", re.IGNORECASE)

_TELEMETRY_KV_RE = re.compile(
    r"\b(speed|rpm|engine[_\s-]*rpm|throttle|throttle[_\s-]*position|load|engine[_\s-]*load)\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)

def _extract_retry_seconds(error_text: str) -> int | None:
    if not error_text:
        return None
    m = _RETRY_IN_SECONDS_RE.search(error_text)
    if m:
        try:
            return max(0, int(float(m.group(1))))
        except Exception:
            return None
    m = _RETRY_DELAY_SECONDS_RE.search(error_text)
    if m:
        try:
            return max(0, int(m.group(1)))
        except Exception:
            return None
    return None

def _is_quota_429(error_text: str) -> bool:
    if not error_text:
        return False
    t = error_text.lower()
    return (" 429" in t or "too many requests" in t or "quota exceeded" in t) and (
        "rate limit" in t or "quota" in t or "exceeded your current quota" in t
    )

def _parse_telemetry_from_text(text: str) -> dict:
    """
    Extract telemetry from free-form text.
    Returns dict with keys: Speed, Engine_RPM, Throttle_Position, Engine_Load if present.
    """
    out: dict[str, float] = {}
    for key, val in _TELEMETRY_KV_RE.findall(text or ""):
        k = key.lower().replace(" ", "_").replace("-", "_")
        try:
            f = float(val)
        except Exception:
            continue
        if k in ("speed",):
            out["Speed"] = f
        elif k in ("rpm", "engine_rpm"):
            out["Engine_RPM"] = f
        elif k in ("throttle", "throttle_position"):
            out["Throttle_Position"] = f
        elif k in ("load", "engine_load"):
            out["Engine_Load"] = f
    return out

# Robust API key loading
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Manual fallback if load_dotenv fails
if not GEMINI_API_KEY and os.path.exists(".env"):
    try:
        with open(".env", "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == "GEMINI_API_KEY":
                        GEMINI_API_KEY = v.strip().replace('"', '').replace("'", "")
                        break
    except:
        pass

if GEMINI_API_KEY:
    GEMINI_API_KEY = GEMINI_API_KEY.strip().replace('"', '').replace("'", "")
    genai.configure(api_key=GEMINI_API_KEY)

MODEL_FILE = 'ecodrive_model.pkl'
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Set appearance and theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue") # Using blue as base for teal overrides

class EcoDriveApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("EcoDrive: Universal Automotive AI")
        self.geometry("1100x800")
        self.configure(fg_color="#0a0a0a")
        
        self.grid_rowconfigure(0, weight=1)

        self.model = None
        self.load_model()
        
        # Initialize Gemini Model (Virtual Chief Engineer - "engineer help")
        self.gemini_connected = False
        self.gemini_init_error = None
        self.gemini_cooldown_until_ts = 0.0
        self.gemini_quota_blocked = False
        self._chat_busy = False
        if GEMINI_API_KEY:
            try:
                # Gemini model can be overridden via GEMINI_MODEL env var
                self.gemini_model = genai.GenerativeModel(
                    model_name=GEMINI_MODEL_NAME,
                    system_instruction=(
                        "You are 'engineer help', an advanced Automotive Engineering AI and Virtual Chief Engineer. "
                        "Your expertise includes high-performance telemetry analysis, ICE mechanics, EV battery management systems, "
                        "and UN SDGs 11 and 13. Tone: Professional, expert, data-driven. "
                        "If a query is non-automotive, professionally redirect the user to automotive engineering topics."
                    )
                )
                self.gemini_connected = True
                print(f"Gemini API Initialized for 'engineer help' using {GEMINI_MODEL_NAME}.")
            except Exception as e:
                print(f"Gemini Init Error: {e}")
                try:
                    # Final fallback without system instruction
                    self.gemini_model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
                    self.gemini_connected = True
                    print("Gemini API Initialized (Legacy Mode).")
                except Exception as e2:
                    self.gemini_init_error = str(e2)
                    print(f"Gemini Critical Error: {e2}")
                    self.gemini_connected = False
        else:
            self.gemini_init_error = "API Key not found."
        
        # Local database removed as per user request to rely on Gemini Output.
        self.model_db = {}
        
        self.create_widgets()
        
    def _set_chat_busy(self, busy: bool, status_text: str | None = None):
        self._chat_busy = busy
        try:
            state = "disabled" if busy else "normal"
            self.chat_entry.configure(state=state)
            self.send_button.configure(state=state)
        except Exception:
            pass
        if status_text:
            try:
                self.bot_status_label.configure(text=status_text)
            except Exception:
                pass

    def _append_bot_message(self, text: str):
        self.update_chat_history(f"🛠️ engineer help: {text}\n\n")

    def _respond_in_thread(self, user_msg: str):
        """
        Do not call UI methods directly from this thread.
        Compute response text, then schedule UI updates with self.after().
        """
        now = time.time()
        # If quota blocked or cooling down, always local.
        force_local = self.gemini_quota_blocked or (now < self.gemini_cooldown_until_ts) or (not self.gemini_connected)

        def finalize(text: str, status: str):
            self.after(
                0,
                lambda: (
                    self._append_bot_message(text),
                    self._set_chat_busy(False, status),
                ),
            )

        if force_local:
            if self.gemini_quota_blocked:
                status = "🛠️ engineer help: Offline (Local Mode)"
            elif now < self.gemini_cooldown_until_ts:
                remaining = max(1, int(self.gemini_cooldown_until_ts - now))
                status = f"🛠️ engineer help: Cooling down… (~{remaining}s)"
            else:
                status = "🛠️ engineer help: Offline (Local Mode)"
            finalize(self.offline_engineer_help(user_msg), status)
            return

        # Cloud attempt
        try:
            response = self.gemini_model.generate_content(user_msg)
            if response and response.text:
                finalize(response.text, "🛠️ engineer help: Ready for Optimization")
                return
            raise Exception("Empty response or blocked by safety filters.")
        except Exception as e:
            err_text = str(e)
            print(f"Gemini Generation Error: {err_text}")
            if _is_quota_429(err_text):
                retry_s = _extract_retry_seconds(err_text) or 30
                self.gemini_cooldown_until_ts = time.time() + retry_s

                if "limit: 0" in err_text.lower() or "free_tier_requests" in err_text.lower():
                    self.gemini_quota_blocked = True
                    local = self.offline_engineer_help(user_msg)
                    finalize(
                        "Cloud quota blocked; switched to local mode.\n\n" + local,
                        "🛠️ engineer help: Offline (Local Mode)",
                    )
                else:
                    local = self.offline_engineer_help(user_msg)
                    finalize(
                        f"Rate limited by Gemini API; cooling down ~{retry_s}s. Local response below.\n\n{local}",
                        "🛠️ engineer help: Cooling down…",
                    )
            else:
                local = self.offline_engineer_help(user_msg)
                finalize(
                    "Cloud error; switched to local mode.\n\n" + local,
                    "🛠️ engineer help: Offline (Local Mode)",
                )

    def offline_engineer_help(self, user_msg: str) -> str:
        """
        Local fallback response generator when Gemini is unavailable.
        Tries to:
        - Answer common automotive optimization questions with heuristics
        - Parse telemetry from the message and (if model loaded) predict L/100km
        """
        msg = (user_msg or "").strip()
        if not msg:
            return "I’m offline, but ready. Ask about ICE/EV efficiency, gearing, aero, tires, or paste telemetry (speed/rpm/throttle/load)."

        telemetry = _parse_telemetry_from_text(msg)
        has_all = all(k in telemetry for k in ("Speed", "Engine_RPM", "Throttle_Position", "Engine_Load"))

        lines: list[str] = []
        lines.append("Offline Engineer Mode (local heuristics).")

        # Vehicle lookup (VIN or year/make/model)
        vin = extract_vin(msg)
        if vin:
            prof = vpci_decode_vin(vin, use_cache=True)
            if prof:
                lines.append("")
                lines.append("Vehicle identification (cached lookup):")
                lines.append(format_vehicle_profile(prof))
            else:
                lines.append("")
                lines.append("Vehicle identification: VIN detected but lookup failed (offline or service unavailable).")

        # Basic year/make/model attempt (best-effort)
        if not vin:
            parsed = parse_year_make_model(msg)
            if parsed:
                y, mk, md = parsed
                prof = carquery_search(y, mk, md, use_cache=True)
                if prof:
                    lines.append("")
                    lines.append("Vehicle trims/specs (cached lookup):")
                    lines.append(format_vehicle_profile(prof))
                else:
                    # vPIC fallback for older vehicles / trim coverage gaps
                    vp = vpci_models_for_make_year(mk, y, use_cache=True)
                    if vp:
                        lines.append("")
                        lines.append("Vehicle catalog (cached lookup):")
                        lines.append(format_vehicle_profile(vp))
                        if md.lower() not in " ".join((vp.data.get("models") or [])).lower():
                            lines.append(f"- Note: '{md.title()}' not found in vPIC model list for {mk.title()} {y}.")
                    else:
                        # Some makes/years return empty from vPIC (especially older years). Try make-only list.
                        vp2 = vpci_models_for_make(mk, use_cache=True)
                        if vp2:
                            lines.append("")
                            lines.append("Vehicle catalog (cached lookup):")
                            lines.append(format_vehicle_profile(vp2))
                            lines.append(f"- Note: vPIC did not return a {y}-specific list for {mk.title()}; showing all-years catalog instead.")
                            if md.lower() not in " ".join((vp2.data.get("models") or [])).lower():
                                lines.append(f"- Note: '{md.title()}' not found in vPIC model list for {mk.title()} (all years).")
                        else:
                            lines.append("")
                            lines.append("Vehicle lookup: could not reach vehicle catalog service (check internet) and no cached data found.")
            else:
                mm = parse_make_model(msg)
                if mm:
                    mk, md = mm
                    vp = vpci_models_for_make(mk, use_cache=True)
                    if vp:
                        lines.append("")
                        lines.append("Vehicle catalog (cached lookup):")
                        lines.append(format_vehicle_profile(vp))
                        if md.lower() not in " ".join((vp.data.get("models") or [])).lower():
                            lines.append(f"- Note: '{md.title()}' not found in vPIC model list for {mk.title()} (all years).")
                    else:
                        lines.append("")
                        lines.append("Vehicle lookup: could not reach vehicle catalog service (check internet) and no cached data found.")

        # Telemetry-driven advice
        if telemetry:
            lines.append("")
            lines.append("Telemetry detected:")
            if "Speed" in telemetry:
                lines.append(f"- Speed: {telemetry['Speed']:.1f} km/h")
            if "Engine_RPM" in telemetry:
                lines.append(f"- RPM: {telemetry['Engine_RPM']:.0f}")
            if "Throttle_Position" in telemetry:
                lines.append(f"- Throttle: {telemetry['Throttle_Position']:.0f}%")
            if "Engine_Load" in telemetry:
                lines.append(f"- Load: {telemetry['Engine_Load']:.0f}%")

            # Model-based prediction if possible
            if has_all and self.model is not None:
                try:
                    input_df = pd.DataFrame(
                        {
                            "Speed": [telemetry["Speed"]],
                            "Engine_RPM": [telemetry["Engine_RPM"]],
                            "Throttle_Position": [telemetry["Throttle_Position"]],
                            "Engine_Load": [telemetry["Engine_Load"]],
                        }
                    )
                    pred = float(self.model.predict(input_df)[0])
                    lines.append("")
                    lines.append(f"Estimated fuel consumption: {pred:.2f} L/100km (model).")
                    if pred > 50.0:
                        lines.append("Status: UNSUSTAINABLE consumption (very high).")
                    elif pred > 30.0:
                        lines.append("Status: HIGH consumption.")
                    else:
                        lines.append("Status: OPTIMAL efficiency range.")
                except Exception:
                    lines.append("")
                    lines.append("Fuel estimate unavailable (model error).")

            # Heuristic coaching (works even without model)
            rpm = telemetry.get("Engine_RPM")
            speed = telemetry.get("Speed")
            throttle = telemetry.get("Throttle_Position")
            load = telemetry.get("Engine_Load")
            tips: list[str] = []
            if rpm is not None and speed is not None and rpm > 4000 and speed < 100:
                tips.append("Shift up / reduce RPM (high RPM at low road speed wastes fuel).")
            if throttle is not None and throttle > 70:
                tips.append("Soften throttle inputs; aim for smoother pedal ramps.")
            if load is not None and load > 80:
                tips.append("High load: avoid abrupt acceleration; keep momentum, reduce grade/drag where possible.")
            if speed is not None and speed > 120:
                tips.append("Aerodynamic drag rises fast above ~100–110 km/h; reducing speed is the biggest win.")
            if tips:
                lines.append("")
                lines.append("Optimization actions:")
                lines.extend([f"- {t}" for t in tips])
            else:
                lines.append("")
                lines.append("Optimization actions: your inputs don’t trigger any major red flags; focus on smoothness and steady-state cruising.")

        # General offline knowledge for common topics
        lower = msg.lower()
        if any(w in lower for w in ("mileage", "fuel", "consumption", "mpg", "l/100", "efficiency")) and not telemetry:
            lines.append("")
            lines.append("General efficiency levers (ranked): speed (aero), throttle smoothness, gearing/RPM, tire pressure/rolling resistance, unnecessary mass, and maintenance (alignment, plugs, filters).")
        if any(w in lower for w in ("ev", "battery", "bms", "regen")):
            lines.append("")
            lines.append("EV quick wins: precondition battery when possible, moderate speed, use regen strategically (avoid brake-to-zero when you could coast), and keep tires properly inflated.")
        if any(w in lower for w in ("turbo", "boost", "knock", "octane")):
            lines.append("")
            lines.append("Turbo/knock basics: higher load + high IAT increases knock risk; good intercooling, correct octane, and conservative timing/boost targets protect the engine.")

        lines.append("")
        lines.append("Tip: you can paste a VIN (17 chars) or 'YEAR MAKE MODEL' for cached vehicle lookup.")
        lines.append("If you want a precise analysis, paste: speed=…, rpm=…, throttle=…, load=… (all four).")
        return "\n".join(lines)

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            try:
                self.model = joblib.load(MODEL_FILE)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
        else:
            messagebox.showwarning("Warning", f"{MODEL_FILE} not found. Please run train_model.py first.")
            
    def create_widgets(self):
        # Create Tabview
        self.tabview = ctk.CTkTabview(self, fg_color="#121212", segmented_button_selected_color="#00adb5", 
                                       segmented_button_unselected_hover_color="#393e46")
        self.tabview.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.tabview.add("Predictor")
        self.tabview.add("engineer help")
        
        self.setup_predictor_tab()
        self.setup_autochat_tab()

    def setup_predictor_tab(self):
        parent = self.tabview.tab("Predictor")
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        # Sidebar-like Inner Frame (Input Panel)
        self.input_panel = ctk.CTkFrame(parent, width=320, corner_radius=15, fg_color="#1b1b1b")
        self.input_panel.grid(row=0, column=0, padx=(10, 20), pady=10, sticky="nsew")
        self.input_panel.grid_rowconfigure(6, weight=1)

        self.logo_label = ctk.CTkLabel(self.input_panel, text="⚡ EcoDrive", font=ctk.CTkFont(size=28, weight="bold"), text_color="#00adb5")
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 5))

        self.subtitle_label = ctk.CTkLabel(self.input_panel, text="Universal Analytics Engine", font=ctk.CTkFont(size=14, slant="italic"), text_color="#393e46")
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 20))

        # Input elements
        input_label_color = "#eeeeee"
        
        self.speed_label = ctk.CTkLabel(self.input_panel, text="Speed (km/h):", text_color=input_label_color)
        self.speed_label.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")
        self.speed_entry = ctk.CTkEntry(self.input_panel, placeholder_text="e.g. 110", border_color="#393e46", fg_color="#0a0a0a")
        self.speed_entry.grid(row=2, column=0, padx=20, pady=(35, 10), sticky="ew")

        self.rpm_label = ctk.CTkLabel(self.input_panel, text="Engine RPM:", text_color=input_label_color)
        self.rpm_label.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.rpm_entry = ctk.CTkEntry(self.input_panel, placeholder_text="e.g. 3200", border_color="#393e46", fg_color="#0a0a0a")
        self.rpm_entry.grid(row=3, column=0, padx=20, pady=(35, 10), sticky="ew")

        self.throttle_label = ctk.CTkLabel(self.input_panel, text="Throttle % (0-100):", text_color=input_label_color)
        self.throttle_label.grid(row=4, column=0, padx=20, pady=(10, 0), sticky="w")
        self.throttle_entry = ctk.CTkEntry(self.input_panel, placeholder_text="e.g. 45", border_color="#393e46", fg_color="#0a0a0a")
        self.throttle_entry.grid(row=4, column=0, padx=20, pady=(35, 10), sticky="ew")

        self.load_label = ctk.CTkLabel(self.input_panel, text="Engine Load % (0-100):", text_color=input_label_color)
        self.load_label.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        self.load_entry = ctk.CTkEntry(self.input_panel, placeholder_text="e.g. 60", border_color="#393e46", fg_color="#0a0a0a")
        self.load_entry.grid(row=5, column=0, padx=20, pady=(35, 10), sticky="ew")

        self.predict_button = ctk.CTkButton(self.input_panel, text="Analyze Efficiency", command=self.predict, 
                                            font=ctk.CTkFont(weight="bold"), fg_color="#00adb5", hover_color="#008c95", text_color="#eeeeee")
        self.predict_button.grid(row=7, column=0, padx=20, pady=20)

        # Main Content Area
        self.main_content = ctk.CTkFrame(parent, fg_color="transparent")
        self.main_content.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.main_content.grid_rowconfigure(2, weight=1)
        self.main_content.grid_columnconfigure(0, weight=1)

        self.result_card = ctk.CTkFrame(self.main_content, corner_radius=15, border_width=1, border_color="#393e46", fg_color="#1b1b1b")
        self.result_card.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.result_title = ctk.CTkLabel(self.result_card, text="Carbon Footprint Analysis", font=ctk.CTkFont(size=20, weight="bold"), text_color="#00adb5")
        self.result_title.pack(pady=(20, 5))

        self.result_val_label = ctk.CTkLabel(self.result_card, text="-- L/100km", font=ctk.CTkFont(size=48, weight="bold"), text_color="#eeeeee")
        self.result_val_label.pack(pady=(0, 20))

        self.status_card = ctk.CTkFrame(self.main_content, corner_radius=15, fg_color="#1b1b1b")
        self.status_card.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.status_label = ctk.CTkLabel(self.status_card, text="Standby Mode", font=ctk.CTkFont(size=18, weight="bold"), text_color="#393e46")
        self.status_label.pack(pady=20)

        self.rec_card = ctk.CTkFrame(self.main_content, corner_radius=15, fg_color="#222831", border_width=1, border_color="#00adb5")
        self.rec_card.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        self.rec_title = ctk.CTkLabel(self.rec_card, text="AI Driving Insights", font=ctk.CTkFont(size=18, weight="bold"), text_color="#00adb5")
        self.rec_title.pack(pady=(15, 5))

        self.rec_text_label = ctk.CTkLabel(self.rec_card, text="Initialize telemetry stream for universal vehicle profile matching and efficiency optimization.", 
                                          font=ctk.CTkFont(size=14, slant="italic"), text_color="#eeeeee", wraplength=550)
        self.rec_text_label.pack(padx=20, pady=(0, 20))

    def setup_autochat_tab(self):
        parent = self.tabview.tab("engineer help")
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)

        # Status Display
        self.bot_status_label = ctk.CTkLabel(parent, text="🛠️ engineer help: Virtual Chief Engineer Online", font=ctk.CTkFont(size=13, weight="bold"), text_color="#00adb5")
        self.bot_status_label.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="w")

        self.chat_frame = ctk.CTkFrame(parent, corner_radius=15, fg_color="#1b1b1b", border_width=1, border_color="#393e46")
        self.chat_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.chat_frame.grid_rowconfigure(0, weight=1)
        self.chat_frame.grid_columnconfigure(0, weight=1)

        self.chat_history = ctk.CTkTextbox(self.chat_frame, font=ctk.CTkFont(size=15), text_color="#eeeeee", fg_color="#0a0a0a", border_color="#393e46")
        self.chat_history.grid(row=0, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")
        
        init_msg = "🛠️ engineer help: Telemetry linked. I am your Virtual Chief Engineer.\n"
        if self.gemini_connected:
            init_msg += "⚡ Status: Gemini GenAI Online (Automotive Expert Mode)\n"
        else:
            init_msg += f"⚡ Status: Offline ({self.gemini_init_error if self.gemini_init_error else 'Check API Key'})\n"
            
        init_msg += "\nHow can we optimize your vehicle's setup and emissions today?\n\n"
        
        self.chat_history.insert("0.0", init_msg)
        self.chat_history.configure(state="disabled")

        # Quick Links Frame
        self.quick_links_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.quick_links_frame.grid(row=2, column=0, padx=20, pady=(10, 5), sticky="ew")
        
        links = [("📜 Classic (Pre-1950)", "past"), ("🚀 Modern (Current)", "present"), ("🛸 Future Concepts", "future"), ("🗑️ Clear Scan", "clear")]
        for i, (text, cmd) in enumerate(links):
            btn = ctk.CTkButton(self.quick_links_frame, text=text, width=150, height=32, corner_radius=20,
                                fg_color="#222831", border_width=1, border_color="#393e46", hover_color="#393e46",
                                command=lambda c=cmd: self.quick_action(c))
            btn.grid(row=0, column=i, padx=5, sticky="ew")
            self.quick_links_frame.grid_columnconfigure(i, weight=1)

        # Chat Entry Frame
        self.input_area = ctk.CTkFrame(parent, fg_color="transparent")
        self.input_area.grid(row=3, column=0, padx=20, pady=(5, 20), sticky="ew")
        self.input_area.grid_columnconfigure(0, weight=1)

        self.chat_entry = ctk.CTkEntry(self.input_area, placeholder_text="Ask about any vehicle or era...", height=45, border_color="#393e46", fg_color="#0a0a0a")
        self.chat_entry.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.chat_entry.bind("<Return>", lambda e: self.send_message())

        self.send_button = ctk.CTkButton(self.input_area, text="Transmit", width=120, height=45, command=self.send_message, fg_color="#00adb5", hover_color="#008c95")
        self.send_button.grid(row=0, column=1)

    def quick_action(self, cmd):
        if cmd == "clear":
            self.chat_history.configure(state="normal")
            self.chat_history.delete("1.0", "end")
            self.chat_history.insert("1.0", "🛠️ engineer help: Memory banks cleared. Ready for new input.\n\n")
            self.chat_history.configure(state="disabled")
            return
        
        self.chat_entry.delete(0, 'end')
        self.chat_entry.insert(0, cmd)
        self.send_message()

    def send_message(self):
        user_msg = self.chat_entry.get().strip()
        if not user_msg:
            return

        self.update_chat_history(f"You: {user_msg}\n")
        self.chat_entry.delete(0, 'end')

        if self._chat_busy:
            return
        self._set_chat_busy(True, "🛠️ engineer help: Analyzing telemetry patterns…")
        threading.Thread(target=self._respond_in_thread, args=(user_msg,), daemon=True).start()

    def process_response(self, user_msg):
        # Only use Gemini as per user request
        if self.gemini_connected:
            try:
                response = self.gemini_model.generate_content(user_msg)
                if response and response.text:
                    bot_reply = response.text
                    self.update_chat_history(f"🛠️ engineer help: {bot_reply}\n\n")
                else:
                    raise Exception("Empty response or blocked by safety filters.")
            except Exception as e:
                err_text = str(e)
                print(f"Gemini Generation Error: {err_text}")
                if _is_quota_429(err_text):
                    retry_s = _extract_retry_seconds(err_text) or 30
                    self.gemini_cooldown_until_ts = time.time() + retry_s

                    # If the API reports quota limit 0, treat as hard-blocked until user fixes billing/quota.
                    if "limit: 0" in err_text.lower() or "free_tier_requests" in err_text.lower():
                        self.gemini_quota_blocked = True
                        self.bot_status_label.configure(text="🛠️ engineer help: Offline (Local Mode)")
                        bot_reply = self.offline_engineer_help(user_msg)
                        self.update_chat_history(
                            "🛠️ engineer help (System): Cloud quota blocked; switching to local mode.\n"
                            f"🛠️ engineer help: {bot_reply}\n\n"
                        )
                    else:
                        self.bot_status_label.configure(text="🛠️ engineer help: Cooling down…")
                        bot_reply = self.offline_engineer_help(user_msg)
                        self.update_chat_history(
                            f"🛠️ engineer help (System): Rate limited by Gemini API. Cooling down ~{retry_s}s.\n"
                            f"🛠️ engineer help: (Local)\n{bot_reply}\n\n"
                        )
                else:
                    bot_reply = self.offline_engineer_help(user_msg)
                    self.update_chat_history(
                        "🛠️ engineer help (System): Cloud connection issue; using local mode.\n"
                        f"🛠️ engineer help: {bot_reply}\n\n"
                    )
        else:
            bot_reply = self.offline_engineer_help(user_msg)
            self.update_chat_history(
                "🛠️ engineer help (System): Cloud is offline (missing API key or init error); using local mode.\n"
                f"🛠️ engineer help: {bot_reply}\n\n"
            )
            
        self.bot_status_label.configure(text="🛠️ engineer help: Ready for Optimization")

    def update_chat_history(self, message):
        self.chat_history.configure(state="normal")
        self.chat_history.insert("end", message)
        self.chat_history.see("end")
        self.chat_history.configure(state="disabled")

    def get_universal_response(self, message):
        return "Local fallback retired. Please reconnect to the Gemini network."

    def predict(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded. Please ensure ecodrive_model.pkl is present.")
            self.load_model()
            if self.model is None:
                return
            
        try:
            speed = float(self.speed_entry.get())
            rpm = float(self.rpm_entry.get())
            throttle = float(self.throttle_entry.get())
            load = float(self.load_entry.get())
            
            input_df = pd.DataFrame({
                'Speed': [speed],
                'Engine_RPM': [rpm],
                'Throttle_Position': [throttle],
                'Engine_Load': [load]
            })
            
            prediction = self.model.predict(input_df)[0]
            
            self.result_val_label.configure(text=f"{prediction:.2f} L/100km")
            
            # Update status color and text
            if prediction > 50.0:
                 self.status_label.configure(text="⚠️ UNSUSTAINABLE CONSUMPTION", text_color="#e74c3c")
            elif prediction > 30.0:
                 self.status_label.configure(text="⚠️ HIGH CONSUMPTION", text_color="#f39c12")
            else:
                 self.status_label.configure(text="✅ OPTIMAL EFFICIENCY", text_color="#2ecc71")
                 
            # Recommendation logic
            recommendations = []
            if rpm > 4000 and speed < 100:
                recommendations.append("Shift to a higher gear to reduce RPM.")
            if throttle > 70:
                recommendations.append("Ease off the accelerator to save fuel.")
            if load > 80:
                recommendations.append("Engine load is high; try maintaining momentum to reduce strain.")
            if speed > 120:
                recommendations.append("Reduce speed to improve aerodynamic efficiency.")
            
            if not recommendations and prediction <= 30.0:
                rec_text = "Universal profile matching successful. Your current driving is highly efficient for this vehicle class."
            elif not recommendations:
                rec_text = "Adjusting driving behavior for smoother transitions will further optimize your footprint."
            else:
                rec_text = "💡 ECO-OPTIMIZATION: " + " ".join(recommendations)
                
            self.rec_text_label.configure(text=rec_text)
                 
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    app = EcoDriveApp()
    app.mainloop()
