import tkinter as tk
import customtkinter as ctk
import google.generativeai as genai
import os
from dotenv import load_dotenv
import threading
import time
import re
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

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not found in .env file.")

# --- Gemini error handling helpers ---
_RETRY_IN_SECONDS_RE = re.compile(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)
_RETRY_DELAY_SECONDS_RE = re.compile(r"retry_delay\s*\{\s*seconds:\s*([0-9]+)\s*\}", re.IGNORECASE)

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

def _offline_engineer_help(user_msg: str) -> str:
    msg = (user_msg or "").strip()
    if not msg:
        return "System: Offline local mode ready. Ask about ICE/EV efficiency, gearing, aero, tires, maintenance, or driving technique."

    lower = msg.lower()
    lines: list[str] = []
    lines.append("Offline Engineer Mode (local heuristics).")

    # Vehicle lookup (VIN or year/make/model)
    vin = extract_vin(msg)
    if vin:
        prof = vpci_decode_vin(vin, use_cache=True)
        lines.append("")
        if prof:
            lines.append("Vehicle identification (cached lookup):")
            lines.append(format_vehicle_profile(prof))
        else:
            lines.append("Vehicle identification: VIN detected but lookup failed (offline or service unavailable).")
    else:
        parsed = parse_year_make_model(msg)
        if parsed:
            y, mk, md = parsed
            prof = carquery_search(y, mk, md, use_cache=True)
            if prof:
                lines.append("")
                lines.append("Vehicle trims/specs (cached lookup):")
                lines.append(format_vehicle_profile(prof))
            else:
                vp = vpci_models_for_make_year(mk, y, use_cache=True)
                if vp:
                    lines.append("")
                    lines.append("Vehicle catalog (cached lookup):")
                    lines.append(format_vehicle_profile(vp))
                    if md.lower() not in " ".join((vp.data.get("models") or [])).lower():
                        lines.append(f"- Note: '{md.title()}' not found in vPIC model list for {mk.title()} {y}.")
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

    # Lightweight topic routing
    if any(w in lower for w in ("fuel", "mpg", "l/100", "consumption", "mileage", "efficiency")):
        lines.append("")
        lines.append("Top efficiency levers (highest impact first):")
        lines.append("- Reduce cruising speed (aero drag grows rapidly with speed).")
        lines.append("- Smooth throttle; avoid unnecessary accelerations/braking.")
        lines.append("- Keep RPM moderate via appropriate gearing.")
        lines.append("- Tire pressure/alignment; avoid roof racks and excess mass.")
        lines.append("- Maintenance: filters, plugs, correct oil grade, brake drag.")

    if any(w in lower for w in ("rpm", "gear", "shift")):
        lines.append("")
        lines.append("Gearing/RPM guidance:")
        lines.append("- If RPM is high at low road speed: shift up (if lugging isn’t occurring).")
        lines.append("- Avoid full-throttle at very low RPM in high gear (knock/lug risk).")

    if any(w in lower for w in ("ev", "battery", "bms", "regen", "range")):
        lines.append("")
        lines.append("EV range guidance:")
        lines.append("- Speed discipline matters most on highways.")
        lines.append("- Precondition battery when possible; cold packs reduce efficiency.")
        lines.append("- Prefer gentle regen/coast; avoid brake-to-zero when you can glide.")
        lines.append("- Cabin HVAC can be a major load; use seat heaters when available.")

    if any(w in lower for w in ("turbo", "boost", "knock", "octane", "iat")):
        lines.append("")
        lines.append("Turbo/knock basics:")
        lines.append("- Knock risk rises with high load, high intake temps, and low octane.")
        lines.append("- Prioritize intercooling, conservative boost, and correct fuel.")
        lines.append("- Watch coolant/oil temps; heat soak can force timing pull.")

    if not any(w in lower for w in ("fuel", "mpg", "consumption", "efficiency", "rpm", "gear", "shift", "ev", "battery", "bms", "regen", "turbo", "boost", "knock", "octane", "iat")):
        lines.append("")
        lines.append("Tell me your vehicle and goal, and I’ll respond with a checklist.")
        lines.append("- Vehicle: year/make/model + engine (ICE/Hybrid/EV)")
        lines.append("- Use-case: city / highway / track / towing")
        lines.append("- Goal: efficiency / cooling / performance / reliability")

    lines.append("")
    lines.append("Tip: paste a VIN (17 chars) or 'YEAR MAKE MODEL' to fetch + cache vehicle info.")
    lines.append("System: For cloud answers, enable Gemini billing/quota and restart.")
    return "\n".join(lines)

# --- Configuration & Persona ---
SYSTEM_PROMPT = """
You are "engineer help", an advanced Automotive Engineering AI and Virtual Chief Engineer.
Your expertise includes:
1. High-performance telemetry analysis.
2. Internal Combustion Engine (ICE) mechanics and optimization.
3. EV battery management systems (BMS) and electric drivetrain efficiency.
4. Sustainability in automotive engineering, specifically aligning with UN SDGs 11 (Sustainable Cities and Communities) and 13 (Climate Action).

Tone: Professional, expert, data-driven, yet accessible. You provide technical insights and optimization strategies.
"""

INITIAL_MESSAGE = "Telemetry linked. I am engineer help, your Virtual Chief Engineer. How can we optimize your vehicle's setup and emissions today?"

# --- UI Setup ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")  # We will custom style the bubbles

class ChatBubble(ctk.CTkFrame):
    def __init__(self, master, message, sender="bot", **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        
        self.columnconfigure(0, weight=1)
        
        # Define colors and alignment
        if sender == "user":
            fg_color = "#3d3d3d"  # Muted grey
            text_color = "white"
            anchor = "e"
            padx = (50, 10)
        else:
            fg_color = "#004d4d"  # Dark cyan/teal base
            text_color = "#00ffff"  # Glowing cyan
            anchor = "w"
            padx = (10, 50)
            
        # Bubble frame
        bubble = ctk.CTkFrame(self, fg_color=fg_color, corner_radius=15)
        bubble.grid(row=0, column=0, sticky=anchor, padx=padx, pady=5)
        
        # Message label
        label = ctk.CTkLabel(
            bubble, 
            text=message, 
            wraplength=400, 
            justify="left",
            text_color=text_color,
            font=("Inter", 13)
        )
        label.pack(padx=15, pady=10)

class EngineerHelpBot(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("EcoDrive - Virtual Chief Engineer")
        self.geometry("500x700")
        self.configure(fg_color="#1a1a1a")  # Deep charcoal background

        # --- Chat History Area ---
        self.scroll_canvas = ctk.CTkScrollableFrame(
            self, 
            fg_color="#1a1a1a", 
            scrollbar_button_color="#333333",
            scrollbar_button_hover_color="#444444"
        )
        self.scroll_canvas.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        # --- Input Area ---
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.pack(fill="x", padx=10, pady=20)
        
        self.entry = ctk.CTkEntry(
            self.input_frame, 
            placeholder_text="Ask your Chief Engineer...",
            fg_color="#2b2b2b",
            border_color="#3d3d3d",
            height=45,
            font=("Inter", 13)
        )
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.entry.bind("<Return>", lambda e: self.send_message())

        self.send_button = ctk.CTkButton(
            self.input_frame, 
            text="Send", 
            command=self.send_message,
            fg_color="#006666",
            hover_color="#008080",
            width=80,
            height=45,
            font=("Inter", 13, "bold"),
            text_color="#00ffff"
        )
        self.send_button.pack(side="right")

        self.gemini_cooldown_until_ts = 0.0
        self.gemini_quota_blocked = False
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self._busy = False

        # Initialize Chat Session
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=SYSTEM_PROMPT
            )
            self.chat = self.model.start_chat(history=[])
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            self.chat = None

        # Add initial welcome message
        self.add_message(INITIAL_MESSAGE, "bot")

    def add_message(self, message, sender):
        bubble = ChatBubble(self.scroll_canvas, message, sender)
        bubble.pack(fill="x")
        # Auto-scroll to bottom
        # CustomTkinter's ScrollableFrame handles this somewhat, but we can nudge it
        self._after_id = self.after(10, lambda: self.scroll_canvas._parent_canvas.yview_moveto(1.0))

    def send_message(self):
        user_input = self.entry.get().strip()
        if not user_input:
            return

        if self._busy:
            return

        now = time.time()
        if self.gemini_quota_blocked:
            self.add_message(_offline_engineer_help(user_input), "bot")
            return
        if now < self.gemini_cooldown_until_ts:
            remaining = max(1, int(self.gemini_cooldown_until_ts - now))
            self.add_message(f"System: Cooling down (~{remaining}s). Local response below.\n\n{_offline_engineer_help(user_input)}", "bot")
            return

        # Clear entry and show user message
        self.entry.delete(0, tk.END)
        self.add_message(user_input, "user")

        # Disable input while waiting
        self._busy = True
        self.entry.configure(state="disabled")
        self.send_button.configure(state="disabled")
        self.add_message("System: Analyzing…", "bot")

        # Run LLM request in a separate thread to keep UI responsive
        threading.Thread(target=self.get_ai_response, args=(user_input,), daemon=True).start()

    def get_ai_response(self, user_input):
        try:
            if self.chat:
                response = self.chat.send_message(user_input)
                ai_text = response.text
            else:
                ai_text = _offline_engineer_help(user_input)
        except Exception as e:
            err_text = str(e)
            if _is_quota_429(err_text):
                retry_s = _extract_retry_seconds(err_text) or 30
                self.gemini_cooldown_until_ts = time.time() + retry_s
                if "limit: 0" in err_text.lower() or "free_tier_requests" in err_text.lower():
                    self.gemini_quota_blocked = True
                    ai_text = _offline_engineer_help(user_input)
                else:
                    ai_text = f"System: Rate limited (~{retry_s}s). Local response below.\n\n{_offline_engineer_help(user_input)}"
            else:
                ai_text = f"System: Cloud connection error. Local response below.\n\n{_offline_engineer_help(user_input)}"

        # Display response in main thread
        self.after(0, lambda: self.display_ai_response(ai_text))

    def display_ai_response(self, ai_text):
        self.add_message(ai_text, "bot")
        self._busy = False
        self.entry.configure(state="normal")
        self.send_button.configure(state="normal")
        self.entry.focus()

if __name__ == "__main__":
    app = EngineerHelpBot()
    app.mainloop()
