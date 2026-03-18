import customtkinter as ctk
import joblib
import pandas as pd
import os
from tkinter import messagebox
import random
import time
import re
import google.generativeai as genai
from dotenv import load_dotenv

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
        
        # Initialize Gemini Model
        self.gemini_connected = False
        self.gemini_init_error = None
        if GEMINI_API_KEY:
            try:
                # Switching to gemini-1.5-flash which is confirmed available
                self.gemini_model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    system_instruction=(
                        "You are the AutoBot Professional Engineering Consultant, a senior automotive engineer and industry historian. "
                        "Your goal is to provide profound, technically accurate insights into vehicle engineering, aerodynamics, "
                        "thermodynamics, and mechanical design. Maintain a formal, authoritative, yet accessible professional tone. "
                        "Prioritize data on performance metrics, platform architectures, and engineering innovation. "
                        "If a query is non-automotive, professionally redirect the user to automotive engineering topics."
                    )
                )
                self.gemini_connected = True
                print("Gemini API Initialized with gemini-1.5-flash.")
            except Exception as e:
                print(f"Gemini Init Error: {e}")
                # Try without system instruction as a final fallback
                try:
                    self.gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
                    self.gemini_connected = True
                    print("Gemini API Initialized (Legacy Mode).")
                except Exception as e2:
                    self.gemini_init_error = str(e2)
                    print(f"Gemini Critical Error: {e2}")
                    self.gemini_connected = False
        else:
            self.gemini_init_error = "API Key not found."
            print("No Gemini API Key found in environment or .env file.")
        
        # Car model database for deep knowledge (Professional Local Fallback)
        self.model_db = {
            "fortuner": {
                "name": "Toyota Fortuner (AN150/AN160)",
                "era": "modern",
                "desc": (
                    "A project of the IMV (Innovative International Multi-purpose Vehicle) platform. "
                    "Features a robust body-on-frame chassis, double-wishbone front suspension, and a 4-link rear with coil springs. "
                    "Renowned for high torsional rigidity and exceptional approach/departure angles for off-road traversal."
                ),
                "fact": "Engineering Note: Uses a part-time 4WD system with a high/low range transfer case for maximum torque multiplication."
            },
            "mustang": {
                "name": "Ford Mustang (Generation I)",
                "era": "past",
                "desc": (
                    "The progenitor of the Pony Car segment, introduced in 1964. The 1967 GT500 variant featured a "
                    "428-cubic-inch V8 with dual Holley carburetors, producing approximately 355 brake horsepower. "
                    "Utilized a unibody construction with an independent front suspension layout."
                ),
                "fact": "Historical Note: The 1965 GT350R was a purpose-built competition variant with a modified K-code 289 V8."
            },
            "tesla": {
                "name": "Tesla Model S Plaid",
                "era": "present",
                "desc": (
                    "A tri-motor AWD platform utilizing carbon-sleeved rotors to maintain structural integrity at high RPM (approx 20,000 RPM). "
                    "Boasts a drag coefficient (Cd) of just 0.208, among the lowest in production vehicles. Features a structural battery pack design."
                ),
                "fact": "Technical Note: The Plaid powertrain can sustain over 1,000 hp across the entire power band through advanced thermal management."
            },
            "supra": {
                "name": "Toyota Supra (A80)",
                "era": "past",
                "desc": (
                    "Famous for the 2JZ-GTE inline-6 engine, featuring a cast-iron block and sequential twin-turbochargers. "
                    "Designed using CAD/CAM systems that were cutting-edge in the early 90s, focusing on high-speed stability and massive braking performance."
                ),
                "fact": "Performance Note: The A80's hollow-fiber spoiler and magnesium-alloy steering wheel were key weight-reduction innovations."
            },
            "beetle": {
                "name": "Volkswagen Type 1 (Beetle)",
                "era": "past",
                "desc": (
                    "An air-cooled, rear-engine, rear-wheel-drive layout on a backbone chassis. "
                    "Engineered for simplicity and durability, featuring a torsion bar suspension system that provided a 'fully independent' setup on a budget."
                ),
                "fact": "Design Note: Its silhouette was aerodynamically advanced for 1938, achieving a Cd of approx 0.48."
            },
            "porsche 911": {
                "name": "Porsche 911 (992 Generation)",
                "era": "present",
                "desc": (
                    "The pinnacle of rear-engine evolution. The 992 chassis employs a high percentage of aluminum (approx 70%). "
                    "Integrated with PASM (Porsche Active Suspension Management) and optional PDCC (Active Roll Stabilization) for superior lateral dynamics."
                ),
                "fact": "Engineering Note: The 911's distinctive engine mounting has been moved forward across generations to optimize polar moment of inertia."
            }
        }
        
        self.create_widgets()
        
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
        self.tabview.add("AutoBot Professional Engineering Suite")
        
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
        parent = self.tabview.tab("AutoBot Professional Engineering Suite")
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)

        # Status Display
        self.bot_status_label = ctk.CTkLabel(parent, text="🤖 AutoBot: Senior Engineering Intelligence Online", font=ctk.CTkFont(size=13, weight="bold"), text_color="#00adb5")
        self.bot_status_label.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="w")

        self.chat_frame = ctk.CTkFrame(parent, corner_radius=15, fg_color="#1b1b1b", border_width=1, border_color="#393e46")
        self.chat_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.chat_frame.grid_rowconfigure(0, weight=1)
        self.chat_frame.grid_columnconfigure(0, weight=1)

        self.chat_history = ctk.CTkTextbox(self.chat_frame, font=ctk.CTkFont(size=15), text_color="#eeeeee", fg_color="#0a0a0a", border_color="#393e46")
        self.chat_history.grid(row=0, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")
        
        init_msg = "🤖 AutoBot: Greetings. I am your Lead Automotive Engineering Consultant.\n"
        if self.gemini_connected:
            init_msg += "⚡ Status: Gemini GenAI Online (Professional Intelligence Mode)\n"
        else:
            init_msg += f"⚡ Status: Local Engineering DB Only ({self.gemini_init_error if self.gemini_init_error else 'Offline'})\n"
            
        init_msg += "\nPlease provide a technical query regarding chassis dynamics, engine thermodynamics, or historical platform architectures.\n\n"
        
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
            self.chat_history.insert("1.0", "🤖 AutoBot: Memory banks cleared. Ready for new input.\n\n")
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
        
        # Simulate thinking
        self.bot_status_label.configure(text="🤖 AutoBot: Scanning Universal Vehicle Database...")
        self.after(random.randint(800, 1500), lambda: self.process_response(user_msg))

    def process_response(self, user_msg):
        # 1. Try Gemini First if connected
        if self.gemini_connected:
            try:
                # Adding some safety settings or just a generic call
                response = self.gemini_model.generate_content(user_msg)
                if response and response.text:
                    bot_reply = response.text
                    self.update_chat_history(f"🤖 AutoBot (GenAI): {bot_reply}\n\n")
                else:
                    raise Exception("Empty response or blocked by safety filters.")
            except Exception as e:
                print(f"Gemini Generation Error: {e}")
                # Fallback to local
                bot_reply = self.get_universal_response(user_msg.lower())
                self.update_chat_history(f"🤖 AutoBot (Offline): {bot_reply}\n\n")
        else:
            bot_reply = self.get_universal_response(user_msg.lower())
            self.update_chat_history(f"🤖 AutoBot: {bot_reply}\n\n")
            
        self.bot_status_label.configure(text="🤖 AutoBot: Ready for Universal Expansion")

    def update_chat_history(self, message):
        self.chat_history.configure(state="normal")
        self.chat_history.insert("end", message)
        self.chat_history.see("end")
        self.chat_history.configure(state="disabled")

    def get_universal_response(self, message):
        # 1. Try Specific Model Match First (Fuzzy-ish)
        for key, data in self.model_db.items():
            # Check if key is in message (handles "tell me about fortuner")
            # Improved pattern matching for common car names
            key_pattern = key.replace(" ", ".*")
            if re.search(key_pattern, message) or (key == "fortuner" and ("forturner" in message or "fortuner" in message)):
                return f"Ah, the {data['name']}! {data['desc']}\n\n💡 Fun Fact: {data['fact']}"

        knowledge = {
            "past": [
                "In 1886, Karl Benz patented the Motorwagen, the first true automobile. It had 3 wheels and a massive 0.75 horsepower engine!",
                "The 1920s saw the rise of the Duesenberg Model J, a car so powerful and expensive that it remainded a status symbol for a century.",
                "Steam-powered buses were actually quite common in London during the early 1800s before the Red Flag Act slowed them down.",
                "The 1960s was the golden age of muscle cars. The 1969 Dodge Charger Daytona was the first car to break 200 mph in NASCAR history!"
            ],
            "present": [
                "Carbon fiber and active aerodynamics allow modern hypercars like the Rimac Nevera to accelerate from 0-60 mph in under 2 seconds.",
                "Today's Formula 1 cars use hybrid power units that are over 50% thermally efficient—the highest of any internal combustion engine ever.",
                "Autonomous Level 3 systems are now appearing on public roads, allowing drivers to take their hands off the wheel in specific conditions.",
                "Modern trucks are switching to electric and hydrogen fuel cells to reduce the massive carbon footprint of logistics."
            ],
            "future": [
                "By 2040, solid-state batteries are expected to provide 1000km range with 5-minute charge times, effectively ending range anxiety.",
                "Autonomous Flying Taxis (eVTOL) are undergoing testing in Dubai and NYC, aiming to move urban transport into the third dimension by 2030.",
                "In the 2050s, vehicle-to-everything (V2X) communication will likely eliminate traffic jams entirely through perfect AI coordination.",
                "Synthetic fuels (e-fuels) might allow classic internal combustion engines to run with net-zero emissions in the far future."
            ],
            "bike": [
                "Motorcycles are the most efficient form of ICE transport. The Kawasaki Ninja H2R is currently the fastest production bike, reaching 400 km/h.",
                "Electric bikes (E-Bikes) are the fastest-growing vehicle segment, offering a sustainable future for urban commuting."
            ],
            "truck": [
                "The Tesla Semi aims to revolutionize long-haul trucking with a drag coefficient better than a Bugatti Chiron!",
                "Hydrogen trucks like the Nikola Tre offer the fast refueling classic diesel drivers are used to, but with zero emissions."
            ],
            "formula": ["Formula 1 is the pinnacle of engineering. A modern F1 car generates so much downforce it could theoretically drive upside down in a tunnel!"],
            "suv": ["SUVs now dominate over 50% of the global car market, leading manufacturers to focus intensely on making these heavy vehicles more aerodynamic."],
            "luxury": ["Rolls-Royce Spectre is the brand's first fully electric car, proving that silent electric power is the ultimate luxury for a refined ride."]
        }

        # Context-aware eras
        if any(w in message for w in ["old", "past", "history", "classic", "vintage", "antique"]):
            return random.choice(knowledge["past"])
        if any(w in message for w in ["now", "present", "current", "modern", "today"]):
            return random.choice(knowledge["present"])
        if any(w in message for w in ["future", "2050", "tomorrow", "concept", "prediction"]):
            return random.choice(knowledge["future"])
            
        # Specific Categories
        for key in knowledge:
            if key in message:
                return random.choice(knowledge[key])
        
        # 3. Conversational Fallback
        return "I'm always expanding my memory banks! I don't have deep specs on that specific model yet, but I do know a lot about general vehicle history, future concepts, and efficiency optimization. \n\nTry asking me about 'Classic 60s cars', 'Solid state batteries', or a popular model like 'Mustang' or 'Tesla'!"

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
