import customtkinter as ctk
import joblib
import pandas as pd
import os
from tkinter import messagebox

# Set appearance and theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

MODEL_FILE = 'ecodrive_model.pkl'

class EcoDriveApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("EcoDrive: Telemetry-Based Emission Optimizer")
        self.geometry("600x750")
        self.resizable(False, False)

        self.model = None
        self.load_model()

        # UI Components
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
        # Main Container
        self.main_frame = ctk.CTkFrame(self, corner_radius=15)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Header
        self.header_label = ctk.CTkLabel(
            self.main_frame, 
            text="🌱 EcoDrive Optimizer", 
            font=ctk.CTkFont(size=26, weight="bold")
        )
        self.header_label.pack(pady=(30, 10))

        self.sub_header = ctk.CTkLabel(
            self.main_frame, 
            text="Real-time fuel efficiency analytics", 
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.sub_header.pack(pady=(0, 20))

        # Input Section
        self.input_container = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.input_container.pack(pady=10, padx=40, fill="x")

        # Speed
        self.speed_label = ctk.CTkLabel(self.input_container, text="Speed (km/h)", font=ctk.CTkFont(size=13, weight="bold"))
        self.speed_label.pack(anchor="w", padx=5)
        self.speed_entry = ctk.CTkEntry(self.input_container, placeholder_text="e.g. 80", height=40)
        self.speed_entry.pack(fill="x", pady=(0, 15))

        # Engine RPM
        self.rpm_label = ctk.CTkLabel(self.input_container, text="Engine RPM", font=ctk.CTkFont(size=13, weight="bold"))
        self.rpm_label.pack(anchor="w", padx=5)
        self.rpm_entry = ctk.CTkEntry(self.input_container, placeholder_text="e.g. 2500", height=40)
        self.rpm_entry.pack(fill="x", pady=(0, 15))

        # Throttle %
        self.throttle_label = ctk.CTkLabel(self.input_container, text="Throttle % (0-100)", font=ctk.CTkFont(size=13, weight="bold"))
        self.throttle_label.pack(anchor="w", padx=5)
        self.throttle_entry = ctk.CTkEntry(self.input_container, placeholder_text="e.g. 30", height=40)
        self.throttle_entry.pack(fill="x", pady=(0, 15))

        # Engine Load %
        self.load_label = ctk.CTkLabel(self.input_container, text="Engine Load % (0-100)", font=ctk.CTkFont(size=13, weight="bold"))
        self.load_label.pack(anchor="w", padx=5)
        self.load_entry = ctk.CTkEntry(self.input_container, placeholder_text="e.g. 45", height=40)
        self.load_entry.pack(fill="x", pady=(0, 25))

        # Predict Button
        self.predict_btn = ctk.CTkButton(
            self.main_frame, 
            text="Analyze Efficiency", 
            command=self.predict,
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50,
            corner_radius=10
        )
        self.predict_btn.pack(pady=10, padx=40, fill="x")

        # Result Dashboard
        self.result_card = ctk.CTkFrame(self.main_frame, fg_color="#2b2b2b", corner_radius=12)
        self.result_card.pack(pady=20, padx=30, fill="x")

        self.result_value = ctk.CTkLabel(
            self.result_card, 
            text="-- L/100km", 
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#2ecc71"
        )
        self.result_value.pack(pady=(15, 5))

        self.status_label = ctk.CTkLabel(
            self.result_card, 
            text="Awaiting Telemetry Data...", 
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="gray"
        )
        self.status_label.pack(pady=5)

        self.tip_label = ctk.CTkLabel(
            self.result_card, 
            text="", 
            font=ctk.CTkFont(size=12, slant="italic"),
            wraplength=350,
            text_color="#3498db"
        )
        self.tip_label.pack(pady=(5, 15))

    def predict(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded. Please ensure ecodrive_model.pkl is present.")
            self.load_model()
            if self.model is None: return

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
            self.result_value.configure(text=f"{prediction:.2f} L/100km")

            # Status and Color Logic
            if prediction > 50.0:
                self.status_label.configure(text="⚠️ CRITICALLY HIGH CONSUMPTION", text_color="#e74c3c")
                self.result_value.configure(text_color="#e74c3c")
            elif prediction > 30.0:
                self.status_label.configure(text="🔶 MODERATE CONSUMPTION", text_color="#e67e22")
                self.result_value.configure(text_color="#e67e22")
            else:
                self.status_label.configure(text="✅ OPTIMAL EFFICIENCY", text_color="#2ecc71")
                self.result_value.configure(text_color="#2ecc71")

            # Recommendations
            recs = []
            if rpm > 4000 and speed < 100: recs.append("Upshift to lower RPM.")
            if throttle > 70: recs.append("Reduce throttle input.")
            if load > 80: recs.append("High engine strain detected.")
            if speed > 120: recs.append("High aerodynamic drag.")

            if not recs:
                rec_text = "Sustainable driving pattern maintained."
            else:
                rec_text = "💡 ECO-TIP: " + " | ".join(recs)

            self.tip_label.configure(text=rec_text)

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")

if __name__ == "__main__":
    app = EcoDriveApp()
    app.mainloop()

