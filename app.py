import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import os

MODEL_FILE = 'ecodrive_model.pkl'

class EcoDriveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EcoDrive: Telemetry-Based Emission Optimizer")
        self.root.geometry("500x600")
        self.root.configure(bg="#2c3e50")
        self.root.resizable(False, False)
        
        self.model = None
        self.load_model()
        
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
        # Styling
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', background="#2c3e50", foreground="#ecf0f1", font=('Helvetica', 12))
        style.configure('TButton', font=('Helvetica', 12, 'bold'), background="#27ae60", foreground="white")
        style.map('TButton', background=[('active', '#2ecc71')])
        style.configure('TEntry', font=('Helvetica', 12))
        
        # Header Frame
        header_frame = tk.Frame(self.root, bg="#2980b9", pady=20)
        header_frame.pack(fill=tk.X)
        header_lbl = tk.Label(header_frame, text="EcoDrive Efficiency Predictor", font=('Helvetica', 16, 'bold'), bg="#2980b9", fg="white")
        header_lbl.pack()
        
        # Input Frame
        input_frame = tk.Frame(self.root, bg="#2c3e50", pady=20)
        input_frame.pack(pady=10)
        
        # Speed
        ttk.Label(input_frame, text="Speed (km/h):").grid(row=0, column=0, padx=10, pady=15, sticky="e")
        self.speed_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.speed_var, width=18).grid(row=0, column=1, padx=10, pady=15)
        
        # Engine RPM
        ttk.Label(input_frame, text="Engine RPM:").grid(row=1, column=0, padx=10, pady=15, sticky="e")
        self.rpm_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.rpm_var, width=18).grid(row=1, column=1, padx=10, pady=15)
        
        # Throttle Position
        ttk.Label(input_frame, text="Throttle % (0-100):").grid(row=2, column=0, padx=10, pady=15, sticky="e")
        self.throttle_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.throttle_var, width=18).grid(row=2, column=1, padx=10, pady=15)
        
        # Engine Load
        ttk.Label(input_frame, text="Engine Load % (0-100):").grid(row=3, column=0, padx=10, pady=15, sticky="e")
        self.load_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.load_var, width=18).grid(row=3, column=1, padx=10, pady=15)
        
        # Predict Button
        pred_btn = ttk.Button(self.root, text="Predict Efficiency", command=self.predict, padding=10)
        pred_btn.pack(pady=15)
        
        # Result Frame
        result_frame = tk.Frame(self.root, bg="#34495e", pady=15, padx=20, bd=2, relief=tk.GROOVE)
        result_frame.pack(fill=tk.X, padx=30, pady=10)
        
        # Result Label
        self.result_lbl = tk.Label(result_frame, text="Predicted Consumption: -- L/100km", font=('Helvetica', 14, 'bold'), bg="#34495e", fg="#ecf0f1")
        self.result_lbl.pack(pady=5)
        
        # Warning Label
        self.warning_lbl = tk.Label(result_frame, text="Awaiting Input...", font=('Helvetica', 12, 'bold'), bg="#34495e", fg="#bdc3c7")
        self.warning_lbl.pack(pady=5)
        
        # Recommendation Label
        self.rec_lbl = tk.Label(result_frame, text="", font=('Helvetica', 11, 'italic'), bg="#34495e", fg="#3498db", wraplength=400)
        self.rec_lbl.pack(pady=5)

    def predict(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded. Please ensure ecodrive_model.pkl is present.")
            # Attempt to reload just in case they just ran the train script
            self.load_model()
            if self.model is None:
                return
            
        try:
            speed = float(self.speed_var.get())
            rpm = float(self.rpm_var.get())
            throttle = float(self.throttle_var.get())
            load = float(self.load_var.get())
            
            # Prepare input data for prediction
            input_df = pd.DataFrame({
                'Speed': [speed],
                'Engine_RPM': [rpm],
                'Throttle_Position': [throttle],
                'Engine_Load': [load]
            })
            
            prediction = self.model.predict(input_df)[0]
            
            self.result_lbl.config(text=f"{prediction:.2f} L/100km")
            
            # Warning logic: if consumption > 40 L/100km, it's unsustainably high
            if prediction > 50.0:
                 self.warning_lbl.config(text="⚠️ WARNING: Unsustainably High Fuel Consumption!", fg="#e74c3c")
            elif prediction > 30.0:
                 self.warning_lbl.config(text="⚠️ Moderate Warning: High Fuel Consumption.", fg="#f39c12")
            else:
                 self.warning_lbl.config(text="✅ Optimal Driving Efficiency.", fg="#2ecc71")
                 
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
                rec_text = "Keep up the good work! Your driving is highly efficient."
            elif not recommendations:
                rec_text = "Consider smoother acceleration and braking to lower consumption."
            else:
                rec_text = "💡 Tip: " + " ".join(recommendations)
                
            self.rec_lbl.config(text=rec_text)
                 
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = EcoDriveApp(root)
    root.mainloop()
