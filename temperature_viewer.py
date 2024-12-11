"""
Temperature Data Visualization GUI
Displays temperature time series data with configurable time windows and resolution.
"""

import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

API_URL = "http://localhost:8000"

class TemperatureViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Temperature Data Viewer")
        self.root.geometry("1200x800")
        
        # Get initial data range from health check
        health_data = requests.get(f"{API_URL}/health").json()
        self.data_start = datetime.fromisoformat(health_data["time_range"]["start"])
        self.data_end = datetime.fromisoformat(health_data["time_range"]["end"])
        
        self.setup_gui()
        self.load_default_data()
        
    def setup_gui(self):
        # Control Frame
        control_frame = ttk.Frame(self.root, padding="5")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Time Range Selection
        ttk.Label(control_frame, text="Start Time:").pack(side=tk.LEFT, padx=5)
        self.start_time_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.start_time_var, width=20).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="End Time:").pack(side=tk.LEFT, padx=5)
        self.end_time_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.end_time_var, width=20).pack(side=tk.LEFT, padx=5)
        
        # Resolution Selection
        ttk.Label(control_frame, text="Resolution (seconds):").pack(side=tk.LEFT, padx=5)
        self.resolution_var = tk.StringVar(value="60")
        resolutions = ["1", "60", "300", "600", "1800", "3600"]
        resolution_combo = ttk.Combobox(control_frame, textvariable=self.resolution_var, values=resolutions, width=10)
        resolution_combo.pack(side=tk.LEFT, padx=5)
        
        # Update Button
        ttk.Button(control_frame, text="Update", command=self.update_data).pack(side=tk.LEFT, padx=5)
        
        # Browser Frame (where we'll display the HTML)
        self.browser_frame = ttk.Frame(self.root)
        self.browser_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def load_default_data(self):
        # Set default time range to latest hour
        end_time = self.data_end
        start_time = end_time - timedelta(hours=1)
        
        self.start_time_var.set(start_time.isoformat())
        self.end_time_var.set(end_time.isoformat())
        
        self.update_data()
        
    def update_data(self):
        try:
            # Get data from API
            params = {
                "start_time": self.start_time_var.get(),
                "end_time": self.end_time_var.get(),
                "resolution": self.resolution_var.get()
            }
            response = requests.get(f"{API_URL}/aggregate", params=params)
            data = response.json()
            
            # Create the plot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Ambient Temperature', 'Device Temperature'),
                vertical_spacing=0.1,
                shared_xaxes=True
            )
            
            # Get timestamps for x-axis
            timestamps = [point["timestamp"] for point in data["aggregated_data"]]
            
            # Plot Ambient Temperature
            ambient_data = [point["ambient_temperature"] for point in data["aggregated_data"]]
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[d["mean"] for d in ambient_data],
                    name="Ambient Mean",
                    line=dict(color='rgb(31, 119, 180)', width=2)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[d["max"] for d in ambient_data],
                    name="Ambient Max",
                    line=dict(color='rgba(31, 119, 180, 0.3)', width=1),
                    fill=None
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[d["min"] for d in ambient_data],
                    name="Ambient Min",
                    line=dict(color='rgba(31, 119, 180, 0.3)', width=1),
                    fill='tonexty'
                ),
                row=1, col=1
            )
            
            # Plot Device Temperature
            device_data = [point["device_temperature"] for point in data["aggregated_data"]]
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[d["mean"] for d in device_data],
                    name="Device Mean",
                    line=dict(color='rgb(255, 127, 14)', width=2)
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[d["max"] for d in device_data],
                    name="Device Max",
                    line=dict(color='rgba(255, 127, 14, 0.3)', width=1),
                    fill=None
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[d["min"] for d in device_data],
                    name="Device Min",
                    line=dict(color='rgba(255, 127, 14, 0.3)', width=1),
                    fill='tonexty'
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=700,
                showlegend=True,
                title_text=f"Temperature Data ({params['resolution']}s resolution)",
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text="Time")
            fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
            fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
            
            # Save to HTML and display
            html_file = "temp_plot.html"
            fig.write_html(html_file)
            
            # Create a browser widget and load the HTML
            import webbrowser
            webbrowser.open(html_file)
            
        except Exception as e:
            print(f"Error updating data: {e}")
            
if __name__ == "__main__":
    root = tk.Tk()
    app = TemperatureViewer(root)
    root.mainloop()