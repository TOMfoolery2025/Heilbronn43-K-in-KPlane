import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import threading
import time
import sys
import os

# Robust path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Add project root to sys.path so we can do 'from src.solver import ...'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add current dir to sys.path so we can do 'import solver' if needed
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from src.solver import SimulatedAnnealingSolver
    from src.LCNv1.api import LCNSolver, StrategyType
except ImportError:
    try:
        from solver import SimulatedAnnealingSolver
        from LCNv1.api import LCNSolver, StrategyType
    except ImportError as e:
        print(f"Import Error: {e}")
        # If running from src directly, LCNv1 might be tricky
        sys.path.append(os.path.join(current_dir, 'LCNv1'))
        from solver import SimulatedAnnealingSolver
        from api import LCNSolver, StrategyType

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("K-Planar Graph Minimizer")
        self.geometry("1200x800")
        self.closing = False # Flag to track closing state
        
        # Data
        self.nodes = []
        self.edges = []
        self.width_bounds = 10
        self.height_bounds = 10
        self.solver = None
        self.running_optimizer = False
        self.highlight_node_id = None
        
        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.btn_load = ctk.CTkButton(self.sidebar, text="Load JSON", command=self.load_json)
        self.btn_load.pack(pady=20, padx=20)
        
        self.lbl_stats = ctk.CTkLabel(self.sidebar, text="Stats:\nK: -\nTotal: -", justify="left")
        self.lbl_stats.pack(pady=20, padx=20)
        
        # Filter Section
        self.lbl_filter = ctk.CTkLabel(self.sidebar, text="Find Node ID:")
        self.lbl_filter.pack(pady=(20, 5), padx=20)
        self.entry_filter = ctk.CTkEntry(self.sidebar, placeholder_text="ID")
        self.entry_filter.pack(pady=5, padx=20)
        self.entry_filter.bind("<Return>", self.search_node)
        self.btn_search = ctk.CTkButton(self.sidebar, text="Search", command=self.search_node)
        self.btn_search.pack(pady=5, padx=20)
        
        # Strategy Selection
        self.strategy_label = ctk.CTkLabel(self.sidebar, text="Solver Strategy")
        self.strategy_label.pack(pady=(10, 0))
        
        self.strategy_var = ctk.StringVar(value="legacy")
        self.strategy_menu = ctk.CTkOptionMenu(
            self.sidebar,
            variable=self.strategy_var,
            values=["legacy", "new", "numba", "cuda"]
        )
        self.strategy_menu.pack(pady=(0, 10))
        
        self.btn_optimize = ctk.CTkButton(self.sidebar, text="Run Optimizer", command=self.toggle_optimizer)
        self.btn_optimize.pack(pady=20, padx=20)
        
        # Main Area
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Matplotlib
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Toolbar (Zoom/Pan)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.main_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Interaction
        self.drag_node_idx = None
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def load_json(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if not file_path:
            return
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        self.nodes = data['nodes']
        self.edges = data['edges']
        self.width_bounds = data.get('width', 1000000)
        self.height_bounds = data.get('height', 1000000)
        self.highlight_node_id = None
        
        self.update_plot()

    def search_node(self, event=None):
        try:
            nid = int(self.entry_filter.get())
            # Verify node exists
            found = False
            for n in self.nodes:
                if n['id'] == nid:
                    found = True
                    break
            if found:
                self.highlight_node_id = nid
                self.update_plot()
            else:
                messagebox.showerror("Error", f"Node {nid} not found.")
        except ValueError:
            pass

    def update_plot(self, compute_stats=True):
        if self.closing or not self.winfo_exists(): return
        
        # Save current view limits to preserve zoom/pan state
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        self.ax.clear()
        
        # Restore limits or set dynamic limits
        if xlim == (0.0, 1.0) and ylim == (0.0, 1.0):
             # Dynamic scaling based on content
             if self.nodes:
                 xs = [n['x'] for n in self.nodes]
                 ys = [n['y'] for n in self.nodes]
                 min_x, max_x = min(xs), max(xs)
                 min_y, max_y = min(ys), max(ys)
                 
                 # Add 10% padding
                 w = max_x - min_x
                 h = max_y - min_y
                 pad_x = max(w * 0.1, 50)
                 pad_y = max(h * 0.1, 50)
                 
                 self.ax.set_xlim(min_x - pad_x, max_x + pad_x)
                 self.ax.set_ylim(min_y - pad_y, max_y + pad_y)
             else:
                 self.ax.set_xlim(0, self.width_bounds)
                 self.ax.set_ylim(0, self.height_bounds)
        else:
             self.ax.set_xlim(xlim)
             self.ax.set_ylim(ylim)
             
        self.ax.set_aspect('equal')
        
        if not self.nodes:
            self.canvas.draw()
            return
            
        # Prepare arrays for scorer
        nodes_x = np.array([n['x'] for n in self.nodes], dtype=np.float64)
        nodes_y = np.array([n['y'] for n in self.nodes], dtype=np.float64)
        edges_source = np.array([e['source'] for e in self.edges], dtype=np.int32)
        edges_target = np.array([e['target'] for e in self.edges], dtype=np.int32)
        
        edge_crossings = None
        k = 0
        total = 0
        
        if hasattr(self, 'latest_edge_crossings') and self.running_optimizer:
             edge_crossings = self.latest_edge_crossings
             k = self.latest_k
             total = self.latest_total
        elif compute_stats:
             try:
                 from src.scorer import count_crossings
                 edge_crossings, k, total = count_crossings(nodes_x, nodes_y, edges_source, edges_target)
             except ImportError:
                 from scorer import count_crossings
                 edge_crossings, k, total = count_crossings(nodes_x, nodes_y, edges_source, edges_target)
        
        # Update Stats
        try:
            self.lbl_stats.configure(text=f"Stats:\nK: {k}\nTotal: {total}")
        except Exception:
            pass # Widget might be destroyed
        
        # Draw Edges
        if edge_crossings is not None:
            for i, (u, v) in enumerate(zip(edges_source, edges_target)):
                x1, y1 = nodes_x[u], nodes_y[u]
                x2, y2 = nodes_x[v], nodes_y[v]
                
                is_red = edge_crossings[i] > 3
                color = 'red' if is_red else 'gray'
                width = 2 if is_red else 1
                alpha = 0.8 if is_red else 0.5
                
                self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=alpha, zorder=1)
        else:
            for i, (u, v) in enumerate(zip(edges_source, edges_target)):
                x1, y1 = nodes_x[u], nodes_y[u]
                x2, y2 = nodes_x[v], nodes_y[v]
                self.ax.plot([x1, x2], [y1, y2], color='gray', linewidth=1, alpha=0.5, zorder=1)

        # Draw Nodes
        self.ax.scatter(nodes_x, nodes_y, c='blue', s=100, zorder=2, picker=True)
        
        # Highlight Node
        if self.highlight_node_id is not None:
            for n in self.nodes:
                if n['id'] == self.highlight_node_id:
                    self.ax.scatter([n['x']], [n['y']], c='yellow', s=300, zorder=1.5, edgecolors='black')
                    break
        
        # Node Labels
        for i, n in enumerate(self.nodes):
            self.ax.text(n['x'], n['y'], str(n['id']), color='white', ha='center', va='center', zorder=3, fontsize=8, fontweight='bold')
            
        self.canvas.draw()

    def on_press(self, event):
        if event.inaxes != self.ax: return
        if self.toolbar.mode != "": return
        if self.running_optimizer: return
        
        min_dist = float('inf')
        closest_idx = -1
        
        for i, n in enumerate(self.nodes):
            dist = (n['x'] - event.xdata)**2 + (n['y'] - event.ydata)**2
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        view_width = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
        threshold = (view_width * 0.02) ** 2
        
        if min_dist < threshold: 
            self.drag_node_idx = closest_idx

    def on_motion(self, event):
        if self.drag_node_idx is not None and event.inaxes == self.ax:
            self.nodes[self.drag_node_idx]['x'] = event.xdata
            self.nodes[self.drag_node_idx]['y'] = event.ydata
            self.update_plot(compute_stats=True)

    def on_release(self, event):
        self.drag_node_idx = None

    def toggle_optimizer(self):
        if self.running_optimizer:
            self.running_optimizer = False
            self.btn_optimize.configure(text="Run Optimizer")
        else:
            if not self.nodes: return
            self.running_optimizer = True
            self.btn_optimize.configure(text="Stop Optimizer")
            threading.Thread(target=self.run_optimizer_loop, daemon=True).start()

    def run_optimizer_loop(self):
        # Determine Strategy
        strategy = self.strategy_var.get()
        print(f"Starting optimizer with strategy: {strategy}")
        
        # Save current state to temp file for LCNSolver
        temp_file = "temp_optimizer_input.json"
        data = {
            "nodes": self.nodes,
            "edges": self.edges,
            "width": self.width_bounds,
            "height": self.height_bounds
        }
        with open(temp_file, 'w') as f:
            json.dump(data, f)
            
        # Initialize Solver
        try:
            self.lcn_solver = LCNSolver(strategy=strategy)
            self.lcn_solver.load_from_json(temp_file)
        except Exception as e:
            print(f"Error initializing solver: {e}")
            self.running_optimizer = False
            if self.winfo_exists():
                self.btn_optimize.configure(text="Run Optimizer")
            return

        # Initial temperature scaling
        initial_temp = max(self.width_bounds, self.height_bounds) * 0.2
        current_temp = initial_temp
        
        batch_size = 300 # Run 300 iterations between updates
        
        # Cooling rate per STEP (not per batch)
        cooling_rate = 0.9995 
        
        while self.running_optimizer and not self.closing:
            # Run a batch of iterations
            try:
                result = self.lcn_solver.optimize(
                    iterations=batch_size,
                    initial_temp=current_temp,
                    cooling_rate=cooling_rate
                )
                
                # Update App State
                if result.nodes:
                    result_nodes_map = {n['id']: n for n in result.nodes}
                    for node in self.nodes:
                        nid = node['id']
                        if nid in result_nodes_map:
                            node['x'] = result_nodes_map[nid]['x']
                            node['y'] = result_nodes_map[nid]['y']
                
                # Update Stats
                self.latest_k = result.k
                self.latest_total = result.total_crossings
                self.latest_edge_crossings = result.edge_crossings # Get from result
                
                # Update Temperature
                current_temp *= (cooling_rate ** batch_size)
                
                # Reheat check
                if current_temp < 0.1:
                     current_temp = initial_temp * 0.5 # Reheat
                     
                # Update UI
                if not self.closing and self.winfo_exists():
                    try:
                        self.after_id = self.after(0, lambda: self.update_plot(compute_stats=False))
                    except Exception:
                        break
                
            except Exception as e:
                print(f"Optimization error: {e}")
                break
                
            time.sleep(0.01) # Yield to UI thread
            
        self.running_optimizer = False
        if not self.closing and self.winfo_exists():
            self.btn_optimize.configure(text="Run Optimizer")

    def on_closing(self):
        self.closing = True
        self.running_optimizer = False
        if hasattr(self, 'after_id'):
            try:
                self.after_cancel(self.after_id)
            except ValueError:
                pass
        self.quit()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
