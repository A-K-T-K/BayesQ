import logging
logger = logging.getLogger("qbn")
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import itertools
import numpy as np
import logging
import time
import psutil
from typing import Dict, List, Tuple, Optional, Any
from functools import wraps
import configparser
import os
from datetime import datetime

from qiskit.visualization import plot_histogram
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#--- Matplotlib Imports ---
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import matplotlib.pyplot as plt
except ImportError:
    logger.debug("Matplotlib not found. Please install: pip install matplotlib")
    exit()

# --- NetworkX Imports ---
try:
    import networkx as nx
except ImportError:
    logger.debug("NetworkX not found. Please install: pip install networkx")
    exit()

# --- Qiskit Imports ---
try:
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit_machine_learning.algorithms import QBayesian
    from qiskit.visualization import plot_histogram, circuit_drawer
    from qiskit.primitives import BackendSampler
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (NoiseModel, depolarizing_error, 
                                  thermal_relaxation_error, phase_damping_error,
                                  amplitude_damping_error)
except ImportError:
    logger.debug("Qiskit not found. Please install: pip install qiskit qiskit-aer qiskit-machine-learning")
    exit()


# ============================================================================
# UI HELPERS: TOOLTIPS
# ============================================================================

class CreateToolTip(object):
    """
    Create a tooltip for a given widget.
    """
    def __init__(self, widget, text='widget info'):
        self.waittime = 500     # miliseconds
        self.wraplength = 250   # pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = tk.Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background="#ffffe0", relief='solid', borderwidth=1,
                       wraplength = self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw= None
        if tw:
            tw.destroy()

# ============================================================================
# LOGGING AND RESOURCE MONITORING
# ============================================================================

class TextWidgetHandler(logging.Handler):
    """Custom logging handler that writes to a Tkinter Text widget."""
    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))

    def emit(self, record):
        msg = self.format(record)
        def append():
            try:
                self.text_widget.configure(state='normal')
                self.text_widget.insert(tk.END, msg + '\n')
                self.text_widget.configure(state='disabled')
                self.text_widget.see(tk.END)
            except: 
                pass
        self.text_widget.after(0, append)


class ResourceMonitor:
    """Enhanced resource monitoring with real-time updates."""
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.resources = []
        self.start_time = None

    def start_monitoring(self):
        self.monitoring = True
        self.resources = []
        self.start_time = time.time()
        self.record()

    def stop_monitoring(self):
        self.monitoring = False
        self.record()

    def get_snapshot(self):
        mem_info = self.process.memory_info()
        return {
            'timestamp': time.time(), 
            'cpu_percent': self.process.cpu_percent(interval=0),
            'memory_mb': mem_info.rss / 1024 / 1024, 
            'threads': self.process.num_threads()
        }

    def record(self):
        self.resources.append(self.get_snapshot())

    def get_summary(self):
        if len(self.resources) < 2: 
            return "Insufficient monitoring data"
        
        elapsed = self.resources[-1]['timestamp'] - self.resources[0]['timestamp']
        mem_vals = [r['memory_mb'] for r in self.resources]
        cpu_vals = [r['cpu_percent'] for r in self.resources if r['cpu_percent'] > 0]
        
        return (f"=== RESOURCE SUMMARY ===\n"
                f"Time: {elapsed:.3f}s | Peak Mem: {max(mem_vals):.2f} MB | Avg Mem: {sum(mem_vals)/len(mem_vals):.2f} MB\n"
                f"Peak CPU: {max(cpu_vals) if cpu_vals else 0:.1f}% | Avg CPU: {sum(cpu_vals)/len(cpu_vals) if cpu_vals else 0:.1f}% | Threads: {self.resources[-1]['threads']}")


# ============================================================================
# UTILITY CLASSES
# ============================================================================

class ConfigManager:
    """Manages application configuration settings."""

    def __init__(self, config_file: str = 'qbn_config.ini'):
        self.config = configparser.ConfigParser()
        self.config_file = config_file
        self.load_config()

    def load_config(self) -> None:
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
        else:
            self.create_default_config()

    def create_default_config(self) -> None:
        self.config['DEFAULT'] = {
            'shots': '1024',
            'noise_model': 'None (Ideal)',
            'max_network_size': '20',
            'max_parents': '5',
            'auto_save': 'True',
            'log_level': 'INFO'
        }
        self.config['PERFORMANCE'] = {
            'enable_profiling': 'False'
        }
        self.save_config()

    def save_config(self) -> None:
        with open(self.config_file, 'w') as f:
            self.config.write(f)

    def get(self, section: str, key: str, fallback: Any = None) -> str:
        return self.config.get(section, key, fallback=fallback)

    def set(self, section: str, key: str, value: str) -> None:
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, key, value)
        self.save_config()


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f'qbn_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger('QBN')
    logger.info(f"Logging initialized at {log_level} level")
    return logger


class PerformanceMonitor:
    def __init__(self):
        self.metrics: List = []
        self.logger = logging.getLogger('QBN.Performance')

    @staticmethod
    def benchmark(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            mem_after = process.memory_info().rss / 1024 / 1024
            execution_time = end_time - start_time
            memory_used = mem_after - mem_before

            logger = logging.getLogger('QBN.Performance')
            logger.info(f"{func.__name__}: {execution_time:.4f}s, Memory: {memory_used:.2f} MB")

            return result
        return wrapper


class NetworkValidator:
    def __init__(self, max_nodes: int = 20, max_parents: int = 5):
        self.max_nodes = max_nodes
        self.max_parents = max_parents
        self.logger = logging.getLogger('QBN.Validator')

    def validate_network_structure(self, graph: nx.DiGraph) -> Tuple[bool, str]:
        # 1. Empty check
        if len(graph.nodes()) == 0:
             return False, "Network is empty"

        # 2. Max Nodes check
        if len(graph.nodes()) > self.max_nodes:
            return False, f"Network too large: {len(graph.nodes())} nodes (max {self.max_nodes})"

        # 3. Cycle Check (Acyclic property)
        if not nx.is_directed_acyclic_graph(graph):
            try:
                # Try to find the specific cycle for better feedback
                cycle = nx.find_cycle(graph, orientation='original')
                cycle_str = " -> ".join([f"{u}->{v}" for u, v, d in cycle])
                return False, f"Network contains a cycle: {cycle_str}. Network must be a DAG."
            except:
                return False, "Network contains cycles - must be a DAG"

        # 4. Loose Nodes / Connectivity Check
        # A graph with > 1 node must be at least weakly connected to be a valid single BN
        if len(graph.nodes()) > 1:
            if not nx.is_weakly_connected(graph):
                # Identify totally isolated nodes (degree 0)
                isolated_nodes = [n for n in graph.nodes() if graph.degree(n) == 0]
                
                if isolated_nodes:
                    return False, f"Loose nodes detected: {', '.join(isolated_nodes)}. All nodes must be connected."
                else:
                    return False, "Network is fragmented into disconnected components. Please connect all parts."

        # 5. Parent Count Check
        for node in graph.nodes():
            parents = list(graph.predecessors(node))
            if len(parents) > self.max_parents:
                return False, f"Node '{node}' has too many parents: {len(parents)} (max {self.max_parents})"

        return True, ""

    def validate_node_name(self, name: str, existing_nodes: List[str]) -> Tuple[bool, str]:
        if not name or not name.strip():
            return False, "Node name cannot be empty"
        if name in existing_nodes:
            return False, f"Node '{name}' already exists"
        if not name.replace('_', '').isalnum():
            return False, "Node name must be alphanumeric (underscore allowed)"
        if len(name) > 20:
            return False, "Node name too long (max 20 characters)"
        return True, ""

    def validate_cpt(self, cpt: List[float], num_combinations: int, num_states: int = 2) -> Tuple[bool, str]:
        expected_length = num_combinations * num_states
        if len(cpt) != expected_length:
            return False, f"CPT length mismatch: expected {expected_length}, got {len(cpt)}"
        for prob in cpt:
            if not isinstance(prob, (int, float)):
                return False, f"Invalid probability type: {type(prob)}"
            if prob < 0 or prob > 1:
                return False, f"Probability out of range [0,1]: {prob}"
        for i in range(0, len(cpt), num_states):
            prob_sum = sum(cpt[i:i+num_states])
            if not np.isclose(prob_sum, 1.0, atol=1e-6):
                return False, f"Probabilities don't sum to 1 at index {i}: sum = {prob_sum}"
        return True, ""

    def validate_states(self, states_str: str) -> Tuple[bool, List[str], str]:
        if not states_str or not states_str.strip():
            return False, [], "States cannot be empty"
        states = [s.strip() for s in states_str.split(',')]
        if len(states) < 2:
            return False, [], "Must have at least 2 states"
        if len(states) > 2:
            self.logger.warning("More than 2 states specified - using first 2 only")
            states = states[:2]
        if len(set(states)) != len(states):
            return False, [], "Duplicate state names"
        return True, states, ""


class ExportManager:
    def __init__(self):
        self.logger = logging.getLogger('QBN.Export')

    def export_circuit_to_qasm(self, circuit: QuantumCircuit, filename: str) -> bool:
        try:
            from qiskit import qasm2
            qasm_str = qasm2.dumps(circuit)
            with open(filename, 'w') as f:
                f.write(qasm_str)
            self.logger.info(f"Exported QASM to {filename}")
            return True
        except ImportError:
            try:
                qasm_str = circuit.qasm()
                with open(filename, 'w') as f:
                    f.write(qasm_str)
                self.logger.info(f"Exported QASM to {filename}")
                return True
            except AttributeError as e:
                self.logger.error(f"QASM export not supported: {e}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to export QASM: {e}")
            return False

    def export_network_to_dot(self, graph: nx.DiGraph, filename: str) -> bool:
        try:
            with open(filename, 'w') as f:
                f.write("digraph BayesianNetwork {\n")
                for node in graph.nodes():
                    f.write(f'  "{node}";\n')
                for edge in graph.edges():
                    f.write(f'  "{edge[0]}" -> "{edge[1]}";\n')
                f.write("}\n")
            self.logger.info(f"Exported DOT to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export DOT: {e}")
            return False

    def export_results_to_csv(self, samples: Dict[Any, float], filename: str) -> bool:
        try:
            import csv
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['State', 'Probability'])
                for state, prob in sorted(samples.items()):
                    writer.writerow([state, prob])
            self.logger.info(f"Exported results to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")
            return False


# ============================================================================
# MAIN APPLICATION CLASS
# ============================================================================

class QBNApp:
    """Main Quantum Bayesian Network Application with all enhancements."""
    # put these inside your GUI class


    def show_circuit_mpl(self, qc: 'QuantumCircuit') -> None:
        """Render QuantumCircuit (mpl) into the existing circuit_frame/canvas safely."""
        if qc is None:
            self.logger.warning("show_circuit_mpl called with qc=None")
            return

        # Try to get a Figure from Qiskit directly (preferred)
        try:
            fig = qc.draw(output='mpl')  # returns a matplotlib.figure.Figure
        except Exception as e:
            self.logger.exception("qc.draw(output='mpl') failed: %s", e)
            # Fallback: try drawing into existing axis via circuit_drawer
            try:
                from qiskit.visualization import circuit_drawer
                self.circuit_ax.clear()
                circuit_drawer(qc, output='mpl', style='bw', ax=self.circuit_ax,
                               plot_barriers=False, justify='none', fold=-1)
                self.circuit_fig.tight_layout()
                self.circuit_canvas.draw()
                self.root.update()
                return
            except Exception as e2:
                self.logger.exception("Fallback circuit_drawer failed: %s", e2)
                # show text representation as last resort
                txt = qc.draw(output='text')
                self.circuit_ax.clear()
                self.circuit_ax.text(0.01, 0.99, txt, ha='left', va='top', fontsize=7, family='monospace')
                self.circuit_canvas.draw()
                self.root.update()
                return

        # Remove old canvas widget to avoid duplicates
        try:
            if hasattr(self, '_circuit_canvas') and self._circuit_canvas is not None:
                self._circuit_canvas.get_tk_widget().destroy()
        except Exception:
            pass

        # Create a new canvas bound to the Qiskit figure and keep references on self
        self._circuit_fig = fig
        self._circuit_canvas = FigureCanvasTkAgg(fig, master=self.circuit_canvas.get_tk_widget().master)
        self._circuit_canvas.draw()
        widget = self._circuit_canvas.get_tk_widget()
        widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.root.update()

    def show_histogram(self, samples: dict, title: str = "Sampling Results") -> None:
        """Render histogram from a dict(samples) into histogram canvas safely."""
        # samples might be {state_str: probability} or counts; both handled
        if not samples:
            self.logger.warning("show_histogram: samples empty")
            # clear area
            self.histogram_ax.clear()
            self.histogram_ax.text(0.5, 0.5, "No samples", ha='center', va='center', color='gray')
            self.histogram_canvas.draw()
            self.root.update()
            return

        try:
            # Let Qiskit make the figure (consistent style), but fallback to manual bar if needed
            fig = plot_histogram(samples, title=title)
        except Exception as e:
            self.logger.exception("plot_histogram failed: %s", e)
            # fallback to manual drawing on existing axes
            self.histogram_ax.clear()
            sorted_samples = sorted(samples.items(), key=lambda x: x[0])
            labels = [str(k) for k, v in sorted_samples]
            values = [v for k, v in sorted_samples]
            bars = self.histogram_ax.bar(labels, values)
            self.histogram_ax.set_title(title)
            self.histogram_fig.tight_layout()
            self.histogram_canvas.draw()
            self.root.update()
            return

        # Remove prior canvas widget to avoid GC issues
        try:
            if hasattr(self, '_hist_canvas') and self._hist_canvas is not None:
                self._hist_canvas.get_tk_widget().destroy()
        except Exception:
            pass

        self._hist_fig = fig
        self._hist_canvas = FigureCanvasTkAgg(fig, master=self.histogram_canvas.get_tk_widget().master)
        self._hist_canvas.draw()
        w = self._hist_canvas.get_tk_widget()
        w.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.root.update()

    def __init__(self, root: tk.Tk):
        """Initialize the enhanced application."""
        self.root = root
        self.root.title("BayesQ")
        self.root.geometry("1600x900")

        # Initialize configuration
        self.config = ConfigManager()

        # Initialize logging
        log_level = self.config.get('DEFAULT', 'log_level', fallback='INFO')
        self.logger = setup_logging(log_level)
        self.logger.info("="*60)
        self.logger.info("BayesQ")
        self.logger.info("="*60)

        # Initialize validators and utilities
        max_nodes = int(self.config.get('DEFAULT', 'max_network_size', fallback='20'))
        max_parents = int(self.config.get('DEFAULT', 'max_parents', fallback='5'))
        self.validator = NetworkValidator(max_nodes=max_nodes, max_parents=max_parents)
        self.performance = PerformanceMonitor()
        self.exporter = ExportManager()

        # Core Data Structures
        self.graph = nx.DiGraph()
        self.node_data: Dict[str, Dict] = {}
        self.node_positions: Dict[str, Tuple[float, float]] = {}
        self.cpt_entry_widgets: Dict = {}
        self.selected_cpt_node: Optional[str] = None
        self.inference_evidence: Dict[str, int] = {}
        self.inference_query: Dict[str, int] = {}
        
        # Selection state
        self.selected_graph_node: Optional[str] = None

        # Qiskit Objects
        self.qbayesian: Optional[QBayesian] = None
        self.node_name_to_idx: Dict[str, int] = {}
        self.idx_to_node_name: Dict[int, str] = {}
        self.quantum_circuit: Optional[QuantumCircuit] = None
        self.last_samples: Optional[Dict] = None

        # Drag-and-Drop State
        self.dragged_node: Optional[str] = None
        self._dragging: bool = False
        self._last_drag_time: float = 0.0
        self._drag_lock_xlim: Optional[Tuple[float, float]] = None
        self._drag_lock_ylim: Optional[Tuple[float, float]] = None

        # Setup GUI
        self.setup_style()
        self.create_menu()
        
        # Main Layout
        self.main_paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_paned_window.pack(fill=tk.BOTH, expand=True)

        self.controls_frame = ttk.Frame(self.main_paned_window, width=450)
        self.main_paned_window.add(self.controls_frame, weight=1)
        self.create_controls_panel(self.controls_frame)

        self.vis_frame = ttk.Frame(self.main_paned_window)
        self.main_paned_window.add(self.vis_frame, weight=3)
        self.create_visualization_panel(self.vis_frame)

        # Status Bar
        self.status_bar = ttk.Label(root, text="Ready - Use Ctrl+N for new network",
                                    relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Initialize bindings (CALLED AFTER CANVASES ARE CREATED)
        self.setup_shortcuts() 

        # Setup close protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Ensure window has focus on startup so keys work immediately
        self.root.focus_force()

        self.logger.info("GUI initialized successfully")

    def setup_style(self) -> None:
        """Setup ttk styling."""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', padding=6, relief="flat", 
                             background="#007bff", foreground="white")
        self.style.map('TButton', background=[('active', '#0056b3')])
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabelframe', background='#f0f0f0', 
                             borderwidth=2, relief="groove")
        self.style.configure('TLabelframe.Label', background='#f0f0f0', 
                             foreground='#333')

    def setup_shortcuts(self) -> None:
        """Bind keyboard shortcuts with debug logging and focus handling."""
        logger.debug("DEBUG: --- Setup Shortcuts Called ---")
        
        def create_callback(name, func):
            def callback(event=None):
                logger.debug(f"DEBUG: Key pressed: {name}")
                func()
                return "break" # Stop propagation to ensure matplotlib/other widgets don't steal
            return callback

        # Define shortcuts map (Includes uppercase for safety/Caps Lock)
        shortcuts = {
            '<Control-n>': ('New', lambda: self.new_network()),
            '<Control-o>': ('Open', lambda: self.load_network()),
            '<Control-s>': ('Save', lambda: self.save_network()),
            '<Control-b>': ('Build', lambda: self.build_and_display_circuit()),
            '<Control-r>': ('Run', lambda: self.run_inference()),
            '<Control-e>': ('Export', lambda: self.export_circuit_qasm()),
            '<Delete>': ('Delete', self.delete_selected_node_shortcut),
            # Uppercase variants
            '<Control-n>': ('New', lambda: self.new_network()),
            '<Control-o>': ('Open', lambda: self.load_network()),
            '<Control-s>': ('Save', lambda: self.save_network()),
            '<Control-b>': ('Build', lambda: self.build_and_display_circuit()),
            '<Control-r>': ('Run', lambda: self.run_inference()),
            '<Control-e>': ('Export', lambda: self.export_circuit_qasm()),
        }

        # 1. Bind to Root (Global application level)
        for key, (name, func) in shortcuts.items():
            cb = create_callback(name, func)
            self.root.bind(key, cb)
            self.root.bind_all(key, cb) # Fallback

        # 2. Bind specifically to Network Canvas (Matplotlib steals focus)
        if hasattr(self, 'network_canvas'):
            canvas_widget = self.network_canvas.get_tk_widget()
            # Make sure canvas can take focus and receive events
            canvas_widget.config(takefocus=1)
            for key, (name, func) in shortcuts.items():
                canvas_widget.bind(key, create_callback(name, func))

        # 3. Bind to Circuit Canvas
        if hasattr(self, 'circuit_canvas'):
            circuit_widget = self.circuit_canvas.get_tk_widget()
            circuit_widget.config(takefocus=1)
            for key, (name, func) in shortcuts.items():
                circuit_widget.bind(key, create_callback(name, func))
        
        logger.debug("DEBUG: Shortcuts bound to Root and Canvases.")

    def delete_selected_node_shortcut(self, event=None):
        """Wrapper to delete currently selected node from graph."""
        # Prevent deletion if user is typing in a text field
        focused_widget = self.root.focus_get()
        if isinstance(focused_widget, (tk.Entry, tk.Text, ttk.Entry)):
            logger.debug("DEBUG: Delete blocked (typing in entry)")
            return

        if self.selected_graph_node:
            # Ensure the correct node is in the combobox, then call delete
            self.delete_node_combo.set(self.selected_graph_node)
            self.delete_node()
        else:
            self.status_bar.config(text="No node selected to delete (Click a node first)")

    def create_menu(self) -> None:
        """Create enhanced application menu bar."""
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        # File Menu
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="New Network", command=self.new_network, accelerator="Ctrl+N")
        file_menu.add_command(label="Open Network", command=self.load_network, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Network", command=self.save_network, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Load Example (2-node)", command=self.load_2node_example)
        file_menu.add_command(label="Load Example (Burglary Alarm)", command=self.load_burglary_example)
        file_menu.add_separator()
        file_menu.add_command(label="Export Circuit (QASM)", command=self.export_circuit_qasm, accelerator="Ctrl+E")
        file_menu.add_command(label="Export Network (DOT)", command=self.export_network_dot)
        file_menu.add_command(label="Export Results (CSV)", command=self.export_results_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        self.menu_bar.add_cascade(label="File", menu=file_menu)

        # Run Menu
        run_menu = tk.Menu(self.menu_bar, tearoff=0)
        run_menu.add_command(label="Validate Network", command=self.validate_network_ui)
        run_menu.add_command(label="Build Circuit", command=self.build_and_display_circuit, accelerator="Ctrl+B")
        run_menu.add_command(label="Run Inference", command=self.run_inference, accelerator="Ctrl+R")
        self.menu_bar.add_cascade(label="Run", menu=run_menu)

        # Tools Menu
        tools_menu = tk.Menu(self.menu_bar, tearoff=0)
        tools_menu.add_command(label="Network Statistics", command=self.show_network_statistics)
        tools_menu.add_command(label="Settings", command=self.show_settings)
        self.menu_bar.add_cascade(label="Tools", menu=tools_menu)

        # Help Menu
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        help_menu.add_command(label="About", command=self.show_about)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)

    def show_about(self) -> None:
        """Show about dialog."""
        about_text = """BayesQ)

Build and simulate Bayesian Networks using Qiskit Machine Learning.

Keyboard Shortcuts:
-------------------
Ctrl + N : New Network
Ctrl + O : Open Network
Ctrl + S : Save Network
Ctrl + B : Build Circuit
Ctrl + R : Run Inference
Ctrl + E : Export QASM
Delete   : Delete Selected Node

Features:
• Visual network editor with drag-and-drop
• Interactive tooltips for guidance
• Simplified CPT editor (auto-calculates P(0))
• Quantum circuit visualization  
• Multiple noise models
• Performance monitoring
• Strict graph validation

© 2025 - MIT License"""
        messagebox.showinfo("About QBN Builder", about_text)

    def show_user_guide(self) -> None:
        """Show user guide window."""
        guide_window = tk.Toplevel(self.root)
        guide_window.title("User Guide")
        guide_window.geometry("700x600")

        guide_text = scrolledtext.ScrolledText(guide_window, wrap=tk.WORD, font=("Arial", 10))
        guide_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        guide_content = """QUICK START GUIDE

1. CREATE NETWORK
   - Add nodes with binary states (0,1)
   - Connect nodes with edges (Parent → Child)
   - Network must be a DAG (no cycles)
   - All nodes must be connected (no loose nodes)
   * Tip: Hover over buttons for help.

2. EDITING
   - Click a node on the graph to select it.
   - Use the 'Delete' key to remove selected nodes.
   - Use drag-and-drop to rearrange the graph.

3. DEFINE CPTs
   - Select node in CPT Editor tab
   - Enter P(node=1) for each parent configuration
   - P(node=0) is auto-calculated

4. RUN INFERENCE
   - Set evidence (observed values)
   - Set query (what to compute)
   - Choose noise model
   - Click "Run Inference" or Ctrl+R

5. EXPORT
   - File → Export Circuit (QASM)
   - File → Export Network (DOT)
   - File → Export Results (CSV)

For detailed documentation, see README.md"""

        guide_text.insert('1.0', guide_content)
        guide_text.config(state=tk.DISABLED)

    def show_settings(self) -> None:
        """Show settings dialog."""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("500x300")

        settings_frame = ttk.LabelFrame(settings_window, text="Application Settings")
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(settings_frame, text="Max Network Size:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        max_size_var = tk.StringVar(value=self.config.get('DEFAULT', 'max_network_size'))
        max_size_entry = ttk.Entry(settings_frame, textvariable=max_size_var, width=10)
        max_size_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Label(settings_frame, text="Max Parents per Node:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        max_parents_var = tk.StringVar(value=self.config.get('DEFAULT', 'max_parents'))
        max_parents_entry = ttk.Entry(settings_frame, textvariable=max_parents_var, width=10)
        max_parents_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        def save_settings():
            try:
                self.config.set('DEFAULT', 'max_network_size', max_size_var.get())
                self.config.set('DEFAULT', 'max_parents', max_parents_var.get())
                messagebox.showinfo("Success", "Settings saved. Restart for changes to take effect.")
                settings_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save settings: {e}")

        button_frame = ttk.Frame(settings_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(button_frame, text="Save", command=save_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.RIGHT, padx=5)

    def show_network_statistics(self) -> None:
        """Show network statistics."""
        if not self.graph.nodes():
            messagebox.showinfo("Network Statistics", "Network is empty.")
            return

        num_nodes = len(self.graph.nodes())
        num_edges = len(self.graph.edges())
        root_nodes = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        nodes_with_cpt = sum(1 for n in self.node_data.values() if n['cpt'] is not None)

        stats_text = f"""NETWORK STATISTICS

Structure:
  Nodes: {num_nodes}
  Edges: {num_edges}
  Root nodes: {len(root_nodes)}

CPTs:
  Defined: {nodes_with_cpt}/{num_nodes}

Validation:
  Is DAG: {nx.is_directed_acyclic_graph(self.graph)}
  Is Connected: {nx.is_weakly_connected(self.graph)}"""

        if self.quantum_circuit:
            stats_text += f"""

Quantum Circuit:
  Qubits: {self.quantum_circuit.num_qubits}
  Gates: {self.quantum_circuit.size()}
  Depth: {self.quantum_circuit.depth()}"""

        messagebox.showinfo("Network Statistics", stats_text)

    def validate_network_ui(self) -> None:
        """Validate network and show results."""
        # The validator handles empty checks inside
        is_valid, error_msg = self.validator.validate_network_structure(self.graph)

        if not is_valid:
            messagebox.showerror("Validation Failed", error_msg)
            return

        missing_cpts = [n for n, data in self.node_data.items() if data['cpt'] is None]

        if missing_cpts:
            messagebox.showwarning("Validation", 
                                   f"CPTs missing for nodes: {', '.join(missing_cpts)}")
            return

        for node_name, node_info in self.node_data.items():
            parents = list(self.graph.predecessors(node_name))
            parent_states_list = [self.node_data[p]['states'] for p in parents]
            num_combinations = len(list(itertools.product(*parent_states_list))) if parents else 1

            is_valid, error_msg = self.validator.validate_cpt(node_info['cpt'], num_combinations)

            if not is_valid:
                messagebox.showerror("Validation Failed", 
                                     f"CPT validation failed for '{node_name}': {error_msg}")
                return

        messagebox.showinfo("Validation", "✓ Network is valid!\n\nAll constraints satisfied.")
        self.logger.info("Network validation successful")

    def on_closing(self) -> None:
        """Handle application closing."""
        if self.config.get('DEFAULT', 'auto_save') == 'True' and self.graph.nodes():
            try:
                auto_save_path = 'autosave.qbn.json'
                self.save_network_to_file(auto_save_path)
                self.logger.info(f"Auto-saved to {auto_save_path}")
            except Exception as e:
                self.logger.error(f"Auto-save failed: {e}")

        self.logger.info("Application closing")
        self.root.destroy()

    def export_circuit_qasm(self) -> None:
        """Export quantum circuit to QASM format."""
        if not self.quantum_circuit:
            messagebox.showwarning("Warning", "Build circuit first")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".qasm",
            filetypes=[("QASM files", "*.qasm"), ("All files", "*.*")]
        )

        if filename:
            if self.exporter.export_circuit_to_qasm(self.quantum_circuit, filename):
                messagebox.showinfo("Success", f"Circuit exported to {filename}")

    def export_network_dot(self) -> None:
        """Export network structure to DOT format."""
        if not self.graph.nodes():
            messagebox.showwarning("Warning", "Network is empty")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".dot",
            filetypes=[("DOT files", "*.dot"), ("All files", "*.*")]
        )

        if filename:
            if self.exporter.export_network_to_dot(self.graph, filename):
                messagebox.showinfo("Success", f"Network exported to {filename}")

    def export_results_csv(self) -> None:
        """Export inference results to CSV."""
        if not self.last_samples:
            messagebox.showwarning("Warning", "Run sampling inference first")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            if self.exporter.export_results_to_csv(self.last_samples, filename):
                messagebox.showinfo("Success", f"Results exported to {filename}")

    def create_controls_panel(self, parent: ttk.Frame) -> None:
        """Create control panel with tabs."""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tab_build = ttk.Frame(notebook)
        notebook.add(self.tab_build, text="Network Builder")
        self.create_build_tab(self.tab_build)

        self.tab_cpt = ttk.Frame(notebook)
        notebook.add(self.tab_cpt, text="CPT Editor")
        self.create_cpt_tab(self.tab_cpt)

        self.tab_infer = ttk.Frame(notebook)
        notebook.add(self.tab_infer, text="Inference")
        self.create_inference_tab(self.tab_infer)

    def create_build_tab(self, parent: ttk.Frame) -> None:
        """Create network building tab with validation and tooltips."""
        info_frame = ttk.LabelFrame(parent, text="Instructions")
        info_frame.pack(fill=tk.X, padx=10, pady=10)

        info_text = ("1. Add nodes (variables) with binary states\n"
                     "2. Connect nodes with directed edges\n"
                     "3. Define CPTs (probabilities) for each node\n"
                     "4. Build circuit and run inference")
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack(padx=10, pady=10)

        # Node frame
        node_frame = ttk.LabelFrame(parent, text="Add Node")
        node_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(node_frame, text="Name:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.node_name_entry = ttk.Entry(node_frame)
        self.node_name_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        CreateToolTip(self.node_name_entry, "Enter a unique name for the variable (e.g., 'Rain', 'Alarm').")

        ttk.Label(node_frame, text="States:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.node_states_entry = ttk.Entry(node_frame)
        self.node_states_entry.insert(0, "0,1")
        self.node_states_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        CreateToolTip(self.node_states_entry, "Comma-separated states (e.g., '0,1' or 'False,True'). Default is 0,1.")

        add_btn = ttk.Button(node_frame, text="Add Node", command=self.add_node)
        add_btn.grid(row=2, column=0, columnspan=2, pady=10)
        CreateToolTip(add_btn, "Create a new node with the specified name and states.")
        
        node_frame.columnconfigure(1, weight=1)

        # Edge frame
        edge_frame = ttk.LabelFrame(parent, text="Add Edge (Parent → Child)")
        edge_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(edge_frame, text="From (Parent):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.edge_from_combo = ttk.Combobox(edge_frame, state="readonly")
        self.edge_from_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(edge_frame, text="To (Child):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.edge_to_combo = ttk.Combobox(edge_frame, state="readonly")
        self.edge_to_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

        edge_btn = ttk.Button(edge_frame, text="Add Edge", command=self.add_edge)
        edge_btn.grid(row=2, column=0, columnspan=2, pady=10)
        CreateToolTip(edge_btn, "Create a directed dependency between two nodes.")
        
        edge_frame.columnconfigure(1, weight=1)

        # Delete frame
        delete_frame = ttk.LabelFrame(parent, text="Delete")
        delete_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(delete_frame, text="Node:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.delete_node_combo = ttk.Combobox(delete_frame, state="readonly")
        self.delete_node_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        del_btn = ttk.Button(delete_frame, text="Delete Node", command=self.delete_node)
        del_btn.grid(row=0, column=2, padx=5, pady=5)
        CreateToolTip(del_btn, "Remove selected node and all connected edges (Shortcut: Delete Key)")
        
        delete_frame.columnconfigure(1, weight=1)

    def create_cpt_tab(self, parent: ttk.Frame) -> None:
        """Create CPT editor tab."""
        selector_frame = ttk.Frame(parent)
        selector_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(selector_frame, text="Select Node:").pack(side=tk.LEFT, padx=5)
        self.cpt_node_combo = ttk.Combobox(selector_frame, state="readonly")
        self.cpt_node_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.cpt_node_combo.bind("<<ComboboxSelected>>", self.load_cpt_editor)

        cpt_outer_frame = ttk.LabelFrame(parent, text="Conditional Probability Table")
        cpt_outer_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.cpt_canvas = tk.Canvas(cpt_outer_frame, bg='#f0f0f0', highlightthickness=0)
        self.cpt_scrollbar = ttk.Scrollbar(cpt_outer_frame, orient="vertical", 
                                             command=self.cpt_canvas.yview)
        self.cpt_scrollable_frame = ttk.Frame(self.cpt_canvas)

        self.cpt_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.cpt_canvas.configure(scrollregion=self.cpt_canvas.bbox("all"))
        )

        self.cpt_canvas.create_window((0, 0), window=self.cpt_scrollable_frame, anchor="nw")
        self.cpt_canvas.configure(yscrollcommand=self.cpt_scrollbar.set)

        self.cpt_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.cpt_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.save_cpt_button = ttk.Button(parent, text="Save CPT", command=self.save_cpt)
        self.save_cpt_button.pack(pady=10)
        self.save_cpt_button.config(state=tk.DISABLED)
        CreateToolTip(self.save_cpt_button, "Commit changes to the Probability Table for the selected node.")

    def create_inference_tab(self, parent: ttk.Frame) -> None:
        """Create inference tab with noise model selection and AerSimulator option."""
        # Execution parameters
        exec_frame = ttk.LabelFrame(parent, text="Execution Parameters")
        exec_frame.pack(fill=tk.X, padx=10, pady=10)

        # --- SIMULATION MODE DISPLAY ---
        # Replaced radio buttons with a static label since we only support AerSimulator now
        mode_frame = ttk.Frame(exec_frame)
        mode_frame.grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(mode_frame, text="Simulation Backend: AerSimulator (Shot-based)", 
                  font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # --- SHOT-BASED OPTIONS ---
        self.lbl_shots = ttk.Label(exec_frame, text="Shots:")
        self.lbl_shots.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.shots_entry = ttk.Entry(exec_frame, width=12)
        self.shots_entry.insert(0, self.config.get('DEFAULT', 'shots', fallback='1024'))
        self.shots_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        CreateToolTip(self.shots_entry, "Number of quantum measurements (simulation runs). Higher = more precision.")
        
        self.lbl_shots_hint = ttk.Label(exec_frame, text="(per run)", font=('Arial', 9, 'italic'), foreground='gray')
        self.lbl_shots_hint.grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        exec_frame.columnconfigure(1, weight=1)

        # Noise Model Selection
        self.lbl_noise = ttk.Label(exec_frame, text="Noise Model:")
        self.lbl_noise.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.noise_model_combo = ttk.Combobox(exec_frame, state="readonly", width=20)
        self.noise_model_combo['values'] = [
            'None (Ideal)',
            'Depolarizing (0.1%)',
            'Depolarizing (1%)',
            'Depolarizing (5%)',
            'Thermal Relaxation',
            'Phase Damping',
            'Amplitude Damping',
            'Combined (Depol + Thermal)'
        ]
        default_noise = self.config.get('DEFAULT', 'noise_model', fallback='None (Ideal)')
        self.noise_model_combo.set(default_noise)
        self.noise_model_combo.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        CreateToolTip(self.noise_model_combo, "Simulate hardware errors to test algorithm robustness.")

        self.lbl_noise_hint = ttk.Label(exec_frame, text="(AerSimulator)", font=('Arial', 9, 'italic'), foreground='gray')
        self.lbl_noise_hint.grid(row=2, column=2, padx=5, pady=5, sticky=tk.W)
        self.lbl_ci = ttk.Label(exec_frame, text="Confidence Level (e.g. 0.95):")
        self.lbl_ci.grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.cilevel_entry = ttk.Entry(exec_frame, width=12)
        self.cilevel_entry.insert(0, "0.95")
        self.cilevel_entry.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)
        CreateToolTip(self.cilevel_entry, "Target confidence level (e.g. 0.95). Requires 'scipy' library for custom values; otherwise strictly defaults to 95%.")
        # Evidence Frame
        evidence_frame = ttk.LabelFrame(parent, text="Evidence (Observed Values)")
        evidence_frame.pack(fill=tk.X, padx=10, pady=10)

        add_evidence_frame = ttk.Frame(evidence_frame)
        add_evidence_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(add_evidence_frame, text="Node:").pack(side=tk.LEFT, padx=(0, 5))
        self.evidence_node_combo = ttk.Combobox(add_evidence_frame, state="readonly", width=12)
        self.evidence_node_combo.pack(side=tk.LEFT, padx=5)
        self.evidence_node_combo.bind("<<ComboboxSelected>>", self.on_evidence_node_select)

        ttk.Label(add_evidence_frame, text="State:").pack(side=tk.LEFT, padx=5)
        self.evidence_state_combo = ttk.Combobox(add_evidence_frame, state="readonly", width=8)
        self.evidence_state_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(add_evidence_frame, text="Add", command=self.add_evidence_item, 
                   width=8).pack(side=tk.LEFT, padx=5)

        list_evidence_frame = ttk.Frame(evidence_frame)
        list_evidence_frame.pack(fill=tk.X, expand=True, padx=5, pady=(0, 5))

        evidence_scrollbar = ttk.Scrollbar(list_evidence_frame, orient=tk.VERTICAL)
        self.evidence_listbox = tk.Listbox(list_evidence_frame, height=3, 
                                           yscrollcommand=evidence_scrollbar.set,
                                           exportselection=False)
        evidence_scrollbar.config(command=self.evidence_listbox.yview)
        evidence_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.evidence_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(evidence_frame, text="Clear Evidence", 
                   command=self.clear_evidence).pack(pady=5)

        # Query Frame  
        query_frame = ttk.LabelFrame(parent, text="Query (What to Compute)")
        query_frame.pack(fill=tk.X, padx=10, pady=10)

        add_query_frame = ttk.Frame(query_frame)
        add_query_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(add_query_frame, text="Node:").pack(side=tk.LEFT, padx=(0, 5))
        self.query_node_combo = ttk.Combobox(add_query_frame, state="readonly", width=12)
        self.query_node_combo.pack(side=tk.LEFT, padx=5)
        self.query_node_combo.bind("<<ComboboxSelected>>", self.on_query_node_select)

        ttk.Label(add_query_frame, text="State:").pack(side=tk.LEFT, padx=5)
        self.query_state_combo = ttk.Combobox(add_query_frame, state="readonly", width=8)
        self.query_state_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(add_query_frame, text="Add", command=self.add_query_item, 
                   width=8).pack(side=tk.LEFT, padx=5)

        list_query_frame = ttk.Frame(query_frame)
        list_query_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)

        query_scrollbar = ttk.Scrollbar(list_query_frame, orient=tk.VERTICAL)
        self.query_listbox = tk.Listbox(list_query_frame, height=3, 
                                        yscrollcommand=query_scrollbar.set,
                                        exportselection=False)
        query_scrollbar.config(command=self.query_listbox.yview)
        query_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.query_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(query_frame, text="Clear Query", command=self.clear_query).pack(pady=5)

        inf_btn = ttk.Button(parent, text="▶ Run Inference", command=self.run_inference)
        inf_btn.pack(pady=15)
        CreateToolTip(inf_btn, "Build circuit and execute simulation (Shortcut: Ctrl+R)")

        results_frame = ttk.LabelFrame(parent, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=8,
                                                      font=("Courier New", 9), bg="white", fg="black")
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_visualization_panel(self, parent: ttk.Frame) -> None:
        """Create visualization panel with multiple tabs."""
        self.viz_notebook = ttk.Notebook(parent)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_network = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.tab_network, text="Network Graph")
        self.create_network_viz(self.tab_network)

        self.tab_circuit = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.tab_circuit, text="Quantum Circuit")
        self.create_circuit_viz(self.tab_circuit)

        self.tab_histogram = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.tab_histogram, text="Results")
        self.create_histogram_viz(self.tab_histogram)

        self.tab_code = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.tab_code, text="Generated Code")
        self.create_code_viz(self.tab_code)

        # Resource Logs Tab
        self.tab_resource_log = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.tab_resource_log, text="Resource Logs")
        self.resource_log_text = scrolledtext.ScrolledText(self.tab_resource_log, wrap=tk.WORD, height=16, state="disabled")
        self.resource_log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Execution Logs Tab
        self.tab_execution_log = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.tab_execution_log, text="Execution Logs")
        self.exec_log_text = scrolledtext.ScrolledText(self.tab_execution_log, wrap=tk.WORD, height=16, state="disabled")
        self.exec_log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # === CONNECT LOGGER TO GUI ===
        text_handler = TextWidgetHandler(self.exec_log_text)
        self.logger.addHandler(text_handler)
        self.logger.info("Execution Log connected to GUI.")

    def create_network_viz(self, parent: ttk.Frame) -> None:
        """Create network visualization canvas."""
        self.network_fig = Figure(figsize=(8, 6), dpi=100)
        self.network_fig.patch.set_facecolor('#f0f0f0')
        self.network_ax = self.network_fig.add_subplot(111)
        self.network_ax.set_facecolor('#ffffff')
        self.network_ax.axis('off')

        self.network_canvas = FigureCanvasTkAgg(self.network_fig, master=parent)
        self.network_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.network_canvas.draw()

        self.network_toolbar = NavigationToolbar2Tk(self.network_canvas, parent, pack_toolbar=False)
        self.network_toolbar.update()
        self.network_toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.network_canvas.mpl_connect('button_press_event', self.on_press)
        self.network_canvas.mpl_connect('button_release_event', self.on_release)
        self.network_canvas.mpl_connect('motion_notify_event', self.on_motion)

    def create_circuit_viz(self, parent: ttk.Frame) -> None:
        """Create circuit visualization canvas."""
        self.circuit_fig = Figure(figsize=(10, 6), dpi=100)
        self.circuit_fig.patch.set_facecolor('#f0f0f0')
        self.circuit_ax = self.circuit_fig.add_subplot(111)
        self.circuit_ax.set_facecolor('#ffffff')
        self.circuit_ax.axis('off')
        self.circuit_ax.text(0.5, 0.5, "Click 'Build Circuit' (Ctrl+B) to generate",
                             ha='center', va='center', fontsize=14, color='gray',
                             transform=self.circuit_ax.transAxes)

        self.circuit_canvas = FigureCanvasTkAgg(self.circuit_fig, master=parent)
        self.circuit_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.circuit_canvas.draw()

        self.circuit_toolbar = NavigationToolbar2Tk(self.circuit_canvas, parent, pack_toolbar=False)
        self.circuit_toolbar.update()
        self.circuit_toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_histogram_viz(self, parent: ttk.Frame) -> None:
        """Create histogram visualization canvas."""
        self.histogram_fig = Figure(figsize=(8, 6), dpi=100)
        self.histogram_fig.patch.set_facecolor('#f0f0f0')
        self.histogram_ax = self.histogram_fig.add_subplot(111)
        self.histogram_ax.set_facecolor('#ffffff')
        self.histogram_ax.text(0.5, 0.5, "Run inference to see results",
                               ha='center', va='center', fontsize=14, color='gray')
        self.histogram_ax.axis('off')

        self.histogram_canvas = FigureCanvasTkAgg(self.histogram_fig, master=parent)
        self.histogram_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.histogram_canvas.draw()

        self.histogram_toolbar = NavigationToolbar2Tk(self.histogram_canvas, parent, pack_toolbar=False)
        self.histogram_toolbar.update()
        self.histogram_toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_code_viz(self, parent):
        """Code generation tab"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Generate Code", command=self.gen_code).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Copy to Clipboard", command=self.copy_code).pack(side=tk.LEFT, padx=5)

        self.code_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=('Courier', 10), bg='#f8f8f8')
        self.code_text.pack(fill=tk.BOTH, expand=True)
        self.code_text.insert('1.0', '# Run inference first, then click Generate Code')

    def gen_code(self):
        """Generate complete executable code"""
        if not self.quantum_circuit:
            messagebox.showwarning("Warning", "Run inference first")
            return

        try:
            self.code_text.delete('1.0', tk.END)
            L = []

            try:
                shots = int(self.shots_entry.get())
            except:
                shots = 1024
            noise_selection = self.noise_model_combo.get()
            
            L.append("# ================================================")
            L.append("# Quantum Bayesian Network - Generated Code")
            L.append("# ================================================\n")

            L.append("from qiskit import QuantumCircuit, QuantumRegister")
            L.append("from qiskit_machine_learning.algorithms import QBayesian")
            L.append("from qiskit.primitives import BackendSampler")
            L.append("from qiskit_aer import AerSimulator")
            if noise_selection != 'None (Ideal)':
                L.append("from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error")
                
            L.append("import numpy as np")
            L.append("import itertools")
            L.append("import networkx as nx\n")

            L.append(f"# Network: {list(self.node_data.keys())}")
            L.append(f"# Edges: {list(self.graph.edges())}\n")

            L.append("# Node data with CPTs")
            L.append("node_data = {")
            for n, info in self.node_data.items():
                L.append(f"    '{n}': {{")
                L.append(f"        'states': {info['states']},")
                L.append(f"        'cpt': {info['cpt']}")
                L.append("    },")
            L.append("}\n")

            L.append("# Build graph")
            L.append("graph = nx.DiGraph()")
            L.append(f"graph.add_nodes_from({list(self.node_data.keys())})")
            L.append(f"graph.add_edges_from({list(self.graph.edges())})\n")

            nodes = list(self.node_name_to_idx.keys())
            L.append(f"sorted_nodes = {nodes}")
            L.append(f"node_name_to_idx = {dict(self.node_name_to_idx)}\n")

            L.append("# ========== Build Circuit ==========")
            L.append("def build_circuit():")
            L.append('    """Build quantum circuit for Bayesian Network"""')
            L.append("    qr = [QuantumRegister(1, name=n) for n in sorted_nodes]")
            L.append("    qc = QuantumCircuit(*qr)\n")

            L.append("    for i, node in enumerate(sorted_nodes):")
            L.append("        states = node_data[node]['states']")
            L.append("        cpt = node_data[node]['cpt']")
            L.append("        parents = sorted(list(graph.predecessors(node)))\n")

            L.append("        if not parents:")
            L.append("            # Root node")
            L.append("            prob = cpt[1]")
            L.append("            theta = 2 * np.arcsin(np.sqrt(prob))")
            L.append("            qc.ry(theta, i)")
            L.append("            qc.barrier()\n")

            L.append("        else:")
            L.append("            # Node with parents")
            L.append("            p_idx = [node_name_to_idx[p] for p in parents]")
            L.append("            p_states = [node_data[p]['states'] for p in parents]")
            L.append("            combos = list(itertools.product(*p_states))\n")

            L.append("            # All-1s first")
            L.append("            idx = (len(combos)-1) * len(states) + 1")
            L.append("            theta = 2 * np.arcsin(np.sqrt(cpt[idx]))")
            L.append("            if len(parents) == 1:")
            L.append("                qc.cry(theta, p_idx[0], i)")
            L.append("            else:")
            L.append("                qc.mcry(theta, p_idx, i)")
            L.append("            qc.barrier()\n")

            L.append("            # Other combinations")
            L.append("            for ci in range(len(combos)-2, -1, -1):")
            L.append("                combo = combos[ci]")
            L.append("                for p, s in zip(parents, combo):")
            L.append("                    if s == node_data[p]['states'][0]:")
            L.append("                        qc.x(node_name_to_idx[p])")
            L.append("                idx = ci * len(states) + 1")
            L.append("                theta = 2 * np.arcsin(np.sqrt(cpt[idx]))")
            L.append("                if len(parents) == 1:")
            L.append("                    qc.cry(theta, p_idx[0], i)")
            L.append("                else:")
            L.append("                    qc.mcry(theta, p_idx, i)")
            L.append("                for p, s in zip(parents, combo):")
            L.append("                    if s == node_data[p]['states'][0]:")
            L.append("                        qc.x(node_name_to_idx[p])")
            L.append("                qc.barrier()\n")

            L.append("    return qc\n")

            L.append("qc = build_circuit()")
            L.append("print(f'Circuit: {qc.num_qubits} qubits, {qc.size()} gates')\n")

            L.append("# ========== Setup QBayesian ==========")
            
            L.append(f"shots = {shots}")
            if noise_selection != 'None (Ideal)':
                L.append(f"# Noise model: {noise_selection}")
                L.append("noise_model = NoiseModel()")
                
                if 'Depolarizing' in noise_selection:
                    if '0.1%' in noise_selection: rate = 0.001
                    elif '5%' in noise_selection: rate = 0.05
                    else: rate = 0.01
                    L.append(f"error = depolarizing_error({rate}, 1)")
                    L.append(f"error_2q = depolarizing_error({rate}, 2)")
                    L.append("noise_model.add_all_qubit_quantum_error(error, ['ry'])")
                    L.append("noise_model.add_all_qubit_quantum_error(error_2q, ['cry', 'mcry'])")
                
                elif 'Thermal' in noise_selection:
                    L.append("t1, t2 = 50.0, 70.0  # microseconds")
                    L.append("gate_time_1q, gate_time_2q = 50, 300  # nanoseconds")
                    L.append("error = thermal_relaxation_error(t1, t2, gate_time_1q / 1000)")
                    L.append("error_2q = thermal_relaxation_error(t1, t2, gate_time_2q / 1000).tensor(")
                    L.append("            thermal_relaxation_error(t1, t2, gate_time_2q / 1000))")
                    L.append("noise_model.add_all_qubit_quantum_error(error, ['ry'])")
                    L.append("noise_model.add_all_qubit_quantum_error(error_2q, ['cry', 'mcry'])")
                    
                L.append("backend = AerSimulator(noise_model=noise_model)")
            else:
                L.append("backend = AerSimulator()")
        
            L.append("sampler = BackendSampler(backend=backend, options={'shots': shots})")
            L.append("qbayesian = QBayesian(circuit=qc, sampler=sampler)\n")

            L.append("# ========== Evidence and Query ==========")
            L.append("# IMPORTANT: Both evidence and query must be DICTIONARIES")
            L.append("# Keys are register names (strings), values are state indices\n")

            if self.inference_evidence:
                L.append("# Evidence (observed values)")
                L.append("evidence = {")
                for node_name, state_idx in self.inference_evidence.items():
                    state_name = self.node_data[node_name]['states'][state_idx]
                    L.append(f"    '{node_name}': {state_idx},  # {node_name} = {state_name}")
                L.append("}")
            else:
                L.append("evidence = {}")

            L.append("")

            if self.inference_query:
                L.append("# Query (what to compute) - DICTIONARY format")
                L.append("query = {")
                for node_name, state_idx in self.inference_query.items():
                    state_name = self.node_data[node_name]['states'][state_idx]
                    L.append(f"    '{node_name}': {state_idx},  # {node_name} = {state_name}")
                L.append("}")
            else:
                L.append("query = {}  # Empty for sampling")

            L.append("")

            L.append("# ========== Run Inference ==========")
            if self.inference_query:
                L.append("# Perform inference")
                L.append("result = qbayesian.inference(query=query, evidence=evidence)")
                L.append("print(f'P(query | evidence) = {result:.6f}')\n")

                L.append("# 95% Confidence Interval (Wilson score)")
                L.append("n, p, z = 1000, result, 1.96")
                L.append("d = 1 + z**2/n")
                L.append("c = (p + z**2/(2*n))/d")
                L.append("m = z * np.sqrt((p*(1-p)/n + z**2/(4*n**2)))/d")
                L.append("ci_low, ci_high = max(0, c-m), min(1, c+m)")
                L.append("print(f'95% CI: [{ci_low:.6f}, {ci_high:.6f}]')")
            else:
                L.append("# Perform sampling")
                L.append("samples = qbayesian.rejection_sampling(evidence=evidence)")
                L.append("for s, p in sorted(samples.items(), key=lambda x: x[1], reverse=True):")
                L.append("    print(f'State {s}: {p:.6f}')")

            self.code_text.insert('1.0', "\n".join(L))
            self.status_bar.config(text="✓ Code generated")

        except Exception as e:
            messagebox.showerror("Error", f"Failed:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def copy_code(self) -> None:
        """Copy generated code to clipboard."""
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.code_text.get('1.0', tk.END))
            self.status_bar.config(text="✓ Code copied to clipboard")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def add_node(self) -> None:
        """Add node with comprehensive validation."""
        name = self.node_name_entry.get().strip()
        states_str = self.node_states_entry.get().strip()

        # Validate node name
        is_valid, error_msg = self.validator.validate_node_name(name, list(self.graph.nodes()))
        if not is_valid:
            messagebox.showerror("Validation Error", error_msg)
            self.logger.warning(f"Node addition failed: {error_msg}")
            return

        # Validate states
        is_valid, states, error_msg = self.validator.validate_states(states_str)
        if not is_valid:
            messagebox.showerror("Validation Error", error_msg)
            self.logger.warning(f"State validation failed: {error_msg}")
            return

        # Add node
        self.graph.add_node(name)
        self.node_data[name] = {'states': states, 'cpt': None}

        if not self.node_positions:
            self.node_positions = nx.spring_layout(self.graph)
        else:
            self.node_positions[name] = (np.random.rand(), np.random.rand())
            self.node_positions = nx.spring_layout(self.graph, pos=self.node_positions,
                                                   fixed=list(set(self.node_positions.keys()) - {name}))

        self.draw_network()
        self.update_node_lists()
        self.node_name_entry.delete(0, tk.END)
        self.status_bar.config(text=f"Added node: {name}")
        self.logger.info(f"Node added: {name} with states {states}")

    def add_edge(self) -> None:
        """Add edge with validation."""
        from_node = self.edge_from_combo.get()
        to_node = self.edge_to_combo.get()

        if not from_node or not to_node:
            messagebox.showerror("Error", "Both 'From' and 'To' nodes must be selected.")
            return
        if from_node == to_node:
            messagebox.showerror("Error", "Cannot add self-loop.")
            return
        if self.graph.has_edge(from_node, to_node):
            messagebox.showerror("Error", f"Edge already exists.")
            return

        # Temporarily add edge to test
        self.graph.add_edge(from_node, to_node)

        # Validate resulting structure
        # We use the validator, but we only care about structural issues here, not loose nodes (yet)
        # as building happens step by step. But cycles are forbidden immediately.
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(from_node, to_node)
            messagebox.showerror("Validation Error", "Adding this edge would create a cycle.")
            return

        self.node_data[to_node]['cpt'] = None
        self.draw_network()
        self.status_bar.config(text=f"Added edge: {from_node} → {to_node}")
        self.logger.info(f"Edge added: {from_node} → {to_node}")

        # Auto-load CPT editor
        self.cpt_node_combo.set(to_node)
        self.load_cpt_editor()

    def delete_node(self) -> None:
        """Delete node."""
        node = self.delete_node_combo.get()
        if not node:
            messagebox.showwarning("Warning", "Select a node to delete.")
            return

        if messagebox.askyesno("Confirm", f"Delete node '{node}' and all connected edges?"):
            self.graph.remove_node(node)
            del self.node_data[node]
            if node in self.node_positions:
                del self.node_positions[node]
            self.selected_graph_node = None # Clear selection
            self.draw_network()
            self.update_node_lists()
            self.status_bar.config(text=f"Deleted node: {node}")
            self.logger.info(f"Node deleted: {node}")

    def update_node_lists(self) -> None:
        """Update all combobox lists."""
        node_names = sorted(list(self.graph.nodes()))
        self.edge_from_combo['values'] = node_names
        self.edge_to_combo['values'] = node_names
        self.cpt_node_combo['values'] = node_names
        self.evidence_node_combo['values'] = node_names
        self.query_node_combo['values'] = node_names
        self.delete_node_combo['values'] = node_names

    # === CPT Editor (Enhanced with Validation) ===

    def load_cpt_editor(self, event=None) -> None:
        """Load CPT editor for selected node."""
        self.selected_cpt_node = self.cpt_node_combo.get()
        if not self.selected_cpt_node:
            return

        for widget in self.cpt_scrollable_frame.winfo_children():
            widget.destroy()
        self.cpt_entry_widgets = {}

        node_name = self.selected_cpt_node
        node_states = self.node_data[node_name]['states']
        parents = sorted(list(self.graph.predecessors(node_name)))

        # Header
        ttk.Label(self.cpt_scrollable_frame, text="Parent Configuration",
                  font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.cpt_scrollable_frame, text=f"P({node_name}={node_states[1]})",
                  font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(self.cpt_scrollable_frame, text=f"P({node_name}={node_states[0]})",
                  font=('Arial', 9, 'italic'), foreground='gray').grid(row=0, column=2, padx=5, pady=5)

        existing_cpt = self.node_data[node_name].get('cpt', [])

        if not parents:
            # Root node
            ttk.Label(self.cpt_scrollable_frame, text="(Root Node)").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
            entry = ttk.Entry(self.cpt_scrollable_frame, width=10)
            if existing_cpt:
                entry.insert(0, str(existing_cpt[1]))
            entry.grid(row=1, column=1, padx=5, pady=5)
            auto_label = ttk.Label(self.cpt_scrollable_frame, text="(auto)", foreground='gray')
            auto_label.grid(row=1, column=2, padx=5, pady=5)
            self.cpt_entry_widgets['root'] = entry
        else:
            # Node with parents
            parent_states_list = [self.node_data[p]['states'] for p in parents]
            parent_combinations = list(itertools.product(*parent_states_list))

            for i, combo in enumerate(parent_combinations):
                combo_str = ", ".join([f"{p}={s}" for p, s in zip(parents, combo)])
                ttk.Label(self.cpt_scrollable_frame, text=combo_str).grid(row=i + 1, column=0, padx=5, pady=5, sticky=tk.W)
                entry = ttk.Entry(self.cpt_scrollable_frame, width=10)
                if existing_cpt:
                    combo_idx = i
                    prob_idx = combo_idx * 2 + 1
                    entry.insert(0, str(existing_cpt[prob_idx]))
                entry.grid(row=i + 1, column=1, padx=5, pady=5)
                auto_label = ttk.Label(self.cpt_scrollable_frame, text="(auto)", foreground='gray')
                auto_label.grid(row=i + 1, column=2, padx=5, pady=5)
                self.cpt_entry_widgets[combo] = entry

        self.save_cpt_button.config(state=tk.NORMAL)
        self.status_bar.config(text=f"Edit CPT for '{node_name}'")

    def save_cpt(self) -> None:
        """Save CPT with validation."""
        if not self.selected_cpt_node:
            return

        node_name = self.selected_cpt_node
        node_states = self.node_data[node_name]['states']
        parents = sorted(list(self.graph.predecessors(node_name)))
        probabilities = []

        try:
            if not parents:
                entry = self.cpt_entry_widgets['root']
                prob_1 = float(entry.get())
                if prob_1 < 0 or prob_1 > 1:
                    raise ValueError(f"Probability must be between 0 and 1, got {prob_1}")
                prob_0 = 1.0 - prob_1
                probabilities = [prob_0, prob_1]
            else:
                parent_states_list = [self.node_data[p]['states'] for p in parents]
                parent_combinations = list(itertools.product(*parent_states_list))

                for combo in parent_combinations:
                    entry = self.cpt_entry_widgets[combo]
                    prob_1 = float(entry.get())
                    if prob_1 < 0 or prob_1 > 1:
                        combo_str = ", ".join([f"{p}={s}" for p, s in zip(parents, combo)])
                        raise ValueError(f"Probability for '{combo_str}' must be between 0 and 1, got {prob_1}")
                    prob_0 = 1.0 - prob_1
                    probabilities.extend([prob_0, prob_1])

            # Validate CPT
            num_combinations = len(parent_combinations) if parents else 1
            is_valid, error_msg = self.validator.validate_cpt(probabilities, num_combinations)

            if not is_valid:
                messagebox.showerror("Validation Error", error_msg)
                return

            self.node_data[node_name]['cpt'] = probabilities
            self.draw_network()
            self.status_bar.config(text=f"✓ Saved CPT for '{node_name}'")
            self.logger.info(f"CPT saved for {node_name}")
            messagebox.showinfo("Success", f"CPT for '{node_name}' saved successfully")

        except ValueError as e:
            messagebox.showerror("Error", str(e))
            self.logger.error(f"CPT save failed: {e}")

    # === Evidence and Query Management ===

    def on_evidence_node_select(self, event=None) -> None:
        """Handle evidence node selection."""
        node_name = self.evidence_node_combo.get()
        if node_name:
            states = self.node_data[node_name]['states']
            self.evidence_state_combo['values'] = states
            self.evidence_state_combo.set(states[0])

    def add_evidence_item(self) -> None:
        """Add evidence item."""
        node_name = self.evidence_node_combo.get()
        state_name = self.evidence_state_combo.get()

        if not node_name or not state_name:
            messagebox.showwarning("Warning", "Select node and state.")
            return
        if node_name in self.inference_evidence:
            messagebox.showwarning("Warning", f"Evidence for '{node_name}' already set.")
            return

        state_index = self.node_data[node_name]['states'].index(state_name)
        self.inference_evidence[node_name] = state_index
        self.evidence_listbox.insert(tk.END, f"{node_name} = {state_name}")
        self.evidence_node_combo.set('')
        self.evidence_state_combo.set('')
        self.logger.info(f"Evidence added: {node_name} = {state_name}")

    def clear_evidence(self) -> None:
        """Clear all evidence."""
        self.inference_evidence.clear()
        self.evidence_listbox.delete(0, tk.END)

    def on_query_node_select(self, event=None) -> None:
        """Handle query node selection."""
        node_name = self.query_node_combo.get()
        if node_name:
            states = self.node_data[node_name]['states']
            self.query_state_combo['values'] = states
            self.query_state_combo.set(states[0])

    def add_query_item(self) -> None:
        """Add query item."""
        node_name = self.query_node_combo.get()
        state_name = self.query_state_combo.get()

        if not node_name or not state_name:
            messagebox.showwarning("Warning", "Select node and state.")
            return
        if node_name in self.inference_query:
            messagebox.showwarning("Warning", f"Query for '{node_name}' already set.")
            return

        state_index = self.node_data[node_name]['states'].index(state_name)
        self.inference_query[node_name] = state_index
        self.query_listbox.insert(tk.END, f"{node_name} = {state_name}")
        self.query_node_combo.set('')
        self.query_state_combo.set('')
        self.logger.info(f"Query added: {node_name} = {state_name}")

    def clear_query(self) -> None:
        """Clear all queries."""
        self.inference_query.clear()
        self.query_listbox.delete(0, tk.END)

    # === Quantum Circuit Building (Enhanced with Benchmarking) ===
    @PerformanceMonitor.benchmark 
    def build_qbayesian_circuit(self) -> QuantumCircuit:
        """Build quantum circuit for Bayesian Network (binary variables) - Qiskit pattern."""
        # 1) Validate
        if len(self.graph) == 0:
            raise ValueError("Network is empty")

        for node, data in self.node_data.items():
            if data.get('cpt', None) is None:
                raise ValueError(f"CPT for '{node}' is not defined")
            if len(data.get('states', [])) != 2:
                raise NotImplementedError("Only binary nodes are supported by this builder.")

        # 2) Ordering and registers
        sorted_nodes = list(nx.topological_sort(self.graph))
        self.node_name_to_idx = {name: i for i, name in enumerate(sorted_nodes)}
        self.idx_to_node_name = {i: name for name, i in self.node_name_to_idx.items()}

        qr_list = [QuantumRegister(1, name=node_name) for node_name in sorted_nodes]
        qc = QuantumCircuit(*qr_list, name="Bayes_net")

        # Flatten to explicit Qubit objects
        flat_qubits = [qr[0] for qr in qr_list]

        # Helper to get qubit by node name
        def qb(name):
            return flat_qubits[self.node_name_to_idx[name]]

        # 3) Build circuit - QISKIT PATTERN
        for node_idx, node_name in enumerate(sorted_nodes):
            node_states = self.node_data[node_name]['states']
            cpt = self.node_data[node_name]['cpt']
            parents = sorted(list(self.graph.predecessors(node_name)))

            # ROOT NODE: simple Ry rotation
            if not parents:
                prob_1 = cpt[1]
                theta = 2 * np.arcsin(np.sqrt(prob_1))
                qc.ry(theta, flat_qubits[node_idx])
                qc.barrier()
                continue

            # NODE WITH PARENTS: iterate in REVERSE order (all-1s → all-0s)
            parent_indices = [self.node_name_to_idx[p] for p in parents]
            parent_qubits = [flat_qubits[i] for i in parent_indices]
            parent_states_list = [self.node_data[p]['states'] for p in parents]
            parent_combinations = list(itertools.product(*parent_states_list))

            # Loop from LAST combination (all-1s) down to FIRST (all-0s)
            for combo_idx in range(len(parent_combinations) - 1, -1, -1):
                combo = parent_combinations[combo_idx]

                # Apply X gates where parent state is '0'
                # (This converts control from "all-1s" to the current combo)
                flipped = []
                for parent, state in zip(parents, combo):
                    if state == self.node_data[parent]['states'][0]:  # state is '0'
                        qc.x(qb(parent))
                        flipped.append(parent)

                # Get probability and theta for this combination
                start_idx = combo_idx * len(node_states)
                prob_1 = cpt[start_idx + 1]
                theta = 2 * np.arcsin(np.sqrt(prob_1))

                # Apply controlled rotation (cry for 1 parent, mcry for 2+)
                if len(parents) == 1:
                    qc.cry(theta, parent_qubits[0], flat_qubits[node_idx])
                else:
                    try:
                        qc.mcry(theta, parent_qubits, flat_qubits[node_idx])
                    except AttributeError:
                        raise RuntimeError(
                            "mcry not available in this Qiskit version; "
                            "add mcry fallback/decomposition."
                        )

                # Undo X gates (restore original state)
                for parent in reversed(flipped):
                    qc.x(qb(parent))

                qc.barrier()

        return qc

    def build_and_display_circuit(self) -> None:
        """Build and display circuit with error handling."""
        try:
            self.logger.info("Building quantum circuit...")
            self.quantum_circuit = self.build_qbayesian_circuit()
            # use robust helper
            self.show_circuit_mpl(self.quantum_circuit)

            self.status_bar.config(text=f"✓ Circuit built: {self.quantum_circuit.num_qubits} qubits, "
                                     f"{self.quantum_circuit.size()} gates, depth {self.quantum_circuit.depth()}")
            self.viz_notebook.select(self.tab_circuit)
            self.logger.info(f"Circuit built successfully: {self.quantum_circuit.size()} gates")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to build circuit:\n{e}")
            self.logger.error(f"Circuit building failed: {e}", exc_info=True)
            self.status_bar.config(text="Circuit build failed")
    @PerformanceMonitor.benchmark
    def display_circuit(self) -> None:
        """Display circuit diagram."""
        if self.quantum_circuit is None:
            return

        self.circuit_ax.clear()
        self.circuit_ax.axis('off')

        try:
            circuit_drawer(self.quantum_circuit, output='mpl', style='bw', ax=self.circuit_ax,
                           plot_barriers=False, justify='none', fold=-1)
            self.circuit_fig.tight_layout()
            self.circuit_canvas.draw()
            
            # 2. Add this line to ensure the timer captures the full render
            self.root.update() 
            
        except Exception as e:
            self.circuit_ax.text(0.5, 0.5, f"Error:\n{str(e)}",
                                 ha='center', va='center', fontsize=12, color='red',
                                 transform=self.circuit_ax.transAxes)
            self.circuit_canvas.draw()
            self.root.update() # Add here too

    def build_noise_model(self) -> Optional[NoiseModel]:
        """Build noise model based on user selection."""
        selection = self.noise_model_combo.get()

        if selection == 'None (Ideal)':
            return None

        noise_model = NoiseModel()

        if selection == 'Depolarizing (0.1%)':
            error = depolarizing_error(0.001, 1)
            error_2q = depolarizing_error(0.001, 2)
            noise_model.add_all_qubit_quantum_error(error, ['ry'])
            noise_model.add_all_qubit_quantum_error(error_2q, ['cry', 'mcry'])

        elif selection == 'Depolarizing (1%)':
            error = depolarizing_error(0.01, 1)
            error_2q = depolarizing_error(0.01, 2)
            noise_model.add_all_qubit_quantum_error(error, ['ry'])
            noise_model.add_all_qubit_quantum_error(error_2q, ['cry', 'mcry'])

        elif selection == 'Depolarizing (5%)':
            error = depolarizing_error(0.05, 1)
            error_2q = depolarizing_error(0.05, 2)
            noise_model.add_all_qubit_quantum_error(error, ['ry'])
            noise_model.add_all_qubit_quantum_error(error_2q, ['cry', 'mcry'])

        elif selection == 'Thermal Relaxation':
            t1 = 50.0
            t2 = 70.0
            gate_time_1q = 50
            gate_time_2q = 300

            error = thermal_relaxation_error(t1, t2, gate_time_1q / 1000)
            error_2q = thermal_relaxation_error(t1, t2, gate_time_2q / 1000).tensor(
                        thermal_relaxation_error(t1, t2, gate_time_2q / 1000))
            noise_model.add_all_qubit_quantum_error(error, ['ry'])
            noise_model.add_all_qubit_quantum_error(error_2q, ['cry', 'mcry'])

        elif selection == 'Phase Damping':
            error = phase_damping_error(0.01)
            noise_model.add_all_qubit_quantum_error(error, ['ry'])

        elif selection == 'Amplitude Damping':
            error = amplitude_damping_error(0.01)
            noise_model.add_all_qubit_quantum_error(error, ['ry'])

        elif selection == 'Combined (Depol + Thermal)':
            t1 = 50.0
            t2 = 70.0
            gate_time = 50

            depol_error = depolarizing_error(0.01, 1)
            thermal_error = thermal_relaxation_error(t1, t2, gate_time / 1000)
            combined_error = depol_error.compose(thermal_error)
            noise_model.add_all_qubit_quantum_error(combined_error, ['ry'])

        self.logger.info(f"Noise model built: {selection}")
        return noise_model
    @PerformanceMonitor.benchmark
    def run_inference(self) -> None:
        """Run inference using AerSimulator (Shot-based only)."""
        self.logger.info("Starting inference run")

        # Ensure resource monitor exists
        if not hasattr(self, "resource_monitor") or self.resource_monitor is None:
            self.resource_monitor = ResourceMonitor()
        self.resource_monitor.start_monitoring()

        # Prepare UI
        self.results_text.config(state='normal')
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert(tk.END, "▶ Starting inference...\n")
        self.status_bar.config(text="Running inference...")
        self.root.update_idletasks()

        # Basic checks
        if not self.graph:
            self.results_text.insert(tk.END, "✗ Error: Network is empty\n")
            self.status_bar.config(text="Failed: Empty network")
            self._finalize_resource_log()
            self.results_text.config(state='disabled')
            return

        for node, data in self.node_data.items():
            if data['cpt'] is None:
                self.results_text.insert(tk.END, f"✗ Error: CPT for '{node}' not defined\n")
                self.status_bar.config(text="Failed: Missing CPT")
                self._finalize_resource_log()
                self.results_text.config(state='disabled')
                return

        # Parse runs and shots
        try:
            num_runs = int(self.num_runs_entry.get())
            if num_runs <= 0:
                num_runs = 200
        except Exception:
            num_runs = 200

        try:
            shots = int(self.shots_entry.get())
            if shots <= 0:
                raise ValueError
        except Exception:
            self.results_text.insert(tk.END, "Invalid shots value, using default 1024\n")
            shots = 1024
            
        self.results_text.insert(tk.END, f"Mode: Shot-based Sampling ({num_runs} runs, {shots} shots each)\n")

        # Build circuit once (will raise if CPTs are missing)
        self.results_text.insert(tk.END, "Building quantum circuit...\n")
        try:
            self.quantum_circuit = self.build_qbayesian_circuit()
            self.display_circuit()
        except Exception as e:
            self.logger.error(f"Circuit build failed before sampling: {e}", exc_info=True)
            self.results_text.insert(tk.END, f"Error building circuit: {e}\n")
            self.status_bar.config(text="Circuit build failed")
            self._finalize_resource_log()
            self.results_text.config(state='disabled')
            return

        # Build backend (noise) once for this run_inference call
        qbayesian = None
        backend = None

        try:
            noise_model = self.build_noise_model()
            backend = AerSimulator(noise_model=noise_model) if noise_model else AerSimulator()
        except Exception as e:
            self.logger.error("Backend (AerSimulator) init failed: %s", e, exc_info=True)
            self.results_text.insert(tk.END, f"Error initializing backend: {e}\n")
            self._finalize_resource_log()
            self.results_text.config(state='disabled')
            return

        # Create a single BackendSampler with the chosen shots and single QBayesian instance
        try:
            sampler_inst = BackendSampler(backend=backend, options={"shots": shots})
            qbayesian = QBayesian(circuit=self.quantum_circuit, sampler=sampler_inst)
        except Exception as e:
            # If BackendSampler fails for some reason, try fallback dynamic creation inside loop
            self.logger.warning("BackendSampler creation failed upfront: %s. Will create per-iteration.", exc_info=True)
            sampler_inst = None
            qbayesian = None

        # Prepare evidence/query copies
        evidence = dict(self.inference_evidence)
        query = dict(self.inference_query)

        self.results_text.insert(tk.END, f"\nEvidence: {evidence}\n")
        self.results_text.insert(tk.END, f"Query: {query}\n")
        self.results_text.insert(tk.END, "="*50 + "\n")

        # If no query -> rejection_sampling once (for sampling) — keep behavior consistent
        if not query:
            try:
                # If upfront sampler creation failed, create a sampler now (single run for rejection sampling)
                if qbayesian is None:
                    sampler_inst = BackendSampler(backend=backend, options={"shots": shots})
                    qbayesian = QBayesian(circuit=self.quantum_circuit, sampler=sampler_inst)
                
                samples = qbayesian.rejection_sampling(evidence=evidence)

                self.results_text.insert(tk.END, "\n✓ RESULTS\n")
                for state, prob in sorted(samples.items()):
                    self.results_text.insert(tk.END, f"State {state}: {prob:.6f}\n")
                self.last_samples = samples
                self.display_histogram(samples)
                self.viz_notebook.select(self.tab_histogram)

                msg = f"✓ Sampling complete ({shots} shots)"
                self.status_bar.config(text=msg)
                self.results_text.config(state='disabled')
                self._finalize_resource_log()
                return

            except Exception as e:
                self.logger.error(f"Sampling error: {e}", exc_info=True)
                self.results_text.insert(tk.END, f"\nError: {e}\n")
                self.status_bar.config(text="Simulation failed")
                self.results_text.config(state='disabled')
                self._finalize_resource_log()
                return
        # If we have a query: run a single inference and compute Wilson CI (no repeated runs)
        try:
            # Ensure qbayesian instance exists
            if qbayesian is None:
                sampler_inst_local = BackendSampler(backend=backend, options={"shots": shots})
                local_qbayesian = QBayesian(circuit=self.quantum_circuit, sampler=sampler_inst_local)
            else:
                local_qbayesian = qbayesian

            # Single inference call
            result = local_qbayesian.inference(query=query, evidence=evidence)

            # Parse confidence level
            try:
                conf_level = float(self.cilevel_entry.get())
                if not (0 < conf_level < 1):
                    conf_level = 0.95
            except:
                conf_level = 0.95

            # Compute Wilson score interval using shots as effective n
            p = float(result)
            n_eff = float(shots) if shots > 0 else 1.0

           # Compute z critical value (approx); try scipy if available otherwise fallback to 1.96
            try:
                from scipy.stats import norm
                z = norm.ppf(1 - (1 - conf_level) / 2)
                self.logger.info(f"Scipy detected: Confidence Level set to {conf_level}")
            except Exception:
                z = 1.96
                self.logger.warning("Scipy not found: Confidence Level defaulted to 0.95")
                
            d = 1 + (z**2) / n_eff
            c = (p + (z**2) / (2 * n_eff)) / d
            m = (z * np.sqrt((p * (1 - p) / n_eff) + (z**2) / (4 * n_eff**2))) / d
            ci_lower, ci_upper = max(0.0, c - m), min(1.0, c + m)

            # Update UI
            self.results_text.insert(tk.END, f"Result (single run): {p:.6f}\n")
            self.results_text.insert(tk.END, f"{int(conf_level * 100)}% Wilson CI (shots={int(n_eff)}): [{ci_lower:.6f}, {ci_upper:.6f}]\n")

        except Exception as e:
            self.logger.error(f"Inference error: {e}", exc_info=True)
            self.results_text.insert(tk.END, f"\nError during inference: {e}\n")
            self.status_bar.config(text="Inference failed")
            self.results_text.config(state='disabled')
            self._finalize_resource_log()
            return

    def _finalize_resource_log(self):
        """Stop the resource monitor and update the resource log tab safely."""
        self.resource_monitor.stop_monitoring()

        summary = self.resource_monitor.get_summary()
        for snap in self.resource_monitor.resources:
            summary += (
                f"\n{datetime.fromtimestamp(snap['timestamp']).strftime('%H:%M:%S')}"
                f" | CPU: {snap['cpu_percent']:.2f}%"
                f" | MEM: {snap['memory_mb']:.2f} MB"
                f" | Threads: {snap['threads']}"
            )
        self.resource_log_text.config(state='normal')
        self.resource_log_text.delete('1.0', tk.END)
        self.resource_log_text.insert(tk.END, summary)
        self.resource_log_text.config(state='disabled')

    def display_histogram(self, samples: Dict) -> None:
        """Display histogram of results."""
        self.histogram_ax.clear()

        try:
            if samples:
                sorted_samples = sorted(samples.items(), key=lambda x: x[0])
                labels = [str(k) for k, v in sorted_samples]
                values = [v for k, v in sorted_samples]

                bars = self.histogram_ax.bar(labels, values, color='#3498db',
                                             edgecolor='#2874a6', linewidth=1.5)

                for bar in bars:
                    height = bar.get_height()
                    self.histogram_ax.text(bar.get_x() + bar.get_width()/2., height,
                                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)

                self.histogram_ax.set_xlabel('States', fontsize=11)
                self.histogram_ax.set_ylabel('Probability', fontsize=11)
                self.histogram_ax.set_title('Sampling Results', fontsize=13, fontweight='bold')
                self.histogram_ax.tick_params(axis='x', rotation=45)
                self.histogram_ax.grid(axis='y', alpha=0.3)
                self.histogram_fig.tight_layout()
            else:
                self.histogram_ax.text(0.5, 0.5, "No samples",
                                       ha='center', va='center', fontsize=14, color='gray')
                self.histogram_ax.axis('off')

            self.histogram_canvas.draw()
        except Exception as e:
            self.histogram_ax.clear()
            self.histogram_ax.text(0.5, 0.5, f"Error:\n{str(e)}",
                                   ha='center', va='center', fontsize=12, color='red')
            self.histogram_ax.axis('off')
            self.histogram_canvas.draw()

    # === File I/O ===

    def new_network(self, confirm: bool = True) -> None:
        """Create new network."""
        if confirm and self.graph and not messagebox.askyesno("Confirm", "Clear current network?"):
            return

        self.graph.clear()
        self.node_data = {}
        self.node_positions = {}
        self.quantum_circuit = None
        self.cpt_entry_widgets = {}
        self.selected_cpt_node = None
        self.inference_evidence = {}
        self.inference_query = {}
        self.last_samples = None
        self.selected_graph_node = None
        
        # Reset view locks to allow autoscaling for the new network
        self._drag_lock_xlim = None
        self._drag_lock_ylim = None

        self.draw_network()
        self.update_node_lists()

        for widget in self.cpt_scrollable_frame.winfo_children():
            widget.destroy()

        self.evidence_listbox.delete(0, tk.END)
        self.query_listbox.delete(0, tk.END)
        self.results_text.delete('1.0', tk.END)

        self.circuit_ax.clear()
        self.circuit_ax.axis('off')
        self.circuit_ax.text(0.5, 0.5, "Click 'Build Circuit' to generate",
                             ha='center', va='center', fontsize=14, color='gray',
                             transform=self.circuit_ax.transAxes)
        self.circuit_canvas.draw()

        self.histogram_ax.clear()
        self.histogram_ax.text(0.5, 0.5, "Run inference to see results",
                               ha='center', va='center', fontsize=14, color='gray')
        self.histogram_ax.axis('off')
        self.histogram_canvas.draw()

        self.status_bar.config(text="New network - Add nodes and edges to begin")
        self.logger.info("New network created")

    def save_network(self) -> None:
        """Save network to file."""
        if not self.graph:
            messagebox.showwarning("Warning", "Network is empty")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".qbn.json",
            filetypes=[("QBN JSON", "*.qbn.json"), ("All Files", "*.*")]
        )

        if filename:
            self.save_network_to_file(filename)

    def save_network_to_file(self, filename: str) -> None:
        """Save network to file with complete numpy array handling."""
        try:
            import numpy as np

            def convert_numpy(obj):
                """Recursively convert numpy types to Python native types."""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj

            # Convert all data structures
            nodes_serializable = convert_numpy(self.node_data)
            edges_serializable = list(self.graph.edges())

            # Convert positions
            positions_serializable = {}
            for node, pos in self.node_positions.items():
                if isinstance(pos, (tuple, list, np.ndarray)):
                    positions_serializable[node] = [float(pos[0]), float(pos[1])]
                else:
                    positions_serializable[node] = pos

            data = {
                'version': '2.0.1',
                'timestamp': datetime.now().isoformat(),
                'nodes': nodes_serializable,
                'edges': edges_serializable,
                'positions': positions_serializable
            }

            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)

            self.status_bar.config(text=f"✓ Saved to {filename}")
            self.logger.info(f"Network saved to {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{e}")
            self.logger.error(f"Save failed: {e}", exc_info=True)

    def load_network(self) -> None:
        """Load network from file."""
        filename = filedialog.askopenfilename(
            filetypes=[("QBN JSON", "*.qbn.json"), ("All Files", "*.*")]
        )

        if not filename:
            return

        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.new_network(confirm=False)
            self.node_data = data['nodes']
            self.node_positions = data.get('positions', {})

            self.graph.add_nodes_from(self.node_data.keys())
            self.graph.add_edges_from(data['edges'])

            if not self.node_positions:
                self.node_positions = nx.spring_layout(self.graph)
            
            # Reset limits to ensure the loaded network is fully visible initially
            self._drag_lock_xlim = None
            self._drag_lock_ylim = None

            self.draw_network()
            self.update_node_lists()

            self.status_bar.config(text=f"✓ Loaded from {filename}")
            self.logger.info(f"Network loaded from {filename}")
            messagebox.showinfo("Success", "Network loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load:\n{e}")
            self.logger.error(f"Load failed: {e}")

    # === Example Networks ===

    def load_2node_example(self) -> None:
        """Load 2-node example."""
        self.new_network(confirm=False)
        self.graph.add_node("X")
        self.node_data["X"] = {'states': ['0', '1'], 'cpt': [0.8, 0.2]}
        self.graph.add_node("Y")
        self.node_data["Y"] = {'states': ['0', '1'], 'cpt': [0.7, 0.3, 0.1, 0.9]}
        self.graph.add_edge("X", "Y")
        self.node_positions = nx.spring_layout(self.graph)
        self.draw_network()
        self.update_node_lists()
        self.status_bar.config(text="Loaded 2-node example (X → Y)")
        self.logger.info("Loaded 2-node example")

    def load_burglary_example(self) -> None:
        """Load Burglary Alarm example."""
        self.new_network(confirm=False)
        nodes = {
            "B": {'states': ['0', '1'], 'cpt': [0.999, 0.001]},
            "E": {'states': ['0', '1'], 'cpt': [0.998, 0.002]},
            "A": {'states': ['0', '1'], 'cpt': [0.999, 0.001, 0.71, 0.29, 0.06, 0.94, 0.05, 0.95]},
            "J": {'states': ['0', '1'], 'cpt': [0.95, 0.05, 0.1, 0.9]},
            "M": {'states': ['0', '1'], 'cpt': [0.99, 0.01, 0.3, 0.7]}
        }
        for node_name, data in nodes.items():
            self.graph.add_node(node_name)
            self.node_data[node_name] = data
        edges = [("B", "A"), ("E", "A"), ("A", "J"), ("A", "M")]
        for edge in edges:
            self.graph.add_edge(*edge)
        self.node_positions = nx.spring_layout(self.graph, seed=42)
        self.draw_network()
        self.update_node_lists()
        self.status_bar.config(text="Loaded Burglary Alarm example")
        self.logger.info("Loaded Burglary Alarm example")

    # === Network Visualization ===

    def draw_network(self) -> None:
        """Draw network graph with enhanced drag support."""
        self.network_ax.clear()

        # Apply locked limits if they exist (prevents jumping on release)
        if self._drag_lock_xlim and self._drag_lock_ylim:
            self.network_ax.set_xlim(self._drag_lock_xlim)
            self.network_ax.set_ylim(self._drag_lock_ylim)

        if not self.graph:
            self.network_ax.text(0.5, 0.5, "Use 'Network Builder' tab\nto add nodes and edges",
                                 ha='center', va='center', fontsize=14, color='gray')
            self.network_ax.axis('off')
            self.network_canvas.draw()
            return

        if not self.node_positions:
            self.node_positions = nx.spring_layout(self.graph, seed=42)

        node_colors = []
        edge_colors = []
        
        for node in self.graph.nodes():
            # Highlight selected node
            if node == self.selected_graph_node:
                node_colors.append('#3498db') # Blue
            elif self.node_data[node]['cpt'] is None:
                node_colors.append('#e74c3c') # Red
            else:
                node_colors.append('#2ecc71') # Green

        nx.draw_networkx(
            self.graph,
            pos=self.node_positions,
            ax=self.network_ax,
            with_labels=True,
            node_color=node_colors,
            node_size=2000,
            font_size=11,
            font_weight='bold',
            font_color='white',
            edge_color='#34495e',
            arrowstyle='-|>',
            arrowsize=20,
            width=2.5
        )

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='CPT defined'),
            Patch(facecolor='#e74c3c', label='CPT missing'),
            Patch(facecolor='#3498db', label='Selected')
        ]
        self.network_ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

        self.network_ax.axis('off')
        self.network_fig.tight_layout()

        # If dragging, use draw_idle for performance (prevents UI lag)
        # Otherwise use standard draw
        if self._dragging:
            self.network_canvas.draw_idle()
        else:
            self.network_canvas.draw()

    # === Mouse Interaction for Network Graph ===

    def find_node_at_event(self, event):
        """Find node at mouse event position."""
        if not event.xdata or not event.ydata:
            return None

        min_dist_sq = float('inf')
        found_node = None

        for node, (x, y) in self.node_positions.items():
            dist_sq = (event.xdata - x)**2 + (event.ydata - y)**2
            if dist_sq < 0.01:
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    found_node = node

        return found_node
    def on_press(self, event):
        """Handle mouse press."""
        if event.inaxes != self.network_ax:
            return
        self.dragged_node = self.find_node_at_event(event)
        
        # Update selection state
        if self.dragged_node:
            self.selected_graph_node = self.dragged_node
            self.network_canvas.get_tk_widget().config(cursor="hand2")
            self.status_bar.config(text=f"Selected node: {self.dragged_node} (Press Delete to remove)")
            
            # Start drag session
            self._dragging = True
            # Lock the limits to prevent graph auto-scaling while dragging
            # We update this on every press to ensure we lock the CURRENT view
            self._drag_lock_xlim = self.network_ax.get_xlim()
            self._drag_lock_ylim = self.network_ax.get_ylim()
            
            # Auto-select in comboboxes for convenience
            self.cpt_node_combo.set(self.dragged_node)
            self.load_cpt_editor() # Auto-load CPT
        else:
            self.selected_graph_node = None
            
        self.draw_network()

    def on_release(self, event):
        """Handle mouse release."""
        if self.dragged_node:
            self.dragged_node = None
            self._dragging = False
            # Do NOT clear limits here. Persist the view to prevent jumping.
            self.network_canvas.get_tk_widget().config(cursor="")
            self.draw_network()

    def on_motion(self, event):
        """Handle mouse motion with throttling for smooth performance."""
        if self.dragged_node and event.inaxes == self.network_ax:
            # Throttle updates to ~30 FPS to prevent event queue overload
            current_time = time.time()
            if current_time - self._last_drag_time < 0.03:
                return
            self._last_drag_time = current_time

            # Ensure coordinates are valid before updating
            if event.xdata is not None and event.ydata is not None:
                self.node_positions[self.dragged_node] = (event.xdata, event.ydata)
                self.draw_network()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """Main entry point for the application."""
    try:
        logger.debug("="*70)
        logger.debug("Quantum Bayesian Network Builder v2.2 - Enhanced Validation")
        logger.debug("="*70)
        print()
        logger.debug("Initializing application...")

        root = tk.Tk()
        app = QBNApp(root)

        logger.debug("✓ Application initialized successfully")
        print()
        logger.debug("Starting GUI...")
        print()

        root.mainloop()

    except (ImportError, ModuleNotFoundError) as e:
        logger.debug(f"\n✗ Error: Missing required library")
        logger.debug(f"  {e}")
        print()
        logger.debug("Please install required packages:")
        logger.debug("  pip install qiskit qiskit-aer qiskit-machine-learning")
        logger.debug("  pip install networkx matplotlib psutil")
        print()
    except Exception as e:
        logger.debug(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()