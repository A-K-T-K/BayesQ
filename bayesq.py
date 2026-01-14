"""
BayesQ - Quantum Bayesian Network Research Platform
===================================================

A comprehensive framework for the design, validation, and quantum simulation of 
Bayesian Networks. This platform facilitates the translation of classical probabilistic 
graphical models into quantum circuits, enabling the study of quantum inference 
algorithms and noise resilience.

Key Capabilities:
- Interactive Directed Acyclic Graph (DAG) Construction
- Automated Quantum Circuit Synthesis via Qiskit
- Stochastic Noise Simulation & Resource Profiling
- Real-time Topological and Probabilistic Validation
- Posterior Inference via Quantum Rejection Sampling

License: MIT

"""

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
import threading

# --- Matplotlib Visualization Library Dependencies ---
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import matplotlib.pyplot as plt
except ImportError:
    print("Critical Dependency Error: Matplotlib is missing. Please execute: pip install matplotlib")
    exit()

# --- Network Analysis Library Dependencies ---
try:
    import networkx as nx
except ImportError:
    print("Critical Dependency Error: NetworkX is missing. Please execute: pip install networkx")
    exit()

# --- Quantum Computing SDK Dependencies (Qiskit) ---
try:
    from qiskit import QuantumCircuit, QuantumRegister, transpile
    from qiskit.visualization import plot_histogram, circuit_drawer
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (NoiseModel, depolarizing_error, 
                                  thermal_relaxation_error, phase_damping_error,
                                  amplitude_damping_error)
except ImportError:
    print("Critical Dependency Error: Qiskit is missing. Please execute: pip install qiskit qiskit-aer")
    exit()


# ============================================================================
# USER INTERFACE AUXILIARY COMPONENTS
# ============================================================================

class CreateToolTip(object):
    """
    Implements a context-aware tooltip mechanism for UI widgets.
    
    This class manages the scheduling and rendering of transient information windows
    triggered by mouse interaction events.
    """
    def __init__(self, widget, text='widget info'):
        """Initialize the tooltip association with a specific widget."""
        self.waittime = 500     # Latency in milliseconds before display
        self.wraplength = 250   # Maximum width in pixels before text wrapping
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        """Handle the mouse-enter event to initiate the tooltip display schedule."""
        self.schedule()

    def leave(self, event=None):
        """Handle the mouse-leave event to cancel or destroy the tooltip."""
        self.unschedule()
        self.hidetip()

    def schedule(self):
        """Schedule the tooltip display event after the defined latency period."""
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        """Cancel any pending tooltip display events."""
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        """Render the tooltip window at the cursor's current coordinates."""
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background="#ffffe0", relief='solid', borderwidth=1,
                       wraplength = self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        """Destroy the tooltip window if it exists."""
        tw = self.tw
        self.tw= None
        if tw:
            tw.destroy()

# ============================================================================
# SYSTEM TELEMETRY AND LOGGING
# ============================================================================

class TextWidgetHandler(logging.Handler):
    """
    A custom logging handler designed to redirect log records to a Tkinter Text widget.
    
    This facilitates real-time display of system events within the application GUI.
    """
    def __init__(self, text_widget):
        """Initialize the handler with the target Text widget."""
        logging.Handler.__init__(self)
        self.text_widget = text_widget
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))

    def emit(self, record):
        """Process a log record and append it to the widget in a thread-safe manner."""
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
    """
    Facilitates the real-time monitoring and profiling of system resources.
    
    Tracks CPU utilization, memory consumption, and thread count to assess
    the computational overhead of quantum simulations.
    """
    def __init__(self):
        """Initialize the process monitor interface."""
        self.process = psutil.Process()
        self.monitoring = False
        self.resources = []
        self.start_time = None

    def start_monitoring(self):
        """Commence the data collection cycle."""
        self.monitoring = True
        self.resources = []
        self.start_time = time.time()
        self.record()

    def stop_monitoring(self):
        """Terminate the data collection cycle and finalize records."""
        self.monitoring = False
        self.record()

    def get_snapshot(self):
        """Capture an instantaneous snapshot of system metrics."""
        mem_info = self.process.memory_info()
        return {
            'timestamp': time.time(), 
            'cpu_percent': self.process.cpu_percent(interval=0),
            'memory_mb': mem_info.rss / 1024 / 1024, 
            'threads': self.process.num_threads()
        }

    def record(self):
        """Append the current system snapshot to the historical record."""
        self.resources.append(self.get_snapshot())

    def get_summary(self):
        """Generate a statistical summary of resource usage over the monitoring period."""
        if len(self.resources) < 2: 
            return "Insufficient monitoring data for statistical analysis."
        
        elapsed = self.resources[-1]['timestamp'] - self.resources[0]['timestamp']
        mem_vals = [r['memory_mb'] for r in self.resources]
        cpu_vals = [r['cpu_percent'] for r in self.resources if r['cpu_percent'] > 0]
        
        return (f"=== RESOURCE UTILIZATION SUMMARY ===\n"
                f"Duration: {elapsed:.3f}s | Peak Memory: {max(mem_vals):.2f} MB | Mean Memory: {sum(mem_vals)/len(mem_vals):.2f} MB\n"
                f"Peak CPU: {max(cpu_vals) if cpu_vals else 0:.1f}% | Mean CPU: {sum(cpu_vals)/len(cpu_vals) if cpu_vals else 0:.1f}% | Active Threads: {self.resources[-1]['threads']}")


# ============================================================================
# ARCHITECTURAL UTILITIES
# ============================================================================

class ConfigManager:
    """
    Manages persistent application configuration parameters.
    
    Handles serialization and deserialization of user preferences and simulation
    settings via INI files.
    """

    def __init__(self, config_file: str = 'bayesq_config.ini'):
        """Initialize the configuration parser."""
        self.config = configparser.ConfigParser()
        self.config_file = config_file
        self.load_config()

    def load_config(self) -> None:
        """Load configuration from disk or initialize defaults if the file is absent."""
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
        else:
            self.create_default_config()

    def create_default_config(self) -> None:
        """Establish default parameters for the application runtime."""
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
        """Persist current configuration state to disk."""
        with open(self.config_file, 'w') as f:
            self.config.write(f)

    def get(self, section: str, key: str, fallback: Any = None) -> str:
        """Retrieve a configuration value with fallback support."""
        return self.config.get(section, key, fallback=fallback)

    def set(self, section: str, key: str, value: str) -> None:
        """Update a configuration value and persist changes."""
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, key, value)
        self.save_config()


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Initialize the global logging infrastructure."""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f'bayesq_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger('BayesQ')
    logger.info(f"Logging subsystem initialized at {log_level} level")
    return logger


class PerformanceMonitor:
    """
    A decorator-based utility for benchmarking function execution metrics.
    """
    def __init__(self):
        self.metrics: List = []
        self.logger = logging.getLogger('BayesQ.Performance')

    @staticmethod
    def benchmark(func):
        """
        Decorator to measure execution time and memory delta of the target function.
        """
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

            logger = logging.getLogger('BayesQ.Performance')
            logger.info(f"Method {func.__name__}: Duration={execution_time:.4f}s, Memory Delta={memory_used:.2f} MB")

            return result
        return wrapper


class NetworkValidator:
    """
    Enforces topological consistency and probabilistic integrity constraints.
    
    Ensures that the graph structure remains a Directed Acyclic Graph (DAG) and 
    that Conditional Probability Tables (CPTs) represent valid probability distributions.
    """
    def __init__(self, max_nodes: int = 20, max_parents: int = 5):
        """Define validation constraints."""
        self.max_nodes = max_nodes
        self.max_parents = max_parents
        self.logger = logging.getLogger('BayesQ.Validator')

    def validate_network_structure(self, graph: nx.DiGraph) -> Tuple[bool, str]:
        """Verify graph topology against DAG constraints and connectivity rules."""
        if len(graph.nodes()) == 0:
             return False, "Network topology is empty."

        if len(graph.nodes()) > self.max_nodes:
            return False, f"Network complexity exceeds limit: {len(graph.nodes())} nodes (maximum {self.max_nodes})."

        if not nx.is_directed_acyclic_graph(graph):
            try:
                cycle = nx.find_cycle(graph, orientation='original')
                cycle_str = " -> ".join([f"{u}->{v}" for u, v, d in cycle])
                return False, f"Cyclic dependency detected: {cycle_str}. Structure must be a DAG."
            except:
                return False, "Cyclic dependencies detected - topology must be a DAG."

        if len(graph.nodes()) > 1:
            if not nx.is_weakly_connected(graph):
                isolated_nodes = [n for n in graph.nodes() if graph.degree(n) == 0]
                if isolated_nodes:
                    return False, f"Disconnected nodes detected: {', '.join(isolated_nodes)}. The graph must be fully connected."
                else:
                    return False, "Graph fragmentation detected. Ensure all components are weakly connected."

        for node in graph.nodes():
            parents = list(graph.predecessors(node))
            if len(parents) > self.max_parents:
                return False, f"Node '{node}' exceeds parent cardinality limit: {len(parents)} (maximum {self.max_parents})."

        return True, ""

    def validate_node_name(self, name: str, existing_nodes: List[str]) -> Tuple[bool, str]:
        """Validate node nomenclature against formatting rules and uniqueness constraints."""
        if not name or not name.strip():
            return False, "Node identifier cannot be null or empty."
        if name in existing_nodes:
            return False, f"Node identifier '{name}' is not unique."
        if not name.replace('_', '').isalnum():
            return False, "Node identifier contains invalid characters (alphanumeric and underscore allowed)."
        if len(name) > 20:
            return False, "Node identifier exceeds maximum length (20 characters)."
        return True, ""

    def validate_cpt(self, cpt: List[float], num_combinations: int, num_states: int = 2) -> Tuple[bool, str]:
        """Verify the stochastic validity of the Conditional Probability Table."""
        expected_length = num_combinations * num_states
        if len(cpt) != expected_length:
            return False, f"CPT dimensionality mismatch: expected {expected_length}, received {len(cpt)}."
        for prob in cpt:
            if not isinstance(prob, (int, float)):
                return False, f"Invalid data type in probability distribution: {type(prob)}."
            if prob < 0 or prob > 1:
                return False, f"Probability value out of valid range [0,1]: {prob}."
        for i in range(0, len(cpt), num_states):
            prob_sum = sum(cpt[i:i+num_states])
            if not np.isclose(prob_sum, 1.0, atol=1e-6):
                return False, f"Probability axiom violation: Sum at index {i} is {prob_sum}, expected 1.0."
        return True, ""

    def validate_states(self, states_str: str) -> Tuple[bool, List[str], str]:
        """Parse and validate state definitions."""
        if not states_str or not states_str.strip():
            return False, [], "State definitions cannot be empty."
        states = [s.strip() for s in states_str.split(',')]
        if len(states) < 2:
            return False, [], "Variable must possess at least 2 distinct states."
        if len(states) > 2:
            self.logger.warning("Cardinality > 2 detected. Current version restricts variables to binary states; truncating.")
            states = states[:2]
        if len(set(states)) != len(states):
            return False, [], "Duplicate state identifiers detected."
        return True, states, ""


class ExportManager:
    """
    Handles data serialization and export operations for interoperability.
    
    Supports exporting quantum circuits to QASM, network structures to DOT,
    and inference results to CSV formats.
    """
    def __init__(self):
        self.logger = logging.getLogger('BayesQ.Export')

    def export_circuit_to_qasm(self, circuit: QuantumCircuit, filename: str) -> bool:
        """Serialize the quantum circuit to OpenQASM format."""
        try:
            from qiskit import qasm2
            qasm_str = qasm2.dumps(circuit)
            with open(filename, 'w') as f:
                f.write(qasm_str)
            self.logger.info(f"Successfully exported QASM artifact to {filename}")
            return True
        except ImportError:
            try:
                # Fallback to legacy QASM method
                qasm_str = circuit.qasm()
                with open(filename, 'w') as f:
                    f.write(qasm_str)
                self.logger.info(f"Successfully exported QASM artifact to {filename}")
                return True
            except AttributeError as e:
                self.logger.error(f"QASM export protocol unsupported: {e}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to serialize circuit to QASM: {e}")
            return False

    def export_network_to_dot(self, graph: nx.DiGraph, filename: str) -> bool:
        """Serialize the graph topology to Graphviz DOT format."""
        try:
            with open(filename, 'w') as f:
                f.write("digraph BayesianNetwork {\n")
                for node in graph.nodes():
                    f.write(f'  "{node}";\n')
                for edge in graph.edges():
                    f.write(f'  "{edge[0]}" -> "{edge[1]}";\n')
                f.write("}\n")
            self.logger.info(f"Successfully exported DOT artifact to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to serialize graph to DOT: {e}")
            return False

    def export_results_to_csv(self, samples: Dict[Any, float], filename: str) -> bool:
        """Serialize inference results to Comma-Separated Values format."""
        try:
            import csv
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['State', 'Probability'])
                for state, prob in sorted(samples.items()):
                    writer.writerow([state, prob])
            self.logger.info(f"Successfully exported result dataset to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to serialize results to CSV: {e}")
            return False


# ============================================================================
# MAIN APPLICATION CONTROLLER
# ============================================================================

class QBNApp:
    """
    The central controller class for the BayesQ application.
    
    This class orchestrates the interaction between the GUI frontend, the graph 
    theoretical backend (NetworkX), and the quantum simulation engine (Qiskit).
    It manages application state, user inputs, and the execution lifecycle of 
    quantum inference tasks.
    """

    def __init__(self, root: tk.Tk):
        """Initialize the application state and graphical interface."""
        self.root = root
        self.root.title("BayesQ")
        self.root.geometry("1600x900")

        # Initialize configuration subsystem
        self.config = ConfigManager()

        # Initialize logging subsystem
        log_level = self.config.get('DEFAULT', 'log_level', fallback='INFO')
        self.logger = setup_logging(log_level)
        self.logger.info("="*60)
        self.logger.info("BayesQ Application Initialization Sequence Initiated")
        self.logger.info("="*60)

        # Initialize validators and auxiliary managers
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
        
        # Selection state management
        self.selected_graph_node: Optional[str] = None

        # Quantum Circuit Object Management
        self.node_name_to_idx: Dict[str, int] = {}
        self.idx_to_node_name: Dict[int, str] = {}
        self.quantum_circuit: Optional[QuantumCircuit] = None
        self.last_samples: Optional[Dict] = None

        # Drag-and-Drop Interaction State
        self.dragged_node: Optional[str] = None
        self._dragging: bool = False
        self._last_drag_time: float = 0.0
        self._drag_lock_xlim: Optional[Tuple[float, float]] = None
        self._drag_lock_ylim: Optional[Tuple[float, float]] = None

        # Configure GUI components
        self.setup_style()
        self.create_menu()
        
        # Construct Main Layout Architecture
        self.main_paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_paned_window.pack(fill=tk.BOTH, expand=True)

        self.controls_frame = ttk.Frame(self.main_paned_window, width=450)
        self.main_paned_window.add(self.controls_frame, weight=1)
        self.create_controls_panel(self.controls_frame)

        self.vis_frame = ttk.Frame(self.main_paned_window)
        self.main_paned_window.add(self.vis_frame, weight=3)
        self.create_visualization_panel(self.vis_frame)

        # Initialize Status Bar
        self.status_bar = ttk.Label(root, text="System Ready - Initiate new network via Ctrl+N",
                                    relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Initialize Event Bindings
        self.setup_shortcuts() 

        # Establish shutdown protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Request focus for immediate keyboard interaction
        self.root.focus_force()

        self.logger.info("Graphical User Interface successfully initialized.")

    def setup_style(self) -> None:
        """Configure the ttk widget styling theme."""
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
        """Register global keyboard accelerators for common operations."""
        def create_callback(name, func):
            def callback(event=None):
                func()
                return "break" # Prevent event propagation
            return callback

        shortcuts = {
            '<Control-n>': ('New', lambda: self.new_network()),
            '<Control-o>': ('Open', lambda: self.load_network()),
            '<Control-s>': ('Save', lambda: self.save_network()),
            '<Control-b>': ('Build', lambda: self.build_and_display_circuit()),
            '<Control-r>': ('Run', lambda: self.run_inference()),
            '<Control-e>': ('Export', lambda: self.export_circuit_qasm()),
            '<Delete>': ('Delete', self.delete_selected_node_shortcut),
            # Case-insensitive variants
            '<Control-N>': ('New', lambda: self.new_network()),
            '<Control-O>': ('Open', lambda: self.load_network()),
            '<Control-S>': ('Save', lambda: self.save_network()),
            '<Control-B>': ('Build', lambda: self.build_and_display_circuit()),
            '<Control-R>': ('Run', lambda: self.run_inference()),
            '<Control-E>': ('Export', lambda: self.export_circuit_qasm()),
        }

        # 1. Bind to Root Window
        for key, (name, func) in shortcuts.items():
            cb = create_callback(name, func)
            self.root.bind(key, cb)
            self.root.bind_all(key, cb)

        # 2. Bind to Canvas Widgets (explicit focus handling)
        if hasattr(self, 'network_canvas'):
            canvas_widget = self.network_canvas.get_tk_widget()
            canvas_widget.config(takefocus=1)
            for key, (name, func) in shortcuts.items():
                canvas_widget.bind(key, create_callback(name, func))

        if hasattr(self, 'circuit_canvas'):
            circuit_widget = self.circuit_canvas.get_tk_widget()
            circuit_widget.config(takefocus=1)
            for key, (name, func) in shortcuts.items():
                circuit_widget.bind(key, create_callback(name, func))

    def delete_selected_node_shortcut(self, event=None):
        """Invoke node deletion routine via keyboard interaction."""
        focused_widget = self.root.focus_get()
        if isinstance(focused_widget, (tk.Entry, tk.Text, ttk.Entry)):
            return

        if self.selected_graph_node:
            self.delete_node_combo.set(self.selected_graph_node)
            self.delete_node()
        else:
            self.status_bar.config(text="Deletion failed: No node selected.")

    def create_menu(self) -> None:
        """Construct the application menu hierarchy."""
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        # File Operations Menu
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

        # Execution Menu
        run_menu = tk.Menu(self.menu_bar, tearoff=0)
        run_menu.add_command(label="Validate Network", command=self.validate_network_ui)
        run_menu.add_command(label="Build Circuit", command=self.build_and_display_circuit, accelerator="Ctrl+B")
        run_menu.add_command(label="Run Inference", command=self.run_inference, accelerator="Ctrl+R")
        self.menu_bar.add_cascade(label="Run", menu=run_menu)

        # Analysis Tools Menu
        tools_menu = tk.Menu(self.menu_bar, tearoff=0)
        tools_menu.add_command(label="Network Statistics", command=self.show_network_statistics)
        tools_menu.add_command(label="Settings", command=self.show_settings)
        self.menu_bar.add_cascade(label="Tools", menu=tools_menu)

        # Help & Information Menu
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        help_menu.add_command(label="About", command=self.show_about)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)

    def show_about(self) -> None:
        """Display the application attribution dialog."""
        about_text = """BayesQ

A platform for designing and simulating Quantum Bayesian Networks.

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
• Manual Rejection Sampling Implementation
• Quantum circuit visualization  
• Multiple noise models
• Performance monitoring
• Strict graph validation

© 2025 - MIT License"""
        messagebox.showinfo("About BayesQ", about_text)

    def show_user_guide(self) -> None:
        """Display the integrated user documentation."""
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
        """Display the configuration settings dialog."""
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
                messagebox.showinfo("Success", "Settings persisted. Restart required for changes to take effect.")
                settings_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save settings: {e}")

        button_frame = ttk.Frame(settings_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(button_frame, text="Save", command=save_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.RIGHT, padx=5)

    def show_network_statistics(self) -> None:
        """Calculate and display topological metrics of the network."""
        if not self.graph.nodes():
            messagebox.showinfo("Network Statistics", "Network is empty.")
            return

        num_nodes = len(self.graph.nodes())
        num_edges = len(self.graph.edges())
        root_nodes = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        nodes_with_cpt = sum(1 for n in self.node_data.values() if n['cpt'] is not None)

        stats_text = f"""NETWORK TOPOLOGY STATISTICS

Structure:
  Nodes (Variables): {num_nodes}
  Edges (Dependencies): {num_edges}
  Root Nodes: {len(root_nodes)}

Parameters:
  Defined CPTs: {nodes_with_cpt}/{num_nodes}

Constraints:
  Is DAG: {nx.is_directed_acyclic_graph(self.graph)}
  Is Weakly Connected: {nx.is_weakly_connected(self.graph)}"""

        if self.quantum_circuit:
            stats_text += f"""

Quantum Circuit Metrics:
  Qubits: {self.quantum_circuit.num_qubits}
  Gate Count: {self.quantum_circuit.size()}
  Circuit Depth: {self.quantum_circuit.depth()}"""

        messagebox.showinfo("Network Statistics", stats_text)

    def validate_network_ui(self) -> None:
        """Trigger comprehensive network validation and report status."""
        is_valid, error_msg = self.validator.validate_network_structure(self.graph)

        if not is_valid:
            messagebox.showerror("Validation Failed", error_msg)
            return

        missing_cpts = [n for n, data in self.node_data.items() if data['cpt'] is None]

        if missing_cpts:
            messagebox.showwarning("Validation", 
                                   f"Parameter specification incomplete. CPTs missing for: {', '.join(missing_cpts)}")
            return

        for node_name, node_info in self.node_data.items():
            parents = list(self.graph.predecessors(node_name))
            parent_states_list = [self.node_data[p]['states'] for p in parents]
            num_combinations = len(list(itertools.product(*parent_states_list))) if parents else 1

            is_valid, error_msg = self.validator.validate_cpt(node_info['cpt'], num_combinations)

            if not is_valid:
                messagebox.showerror("Validation Failed", 
                                     f"CPT validation error at node '{node_name}': {error_msg}")
                return

        messagebox.showinfo("Validation", "✓ Network integrity confirmed.\n\nAll topological and probabilistic constraints satisfied.")
        self.logger.info("Network validation routine passed successfully.")

    def on_closing(self) -> None:
        """Handle the application shutdown sequence."""
        if self.config.get('DEFAULT', 'auto_save') == 'True' and self.graph.nodes():
            try:
                auto_save_path = 'autosave.qbn.json'
                self.save_network_to_file(auto_save_path)
                self.logger.info(f"State automatically persisted to {auto_save_path}")
            except Exception as e:
                self.logger.error(f"Auto-save procedure failed: {e}")

        self.logger.info("Application shutdown sequence initiated.")
        self.root.destroy()

    def export_circuit_qasm(self) -> None:
        """Initiate the export of the quantum circuit to OpenQASM."""
        if not self.quantum_circuit:
            messagebox.showwarning("Warning", "Circuit synthesis required before export.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".qasm",
            filetypes=[("QASM files", "*.qasm"), ("All files", "*.*")]
        )

        if filename:
            if self.exporter.export_circuit_to_qasm(self.quantum_circuit, filename):
                messagebox.showinfo("Success", f"Circuit successfully serialized to {filename}")

    def export_network_dot(self) -> None:
        """Initiate the export of the network topology to DOT format."""
        if not self.graph.nodes():
            messagebox.showwarning("Warning", "Network topology is empty.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".dot",
            filetypes=[("DOT files", "*.dot"), ("All files", "*.*")]
        )

        if filename:
            if self.exporter.export_network_to_dot(self.graph, filename):
                messagebox.showinfo("Success", f"Network successfully serialized to {filename}")

    def export_results_csv(self) -> None:
        """Initiate the export of inference results to CSV."""
        if not self.last_samples:
            messagebox.showwarning("Warning", "Stochastic sampling required before export.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            if self.exporter.export_results_to_csv(self.last_samples, filename):
                messagebox.showinfo("Success", f"Data successfully serialized to {filename}")

    def create_controls_panel(self, parent: ttk.Frame) -> None:
        """Instantiate and organize the primary control tab widgets."""
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
        """Construct the network structure definition interface."""
        info_frame = ttk.LabelFrame(parent, text="Instructions")
        info_frame.pack(fill=tk.X, padx=10, pady=10)

        info_text = ("1. Define nodes (random variables) with binary state space\n"
                     "2. Establish directed dependencies (edges)\n"
                     "3. Specify Conditional Probability Tables (CPTs)\n"
                     "4. Synthesize circuit and execute inference")
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack(padx=10, pady=10)

        # Node Definition Section
        node_frame = ttk.LabelFrame(parent, text="Add Node")
        node_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(node_frame, text="Name:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.node_name_entry = ttk.Entry(node_frame)
        self.node_name_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        CreateToolTip(self.node_name_entry, "Specify a unique identifier for the random variable (e.g., 'Rain', 'Alarm').")

        ttk.Label(node_frame, text="States:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.node_states_entry = ttk.Entry(node_frame)
        self.node_states_entry.insert(0, "0,1")
        self.node_states_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        CreateToolTip(self.node_states_entry, "Comma-delimited state labels (e.g., '0,1' or 'False,True'). Default: 0,1.")

        add_btn = ttk.Button(node_frame, text="Add Node", command=self.add_node)
        add_btn.grid(row=2, column=0, columnspan=2, pady=10)
        CreateToolTip(add_btn, "Instantiate a new node within the graph.")
        
        node_frame.columnconfigure(1, weight=1)

        # Edge Definition Section
        edge_frame = ttk.LabelFrame(parent, text="Add Edge (Parent → Child)")
        edge_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(edge_frame, text="Source (Parent):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.edge_from_combo = ttk.Combobox(edge_frame, state="readonly")
        self.edge_from_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(edge_frame, text="Target (Child):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.edge_to_combo = ttk.Combobox(edge_frame, state="readonly")
        self.edge_to_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

        edge_btn = ttk.Button(edge_frame, text="Add Edge", command=self.add_edge)
        edge_btn.grid(row=2, column=0, columnspan=2, pady=10)
        CreateToolTip(edge_btn, "Establish a directed probabilistic dependency between nodes.")
        
        edge_frame.columnconfigure(1, weight=1)

        # Node Removal Section
        delete_frame = ttk.LabelFrame(parent, text="Deletion Operations")
        delete_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(delete_frame, text="Node:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.delete_node_combo = ttk.Combobox(delete_frame, state="readonly")
        self.delete_node_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        del_btn = ttk.Button(delete_frame, text="Delete Node", command=self.delete_node)
        del_btn.grid(row=0, column=2, padx=5, pady=5)
        CreateToolTip(del_btn, "Remove the selected node and incident edges (Shortcut: Delete Key)")
        
        delete_frame.columnconfigure(1, weight=1)

    def create_cpt_tab(self, parent: ttk.Frame) -> None:
        """Construct the Conditional Probability Table editor interface."""
        selector_frame = ttk.Frame(parent)
        selector_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(selector_frame, text="Select Variable:").pack(side=tk.LEFT, padx=5)
        self.cpt_node_combo = ttk.Combobox(selector_frame, state="readonly")
        self.cpt_node_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.cpt_node_combo.bind("<<ComboboxSelected>>", self.load_cpt_editor)

        cpt_outer_frame = ttk.LabelFrame(parent, text="Probability Distribution Definition")
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

        self.save_cpt_button = ttk.Button(parent, text="Commit CPT", command=self.save_cpt)
        self.save_cpt_button.pack(pady=10)
        self.save_cpt_button.config(state=tk.DISABLED)
        CreateToolTip(self.save_cpt_button, "Validate and persist the probability distribution for the selected variable.")

    def create_inference_tab(self, parent: ttk.Frame) -> None:
        """Construct the inference configuration and execution interface."""
        # Simulation Parameters
        exec_frame = ttk.LabelFrame(parent, text="Simulation Parameters")
        exec_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(exec_frame, text="Shot Count:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.shots_entry = ttk.Entry(exec_frame, width=12)
        self.shots_entry.insert(0, self.config.get('DEFAULT', 'shots', fallback='1024'))
        self.shots_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        CreateToolTip(self.shots_entry, "Define the number of measurement repetitions for statistical sampling.")
        
        ttk.Label(exec_frame, text="(AerSimulator parameter)", font=('Arial', 9, 'italic'),
                  foreground='gray').grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        exec_frame.columnconfigure(1, weight=1)

        # Noise Model Configuration
        ttk.Label(exec_frame, text="Noise Model:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
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
        self.noise_model_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        CreateToolTip(self.noise_model_combo, "Simulate decoherence and gate errors to assess algorithm robustness.")

        ttk.Label(exec_frame, text="(AerSimulator parameter)", font=('Arial', 9, 'italic'),
                  foreground='gray').grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Label(exec_frame, text="Monte Carlo Iterations:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.num_runs_entry = ttk.Entry(exec_frame, width=12)
        self.num_runs_entry.insert(0, "200")  # Default iteration count
        self.num_runs_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        CreateToolTip(self.num_runs_entry, "Number of independent experiments to establish statistical confidence.")

        ttk.Label(exec_frame, text="Confidence Level (alpha):").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.cilevel_entry = ttk.Entry(exec_frame, width=12)
        self.cilevel_entry.insert(0, "0.95")
        self.cilevel_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)

        # Evidence Definition Section
        evidence_frame = ttk.LabelFrame(parent, text="Evidence (Observed Variables)")
        evidence_frame.pack(fill=tk.X, padx=10, pady=10)

        add_evidence_frame = ttk.Frame(evidence_frame)
        add_evidence_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(add_evidence_frame, text="Variable:").pack(side=tk.LEFT, padx=(0, 5))
        self.evidence_node_combo = ttk.Combobox(add_evidence_frame, state="readonly", width=12)
        self.evidence_node_combo.pack(side=tk.LEFT, padx=5)
        self.evidence_node_combo.bind("<<ComboboxSelected>>", self.on_evidence_node_select)

        ttk.Label(add_evidence_frame, text="State:").pack(side=tk.LEFT, padx=5)
        self.evidence_state_combo = ttk.Combobox(add_evidence_frame, state="readonly", width=8)
        self.evidence_state_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(add_evidence_frame, text="Assert", command=self.add_evidence_item, 
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

        ttk.Button(evidence_frame, text="Reset Evidence", 
                   command=self.clear_evidence).pack(pady=5)

        # Query Definition Section  
        query_frame = ttk.LabelFrame(parent, text="Query (Target Distribution)")
        query_frame.pack(fill=tk.X, padx=10, pady=10)

        add_query_frame = ttk.Frame(query_frame)
        add_query_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(add_query_frame, text="Variable:").pack(side=tk.LEFT, padx=(0, 5))
        self.query_node_combo = ttk.Combobox(add_query_frame, state="readonly", width=12)
        self.query_node_combo.pack(side=tk.LEFT, padx=5)
        self.query_node_combo.bind("<<ComboboxSelected>>", self.on_query_node_select)

        ttk.Label(add_query_frame, text="State:").pack(side=tk.LEFT, padx=5)
        self.query_state_combo = ttk.Combobox(add_query_frame, state="readonly", width=8)
        self.query_state_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(add_query_frame, text="Set Target", command=self.add_query_item, 
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

        ttk.Button(query_frame, text="Reset Query", command=self.clear_query).pack(pady=5)

        inf_btn = ttk.Button(parent, text="▶ Execute Inference", command=self.run_inference)
        inf_btn.pack(pady=15)
        CreateToolTip(inf_btn, "Synthesize circuit and perform stochastic simulation (Shortcut: Ctrl+R)")

        results_frame = ttk.LabelFrame(parent, text="Inference Output")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=8,
                                                      font=("Courier New", 9), bg="white", fg="black")
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


    def create_visualization_panel(self, parent: ttk.Frame) -> None:
        """Construct the multi-tab visualization and analytics panel."""
        self.viz_notebook = ttk.Notebook(parent)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_network = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.tab_network, text="Topology")
        self.create_network_viz(self.tab_network)

        self.tab_circuit = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.tab_circuit, text="Quantum Circuit")
        self.create_circuit_viz(self.tab_circuit)

        self.tab_histogram = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.tab_histogram, text="Posterior Distribution")
        self.create_histogram_viz(self.tab_histogram)

        self.tab_code = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.tab_code, text="Synthesized Code")
        self.create_code_viz(self.tab_code)

        # Confidence Interval Analysis Tab
        self.tab_ci_hist = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.tab_ci_hist, text="Error Analysis")
        self.ci_hist_fig = Figure(figsize=(8, 6), dpi=100)
        self.ci_hist_ax = self.ci_hist_fig.add_subplot(111)
        self.ci_hist_canvas = FigureCanvasTkAgg(self.ci_hist_fig, master=self.tab_ci_hist)
        self.ci_hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Resource Telemetry Tab
        self.tab_resource_log = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.tab_resource_log, text="System Telemetry")
        self.resource_log_text = scrolledtext.ScrolledText(self.tab_resource_log, wrap=tk.WORD, height=16, state="disabled")
        self.resource_log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Execution Logs Tab
        self.tab_execution_log = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.tab_execution_log, text="Event Log")
        self.exec_log_text = scrolledtext.ScrolledText(self.tab_execution_log, wrap=tk.WORD, height=16, state="disabled")
        self.exec_log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # === CONNECT LOGGER TO GUI ===
        text_handler = TextWidgetHandler(self.exec_log_text)
        self.logger.addHandler(text_handler)
        self.logger.info("Event logging subsystem linked to GUI display.")

    def create_network_viz(self, parent: ttk.Frame) -> None:
        """Initialize the graph topology visualization canvas."""
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
        """Initialize the quantum circuit schematic visualization canvas."""
        self.circuit_fig = Figure(figsize=(10, 6), dpi=100)
        self.circuit_fig.patch.set_facecolor('#f0f0f0')
        self.circuit_ax = self.circuit_fig.add_subplot(111)
        self.circuit_ax.set_facecolor('#ffffff')
        self.circuit_ax.axis('off')
        self.circuit_ax.text(0.5, 0.5, "Initiate 'Build Circuit' (Ctrl+B) to generate schematic",
                             ha='center', va='center', fontsize=14, color='gray',
                             transform=self.circuit_ax.transAxes)

        self.circuit_canvas = FigureCanvasTkAgg(self.circuit_fig, master=parent)
        self.circuit_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.circuit_canvas.draw()

        self.circuit_toolbar = NavigationToolbar2Tk(self.circuit_canvas, parent, pack_toolbar=False)
        self.circuit_toolbar.update()
        self.circuit_toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_histogram_viz(self, parent: ttk.Frame) -> None:
        """Initialize the statistical distribution visualization canvas."""
        self.histogram_fig = Figure(figsize=(8, 6), dpi=100)
        self.histogram_fig.patch.set_facecolor('#f0f0f0')
        self.histogram_ax = self.histogram_fig.add_subplot(111)
        self.histogram_ax.set_facecolor('#ffffff')
        self.histogram_ax.text(0.5, 0.5, "Execute inference to visualize distribution",
                               ha='center', va='center', fontsize=14, color='gray')
        self.histogram_ax.axis('off')

        self.histogram_canvas = FigureCanvasTkAgg(self.histogram_fig, master=parent)
        self.histogram_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.histogram_canvas.draw()

        self.histogram_toolbar = NavigationToolbar2Tk(self.histogram_canvas, parent, pack_toolbar=False)
        self.histogram_toolbar.update()
        self.histogram_toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_code_viz(self, parent):
        """Construct the code generation and display interface."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Generate Script", command=self.gen_code).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Copy to Clipboard", command=self.copy_code).pack(side=tk.LEFT, padx=5)

        self.code_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=('Courier', 10), bg='#f8f8f8')
        self.code_text.pack(fill=tk.BOTH, expand=True)
        self.code_text.insert('1.0', '# Execute inference, then invoke script generation.')

    def gen_code(self):
        """Synthesize a complete, executable Python script representing the current simulation state."""
        if not self.quantum_circuit:
            messagebox.showwarning("Warning", "Inference execution required prior to code generation.")
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
            L.append("# BayesQ - Auto-Synthesized Simulation Script")
            L.append("# ================================================\n")

            L.append("from qiskit import QuantumCircuit, QuantumRegister, transpile")
            L.append("from qiskit_aer import AerSimulator")
            
            if noise_selection != 'None (Ideal)':
                L.append("from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error")
                
            L.append("import numpy as np")
            L.append("import itertools")
            L.append("import networkx as nx\n")

            L.append(f"# Topology: Nodes={list(self.node_data.keys())}")
            L.append(f"# Topology: Edges={list(self.graph.edges())}\n")

            L.append("# Parameter Definitions (Conditional Probability Tables)")
            L.append("node_data = {")
            for n, info in self.node_data.items():
                L.append(f"    '{n}': {{")
                L.append(f"        'states': {info['states']},")
                L.append(f"        'cpt': {info['cpt']}")
                L.append("    },")
            L.append("}\n")

            L.append("# Graph Construction")
            L.append("graph = nx.DiGraph()")
            L.append(f"graph.add_nodes_from({list(self.node_data.keys())})")
            L.append(f"graph.add_edges_from({list(self.graph.edges())})\n")

            nodes = list(self.node_name_to_idx.keys())
            L.append(f"sorted_nodes = {nodes}")
            L.append(f"node_name_to_idx = {dict(self.node_name_to_idx)}\n")

            L.append("# ========== Quantum Circuit Synthesis ==========")
            L.append("def build_circuit():")
            L.append('    """Construct the quantum circuit implementing the Bayesian Network."""')
            L.append("    qr = [QuantumRegister(1, name=n) for n in sorted_nodes]")
            L.append("    qc = QuantumCircuit(*qr)\n")

            L.append("    for i, node in enumerate(sorted_nodes):")
            L.append("        states = node_data[node]['states']")
            L.append("        cpt = node_data[node]['cpt']")
            L.append("        parents = sorted(list(graph.predecessors(node)))\n")

            L.append("        if not parents:")
            L.append("            # Root Node Configuration")
            L.append("            prob = cpt[1]")
            L.append("            theta = 2 * np.arcsin(np.sqrt(prob))")
            L.append("            qc.ry(theta, i)")
            L.append("            qc.barrier()\n")

            L.append("        else:")
            L.append("            # Dependent Node Configuration")
            L.append("            p_idx = [node_name_to_idx[p] for p in parents]")
            L.append("            p_states = [node_data[p]['states'] for p in parents]")
            L.append("            combos = list(itertools.product(*p_states))\n")

            L.append("            # Rotation logic for specific control states")
            L.append("            idx = (len(combos)-1) * len(states) + 1")
            L.append("            theta = 2 * np.arcsin(np.sqrt(cpt[idx]))")
            L.append("            if len(parents) == 1:")
            L.append("                qc.cry(theta, p_idx[0], i)")
            L.append("            else:")
            L.append("                qc.mcry(theta, p_idx, i)")
            L.append("            qc.barrier()\n")

            L.append("            # Iterative Multi-Controlled Rotations")
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
            L.append("print(f'Circuit Metrics: {qc.num_qubits} qubits, {qc.size()} gates')\n")

            L.append("# ========== Backend and Noise Configuration ==========")
            L.append(f"shots = {shots}")
            
            if noise_selection != 'None (Ideal)':
                L.append(f"# Active Noise Model: {noise_selection}")
                L.append("noise_model = NoiseModel()")
                if 'Depolarizing' in noise_selection:
                    L.append("# ... (Specific noise configuration generated in app) ...")
                    L.append("backend = AerSimulator(noise_model=noise_model)")
                else:
                    L.append("backend = AerSimulator(noise_model=noise_model)")
            else:
                L.append("backend = AerSimulator()")
            
            L.append("# ========== Inference Specification ==========")
            if self.inference_evidence:
                L.append("# Observational Evidence")
                L.append("evidence = {")
                for node_name, state_idx in self.inference_evidence.items():
                    L.append(f"    '{node_name}': {state_idx},")
                L.append("}")
            else:
                L.append("evidence = {}")

            if self.inference_query:
                L.append("query = {")
                for node_name, state_idx in self.inference_query.items():
                    L.append(f"    '{node_name}': {state_idx},")
                L.append("}")
            else:
                L.append("query = {}")

            L.append("\n# ========== Rejection Sampling Implementation ==========")
            L.append("def run_rejection_sampling(circuit, backend, evidence, query, shots):")
            L.append("    # 1. Measurement Operations")
            L.append("    meas_qc = circuit.copy()")
            L.append("    meas_qc.measure_all()")
            L.append("    ")
            L.append("    # 2. Execution")
            L.append("    t_qc = transpile(meas_qc, backend)")
            L.append("    result = backend.run(t_qc, shots=shots).result()")
            L.append("    counts = result.get_counts()")
            L.append("    ")
            L.append("    total_accepted = 0")
            L.append("    query_hits = 0")
            L.append("    accepted_counts = {}")
            L.append("    ")
            L.append("    # 3. Post-selection Filtering")
            L.append("    for bitstring, count in counts.items():")
            L.append("        # Little-Endian decoding: q0 is rightmost")
            L.append("        consistent = True")
            L.append("        for node, state_idx in evidence.items():")
            L.append("            q_idx = node_name_to_idx[node]")
            L.append("            bit_char = bitstring[-(q_idx+1)]")
            L.append("            if int(bit_char) != state_idx:")
            L.append("                consistent = False")
            L.append("                break")
            L.append("        ")
            L.append("        if consistent:")
            L.append("            total_accepted += count")
            L.append("            accepted_counts[bitstring] = accepted_counts.get(bitstring, 0) + count")
            L.append("            ")
            L.append("            if query:")
            L.append("                match_query = True")
            L.append("                for node, state_idx in query.items():")
            L.append("                    q_idx = node_name_to_idx[node]")
            L.append("                    if int(bitstring[-(q_idx+1)]) != state_idx:")
            L.append("                        match_query = False")
            L.append("                        break")
            L.append("                if match_query:")
            L.append("                    query_hits += count")
            L.append("    ")
            L.append("    if total_accepted == 0: return None, None")
            L.append("    ")
            L.append("    prob = query_hits / total_accepted if query else None")
            L.append("    return prob, accepted_counts")
            L.append("")
            L.append("# Execution")
            L.append("prob, samples = run_rejection_sampling(qc, backend, evidence, query, shots)")
            L.append("if query:")
            L.append("    print(f'P(Query|Evidence) = {prob:.6f}')")
            L.append("else:")
            L.append("    print('Joint Distribution (Top 10):')")
            L.append("    for bit, count in sorted(samples.items(), key=lambda x: x[1], reverse=True)[:10]:")
            L.append("        print(f'{bit}: {count}')")

            self.code_text.insert('1.0', "\n".join(L))
            self.status_bar.config(text="✓ Script synthesized successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Generation failed:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def copy_code(self) -> None:
        """Transfer generated code to the system clipboard."""
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.code_text.get('1.0', tk.END))
            self.status_bar.config(text="✓ Script copied to clipboard")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def add_node(self) -> None:
        """Insert a new node into the graph with validation checks."""
        name = self.node_name_entry.get().strip()
        states_str = self.node_states_entry.get().strip()

        # Validate node identifier
        is_valid, error_msg = self.validator.validate_node_name(name, list(self.graph.nodes()))
        if not is_valid:
            messagebox.showerror("Validation Error", error_msg)
            self.logger.warning(f"Node insertion rejected: {error_msg}")
            return

        # Validate state definitions
        is_valid, states, error_msg = self.validator.validate_states(states_str)
        if not is_valid:
            messagebox.showerror("Validation Error", error_msg)
            self.logger.warning(f"State definition rejected: {error_msg}")
            return

        # Commit node addition
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
        self.status_bar.config(text=f"Added variable: {name}")
        self.logger.info(f"Node instantiated: {name} with state space {states}")

    def add_edge(self) -> None:
        """Establish a directed edge between two existing nodes, checking for cycles."""
        from_node = self.edge_from_combo.get()
        to_node = self.edge_to_combo.get()

        if not from_node or not to_node:
            messagebox.showerror("Error", "Source and Target nodes must be specified.")
            return
        if from_node == to_node:
            messagebox.showerror("Error", "Self-referential edges (loops) are prohibited.")
            return
        if self.graph.has_edge(from_node, to_node):
            messagebox.showerror("Error", f"Dependency already exists.")
            return

        # Tentatively add edge to inspect topology
        self.graph.add_edge(from_node, to_node)

        # Validate acyclic property
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(from_node, to_node)
            messagebox.showerror("Validation Error", "Edge creation rejected: Cycle detected.")
            return

        self.node_data[to_node]['cpt'] = None
        self.draw_network()
        self.status_bar.config(text=f"Dependency established: {from_node} → {to_node}")
        self.logger.info(f"Edge instantiated: {from_node} → {to_node}")

        # Automatically transition to CPT editor
        self.cpt_node_combo.set(to_node)
        self.load_cpt_editor()

    def delete_node(self) -> None:
        """Remove a node and its incident edges from the graph."""
        node = self.delete_node_combo.get()
        if not node:
            messagebox.showwarning("Warning", "Selection required for deletion.")
            return

        if messagebox.askyesno("Confirm", f"Irreversibly delete node '{node}' and associated dependencies?"):
            self.graph.remove_node(node)
            del self.node_data[node]
            if node in self.node_positions:
                del self.node_positions[node]
            self.selected_graph_node = None # Clear selection state
            self.draw_network()
            self.update_node_lists()
            self.status_bar.config(text=f"Node removed: {node}")
            self.logger.info(f"Node deletion committed: {node}")

    def update_node_lists(self) -> None:
        """Refresh all UI dropdown lists to reflect the current graph state."""
        node_names = sorted(list(self.graph.nodes()))
        self.edge_from_combo['values'] = node_names
        self.edge_to_combo['values'] = node_names
        self.cpt_node_combo['values'] = node_names
        self.evidence_node_combo['values'] = node_names
        self.query_node_combo['values'] = node_names
        self.delete_node_combo['values'] = node_names

    # === CPT Editor Logic ===

    def load_cpt_editor(self, event=None) -> None:
        """Populate the CPT editor interface for the selected variable."""
        self.selected_cpt_node = self.cpt_node_combo.get()
        if not self.selected_cpt_node:
            return

        for widget in self.cpt_scrollable_frame.winfo_children():
            widget.destroy()
        self.cpt_entry_widgets = {}

        node_name = self.selected_cpt_node
        node_states = self.node_data[node_name]['states']
        parents = sorted(list(self.graph.predecessors(node_name)))

        # Header Generation
        ttk.Label(self.cpt_scrollable_frame, text="Parent Configuration",
                  font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.cpt_scrollable_frame, text=f"P({node_name}={node_states[1]})",
                  font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(self.cpt_scrollable_frame, text=f"P({node_name}={node_states[0]})",
                  font=('Arial', 9, 'italic'), foreground='gray').grid(row=0, column=2, padx=5, pady=5)

        existing_cpt = self.node_data[node_name].get('cpt', [])

        if not parents:
            # Root Node Handling
            ttk.Label(self.cpt_scrollable_frame, text="(Root Priors)").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
            entry = ttk.Entry(self.cpt_scrollable_frame, width=10)
            if existing_cpt:
                entry.insert(0, str(existing_cpt[1]))
            entry.grid(row=1, column=1, padx=5, pady=5)
            auto_label = ttk.Label(self.cpt_scrollable_frame, text="(complement)", foreground='gray')
            auto_label.grid(row=1, column=2, padx=5, pady=5)
            self.cpt_entry_widgets['root'] = entry
        else:
            # Dependent Node Handling
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
                auto_label = ttk.Label(self.cpt_scrollable_frame, text="(complement)", foreground='gray')
                auto_label.grid(row=i + 1, column=2, padx=5, pady=5)
                self.cpt_entry_widgets[combo] = entry

        self.save_cpt_button.config(state=tk.NORMAL)
        self.status_bar.config(text=f"Editing parameters for '{node_name}'")

    def save_cpt(self) -> None:
        """Validate and persist CPT data to the internal model."""
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
                    raise ValueError(f"Probability domain violation: {prob_1} (Must be [0,1])")
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
                        raise ValueError(f"Probability domain violation for '{combo_str}': {prob_1}")
                    prob_0 = 1.0 - prob_1
                    probabilities.extend([prob_0, prob_1])

            # Validation Call
            num_combinations = len(parent_combinations) if parents else 1
            is_valid, error_msg = self.validator.validate_cpt(probabilities, num_combinations)

            if not is_valid:
                messagebox.showerror("Validation Error", error_msg)
                return

            self.node_data[node_name]['cpt'] = probabilities
            self.draw_network()
            self.status_bar.config(text=f"✓ Parameters persisted for '{node_name}'")
            self.logger.info(f"CPT updated for {node_name}")
            messagebox.showinfo("Success", f"Conditional parameters for '{node_name}' successfully updated.")

        except ValueError as e:
            messagebox.showerror("Error", str(e))
            self.logger.error(f"CPT persistence failure: {e}")

    # === Evidence and Query Management ===

    def on_evidence_node_select(self, event=None) -> None:
        """Update available states when an evidence node is selected."""
        node_name = self.evidence_node_combo.get()
        if node_name:
            states = self.node_data[node_name]['states']
            self.evidence_state_combo['values'] = states
            self.evidence_state_combo.set(states[0])

    def add_evidence_item(self) -> None:
        """Register a new evidence assertion."""
        node_name = self.evidence_node_combo.get()
        state_name = self.evidence_state_combo.get()

        if not node_name or not state_name:
            messagebox.showwarning("Warning", "Variable and State must be specified.")
            return
        if node_name in self.inference_evidence:
            messagebox.showwarning("Warning", f"Conflicting assertion for '{node_name}'.")
            return

        state_index = self.node_data[node_name]['states'].index(state_name)
        self.inference_evidence[node_name] = state_index
        self.evidence_listbox.insert(tk.END, f"{node_name} = {state_name}")
        self.evidence_node_combo.set('')
        self.evidence_state_combo.set('')
        self.logger.info(f"Evidence asserted: {node_name} = {state_name}")

    def clear_evidence(self) -> None:
        """Reset all evidence assertions."""
        self.inference_evidence.clear()
        self.evidence_listbox.delete(0, tk.END)

    def on_query_node_select(self, event=None) -> None:
        """Update available states when a query node is selected."""
        node_name = self.query_node_combo.get()
        if node_name:
            states = self.node_data[node_name]['states']
            self.query_state_combo['values'] = states
            self.query_state_combo.set(states[0])

    def add_query_item(self) -> None:
        """Register a new query target."""
        node_name = self.query_node_combo.get()
        state_name = self.query_state_combo.get()

        if not node_name or not state_name:
            messagebox.showwarning("Warning", "Variable and State must be specified.")
            return
        if node_name in self.inference_query:
            messagebox.showwarning("Warning", f"Duplicate query target for '{node_name}'.")
            return

        state_index = self.node_data[node_name]['states'].index(state_name)
        self.inference_query[node_name] = state_index
        self.query_listbox.insert(tk.END, f"{node_name} = {state_name}")
        self.query_node_combo.set('')
        self.query_state_combo.set('')
        self.logger.info(f"Query target defined: {node_name} = {state_name}")

    def clear_query(self) -> None:
        """Reset all query targets."""
        self.inference_query.clear()
        self.query_listbox.delete(0, tk.END)

    # === Quantum Circuit Synthesis ===

    @PerformanceMonitor.benchmark
    def build_qbayesian_circuit(self) -> QuantumCircuit:
        """
        Constructs the quantum circuit representation of the Bayesian Network.
        
        Implements the mapping of conditional probabilities to quantum rotation gates ($R_y$)
        and controlled rotation gates ($C R_y$).
        """
        if not self.graph:
            raise ValueError("Graph structure is undefined.")

        for node, data in self.node_data.items():
            if data['cpt'] is None:
                raise ValueError(f"Parameter definition missing for variable '{node}'.")

        sorted_nodes = list(nx.topological_sort(self.graph))
        self.node_name_to_idx = {name: i for i, name in enumerate(sorted_nodes)}
        self.idx_to_node_name = {i: name for name, i in self.node_name_to_idx.items()}

        qr_list = [QuantumRegister(1, name=node_name) for node_name in sorted_nodes]
        qc = QuantumCircuit(*qr_list, name="Bayes_net")

        for node_idx, node_name in enumerate(sorted_nodes):
            node_states = self.node_data[node_name]['states']
            cpt = self.node_data[node_name]['cpt']
            parents = sorted(list(self.graph.predecessors(node_name)))

            if not parents:
                # Root Node Processing
                prob_1 = cpt[1]
                theta = 2 * np.arcsin(np.sqrt(prob_1))
                qc.ry(theta, node_idx)
            else:
                # Dependent Node Processing
                parent_indices = [self.node_name_to_idx[p] for p in parents]
                parent_states_list = [self.node_data[p]['states'] for p in parents]
                parent_combinations = list(itertools.product(*parent_states_list))

                # Logic: Encode probability for the all-1s parent state first
                all_ones_idx = len(parent_combinations) - 1
                start_idx = all_ones_idx * len(node_states)
                prob_1 = cpt[start_idx + 1]
                theta = 2 * np.arcsin(np.sqrt(prob_1))

                if len(parents) == 1:
                    qc.cry(theta, parent_indices[0], node_idx)
                else:
                    qc.mcry(theta, parent_indices, node_idx)

                # Gray Code logic or similar traversal to handle other states
                # (Simplified here to iterate and flip bits via X gates)
                for combo_idx in range(len(parent_combinations) - 2, -1, -1):
                    combo = parent_combinations[combo_idx]

                    # Activate control state
                    for parent, state in zip(parents, combo):
                        if state == self.node_data[parent]['states'][0]:
                            qc.x(self.node_name_to_idx[parent])

                    start_idx = combo_idx * len(node_states)
                    prob_1 = cpt[start_idx + 1]
                    theta = 2 * np.arcsin(np.sqrt(prob_1))

                    if len(parents) == 1:
                        qc.cry(theta, parent_indices[0], node_idx)
                    else:
                        qc.mcry(theta, parent_indices, node_idx)

                    # Deactivate control state (Uncompute)
                    for parent, state in zip(parents, combo):
                        if state == self.node_data[parent]['states'][0]:
                            qc.x(self.node_name_to_idx[parent])

        return qc

    def build_and_display_circuit(self) -> None:
        """Orchestrate circuit construction and visualization update."""
        try:
            self.logger.info("Initiating quantum circuit synthesis...")
            self.quantum_circuit = self.build_qbayesian_circuit()
            self.display_circuit()
            self.status_bar.config(text=f"✓ Circuit Synthesized: {self.quantum_circuit.num_qubits} Qubits, "
                                     f"{self.quantum_circuit.size()} Gates, Depth {self.quantum_circuit.depth()}")
            self.viz_notebook.select(self.tab_circuit)
            self.logger.info(f"Circuit synthesis complete: {self.quantum_circuit.size()} gates")
        except Exception as e:
            messagebox.showerror("Error", f"Circuit synthesis failure:\n{e}")
            self.logger.error(f"Circuit synthesis exception: {e}", exc_info=True)
            self.status_bar.config(text="Circuit generation failed")

    @PerformanceMonitor.benchmark
    def display_circuit(self) -> None:
        """Render the circuit schematic to the matplotlib canvas."""
        if self.quantum_circuit is None:
            return

        self.circuit_ax.clear()
        self.circuit_ax.axis('off')

        try:
            circuit_drawer(self.quantum_circuit, output='mpl', style='bw', ax=self.circuit_ax,
                           plot_barriers=False, justify='none', fold=-1)
            self.circuit_fig.tight_layout()
            self.circuit_canvas.draw()
            
            self.root.update() 
            
        except Exception as e:
            self.circuit_ax.text(0.5, 0.5, f"Visualization Error:\n{str(e)}",
                                 ha='center', va='center', fontsize=12, color='red',
                                 transform=self.circuit_ax.transAxes)
            self.circuit_canvas.draw()
            self.root.update()

    def build_noise_model(self) -> Optional[NoiseModel]:
        """Construct the error model configuration based on user selection."""
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

        self.logger.info(f"Noise model instantiation: {selection}")
        return noise_model

    def _execute_rejection_sampling(self, circuit: QuantumCircuit, backend, evidence: Dict[str, int], query: Dict[str, int], shots: int):
        """
        Implementation of the Rejection Sampling algorithm for quantum posterior inference.
        
        This method executes the quantum circuit, filters measurement outcomes based on observed
        evidence, and calculates conditional probabilities.
        """
        # 1. Measurement Injection
        # Assumes one-to-one mapping between qubits and graph nodes via self.node_name_to_idx
        meas_qc = circuit.copy()
        meas_qc.measure_all()

        # 2. Simulation Execution
        try:
            t_qc = transpile(meas_qc, backend)
            job = backend.run(t_qc, shots=shots)
            result = job.result()
            counts = result.get_counts()
        except Exception as e:
            raise RuntimeError(f"Backend execution failure: {e}")

        total_accepted = 0
        query_hits = 0
        accepted_counts = {}

        # 3. Post-Selection / Filtering Logic
        for bitstring, count in counts.items():
            # Qiskit uses Little-Endian bit ordering (qn...q0)
            # Node X at index i corresponds to bit at index: len - 1 - i
            
            is_consistent = True
            
            # Evidence verification
            for node_name, required_state in evidence.items():
                q_idx = self.node_name_to_idx[node_name]
                bit_char = bitstring[-(q_idx + 1)]
                if int(bit_char) != required_state:
                    is_consistent = False
                    break
            
            if is_consistent:
                total_accepted += count
                accepted_counts[bitstring] = accepted_counts.get(bitstring, 0) + count

                # Query verification
                if query:
                    matches_query = True
                    for node_name, required_state in query.items():
                        q_idx = self.node_name_to_idx[node_name]
                        bit_char = bitstring[-(q_idx + 1)]
                        if int(bit_char) != required_state:
                            matches_query = False
                            break
                    if matches_query:
                        query_hits += count

        # 4. Probability Computation
        if total_accepted == 0:
            return None, {}, 0

        # Calculate P(Query|Evidence)
        probability = query_hits / total_accepted if query else None
        
        # Normalize counts for histogram visualization
        normalized_counts = {k: v/total_accepted for k,v in accepted_counts.items()}
        
        return probability, normalized_counts, total_accepted

    @PerformanceMonitor.benchmark
    def run_inference(self) -> None:
        """
        Orchestrate the quantum inference workflow.
        
        Coordinates resource monitoring, circuit synthesis, backend configuration,
        and statistical sampling routines. Supports Wilson Score Interval calculation
        for confidence estimation.
        """
        self.logger.info("Initializing multi-run inference sequence for empirical distribution analysis.")

        if not hasattr(self, "resource_monitor") or self.resource_monitor is None:
            self.resource_monitor = ResourceMonitor()
        self.resource_monitor.start_monitoring()

        self.results_text.config(state='normal')
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert(tk.END, "▶ Initiating inference protocol...\n")
        self.status_bar.config(text="Executing inference protocol...")
        self.root.update_idletasks()

        if not self.graph:
            self.results_text.insert(tk.END, "✗ Error: Network topology undefined\n")
            self.status_bar.config(text="Failure: Empty topology")
            self._finalize_resource_log()
            self.results_text.config(state='disabled')
            return

        for node, data in self.node_data.items():
            if data['cpt'] is None:
                self.results_text.insert(tk.END, f"✗ Error: Parameter missing for node '{node}'\n")
                self.status_bar.config(text="Failure: Incomplete parameters")
                self._finalize_resource_log()
                self.results_text.config(state='disabled')
                return

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
            self.results_text.insert(tk.END, "Invalid shot count; defaulting to 1024.\n")
            shots = 1024

        self.results_text.insert(tk.END, f"Configuration: {num_runs} iterations, {shots} shots per iteration.\n")

        self.results_text.insert(tk.END, "Synthesizing quantum circuit...\n")
        self.quantum_circuit = self.build_qbayesian_circuit()
        self.display_circuit()

        noise_model = self.build_noise_model()
        backend = AerSimulator(noise_model=noise_model) if noise_model else AerSimulator()

        evidence = dict(self.inference_evidence)
        query = dict(self.inference_query)
        self.results_text.insert(tk.END, f"\nEvidence Set: {evidence}\n")
        self.results_text.insert(tk.END, f"Query Target: {query}\n")
        self.results_text.insert(tk.END, "="*50 + "\n")

        # CASE 1: Joint Distribution Sampling (No specific query)
        if not query:
            self.results_text.insert(tk.END, "Executing Rejection Sampling (Joint Distribution Analysis)...\n")
            
            prob, samples, total_accepted = self._execute_rejection_sampling(self.quantum_circuit, backend, evidence, query, shots)
            
            if samples is None:
                self.results_text.insert(tk.END, "\n✗ No samples consistent with evidence set.\n")
            else:
                self.results_text.insert(tk.END, "\n✓ SAMPLING RESULTS (Top 10 States)\n")
                for state, p in sorted(samples.items(), key=lambda x: x[1], reverse=True)[:10]:
                    self.results_text.insert(tk.END, f"State {state}: {p:.6f}\n")
                
                self.last_samples = samples # Cache for export
                self.display_histogram(samples)
                self.viz_notebook.select(self.tab_histogram)
            
            self.status_bar.config(text=f"✓ Sampling protocol complete")
            self.results_text.config(state='disabled')
            self._finalize_resource_log()
            return

        # CASE 2: Posterior Inference with Confidence Intervals
        inference_data = [] # Stores tuple (probability, valid_samples)

        def run_multiple():
            """Background thread for iterative sampling."""
            try:
                for _ in range(num_runs):
                    prob, _, n_acc = self._execute_rejection_sampling(self.quantum_circuit, backend, evidence, query, shots)
                    if prob is not None:
                        inference_data.append((prob, n_acc))
            except Exception as e:
                self.logger.error(f"Inference execution error: {e}", exc_info=True)
                self.root.after(0, lambda: self.results_text.insert(tk.END, f"\nRuntime Error: {e}\n"))
                return

            if not inference_data:
                self.root.after(0, lambda: self.results_text.insert(tk.END, "\nConvergence Failure: No valid samples obtained.\n"))
                self.root.after(0, self._finalize_resource_log)
                return

            # Statistical Aggregation
            total_N = sum(n for _, n in inference_data)
            total_hits = sum(p * n for p, n in inference_data)
            
            if total_N == 0:
                 mean_result = 0.0
            else:
                 mean_result = total_hits / total_N

            try:
                conf_level = float(self.cilevel_entry.get())
                if not (0 < conf_level < 1):
                    conf_level = 0.95
            except Exception:
                conf_level = 0.95

            # Z-score approximation
            if conf_level >= 0.99: z = 2.576
            elif conf_level >= 0.95: z = 1.96
            elif conf_level >= 0.90: z = 1.645
            else: z = 1.96

            # Wilson Score Interval Calculation
            p_hat = mean_result
            n = total_N
            
            denominator = 1 + (z**2) / n
            center_adjusted_probability = (p_hat + (z**2) / (2 * n)) / denominator
            error_margin = (z / denominator) * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2) / (4 * n**2))
            
            ci_lower = max(0.0, center_adjusted_probability - error_margin)
            ci_upper = min(1.0, center_adjusted_probability + error_margin)

            def update_ui():
                """Update GUI with statistical results."""
                self.results_text.insert(tk.END, f"\nMean Posterior Probability ({len(inference_data)} successful runs): {mean_result:.6f}\n")
                self.results_text.insert(tk.END, f"{int(conf_level * 100)}% Confidence Interval (Wilson): [{ci_lower:.6f}, {ci_upper:.6f}]\n")

                # Empirical Distribution Visualization
                results_array = np.array([p for p, _ in inference_data])
                
                self.ci_hist_ax.clear()
                self.ci_hist_ax.hist(results_array, bins=30, color="#3498db", edgecolor="#2874a6", alpha=0.7)
                self.ci_hist_ax.axvline(mean_result, color="green", linestyle="--", label=f"Mean: {mean_result:.4f}")
                self.ci_hist_ax.axvline(ci_lower, color="red", linestyle="-", label=f"Lower: {ci_lower:.4f}")
                self.ci_hist_ax.axvline(ci_upper, color="red", linestyle="-", label=f"Upper: {ci_upper:.4f}")
                self.ci_hist_ax.legend()
                self.ci_hist_ax.set_title(f"Empirical Probability Distribution ({num_runs} iterations)")
                self.ci_hist_canvas.draw_idle()
                self.viz_notebook.select(self.tab_ci_hist)

                self.status_bar.config(text=f"✓ Protocol completed ({num_runs} iterations)")
                self.results_text.config(state='disabled')
                self._finalize_resource_log()

            self.root.after(0, update_ui)

        threading.Thread(target=run_multiple, daemon=True).start()

    def _finalize_resource_log(self):
        """Terminate the resource monitor and display the final telemetry summary."""
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
        """Render the histogram of sampling results."""
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

                self.histogram_ax.set_xlabel('States (Bitstrings)', fontsize=11)
                self.histogram_ax.set_ylabel('Probability P(State)', fontsize=11)
                self.histogram_ax.set_title('Posterior Distribution', fontsize=13, fontweight='bold')
                self.histogram_ax.tick_params(axis='x', rotation=45)
                self.histogram_ax.grid(axis='y', alpha=0.3)
                self.histogram_fig.tight_layout()
            else:
                self.histogram_ax.text(0.5, 0.5, "Null Dataset",
                                       ha='center', va='center', fontsize=14, color='gray')
                self.histogram_ax.axis('off')

            self.histogram_canvas.draw()
        except Exception as e:
            self.histogram_ax.clear()
            self.histogram_ax.text(0.5, 0.5, f"Rendering Error:\n{str(e)}",
                                   ha='center', va='center', fontsize=12, color='red')
            self.histogram_ax.axis('off')
            self.histogram_canvas.draw()

    # === File System I/O ===

    def new_network(self, confirm: bool = True) -> None:
        """Reset the application state to initialize a new network model."""
        if confirm and self.graph and not messagebox.askyesno("Confirm", "Discard current network model?"):
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
        self.circuit_ax.text(0.5, 0.5, "Awaiting circuit generation...",
                             ha='center', va='center', fontsize=14, color='gray',
                             transform=self.circuit_ax.transAxes)
        self.circuit_canvas.draw()

        self.histogram_ax.clear()
        self.histogram_ax.text(0.5, 0.5, "Awaiting inference results...",
                               ha='center', va='center', fontsize=14, color='gray')
        self.histogram_ax.axis('off')
        self.histogram_canvas.draw()

        self.status_bar.config(text="Workspace initialized")
        self.logger.info("New network model initialized")

    def save_network(self) -> None:
        """Persist the current network model to the file system."""
        if not self.graph:
            messagebox.showwarning("Warning", "Network model is empty.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".qbn.json",
            filetypes=[("QBN JSON", "*.qbn.json"), ("All Files", "*.*")]
        )

        if filename:
            self.save_network_to_file(filename)

    def save_network_to_file(self, filename: str) -> None:
        """Serialize the network model to a JSON file, handling Numpy data types."""
        try:
            import numpy as np

            def convert_numpy(obj):
                """Recursively convert numpy types to Python native types for JSON serialization."""
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

            # Convert data structures
            nodes_serializable = convert_numpy(self.node_data)
            edges_serializable = list(self.graph.edges())

            # Convert spatial coordinates
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

            self.status_bar.config(text=f"✓ State persisted to {filename}")
            self.logger.info(f"Network serialized to {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Save operation failed:\n{e}")
            self.logger.error(f"Serialization exception: {e}", exc_info=True)

    def load_network(self) -> None:
        """Reconstruct the network model from a JSON file."""
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

            self.status_bar.config(text=f"✓ Model loaded from {filename}")
            self.logger.info(f"Network deserialized from {filename}")
            messagebox.showinfo("Success", "Network model reconstruction successful.")
        except Exception as e:
            messagebox.showerror("Error", f"Load operation failed:\n{e}")
            self.logger.error(f"Deserialization exception: {e}")

    # === Example Models ===

    def load_2node_example(self) -> None:
        """Instantiate the standard 2-node benchmark model."""
        self.new_network(confirm=False)
        self.graph.add_node("X")
        self.node_data["X"] = {'states': ['0', '1'], 'cpt': [0.8, 0.2]}
        self.graph.add_node("Y")
        self.node_data["Y"] = {'states': ['0', '1'], 'cpt': [0.7, 0.3, 0.1, 0.9]}
        self.graph.add_edge("X", "Y")
        self.node_positions = nx.spring_layout(self.graph)
        self.draw_network()
        self.update_node_lists()
        self.status_bar.config(text="Loaded 2-node benchmark (X → Y)")
        self.logger.info("2-node benchmark initialized")

    def load_burglary_example(self) -> None:
        """Instantiate the classic 'Burglary Alarm' Bayesian Network."""
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
        self.status_bar.config(text="Loaded Burglary Alarm topology")
        self.logger.info("Burglary Alarm topology initialized")

    # === Graph Visualization Logic ===

    def draw_network(self) -> None:
        """Render the network topology using NetworkX and Matplotlib."""
        self.network_ax.clear()

        # Apply view-lock to maintain stability during drag operations
        if self._drag_lock_xlim and self._drag_lock_ylim:
            self.network_ax.set_xlim(self._drag_lock_xlim)
            self.network_ax.set_ylim(self._drag_lock_ylim)

        if not self.graph:
            self.network_ax.text(0.5, 0.5, "Utilize 'Network Builder' tab\nto define topology",
                                 ha='center', va='center', fontsize=14, color='gray')
            self.network_ax.axis('off')
            self.network_canvas.draw()
            return

        if not self.node_positions:
            self.node_positions = nx.spring_layout(self.graph, seed=42)

        node_colors = []
        
        for node in self.graph.nodes():
            # State-dependent coloration
            if node == self.selected_graph_node:
                node_colors.append('#3498db') # Selected state (Blue)
            elif self.node_data[node]['cpt'] is None:
                node_colors.append('#e74c3c') # Incomplete state (Red)
            else:
                node_colors.append('#2ecc71') # Valid state (Green)

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
            Patch(facecolor='#2ecc71', label='Parameters Defined'),
            Patch(facecolor='#e74c3c', label='Parameters Missing'),
            Patch(facecolor='#3498db', label='Active Selection')
        ]
        self.network_ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

        self.network_ax.axis('off')
        self.network_fig.tight_layout()

        # Optimized rendering strategy
        if self._dragging:
            self.network_canvas.draw_idle()
        else:
            self.network_canvas.draw()

    # === Interaction Event Handling ===

    def find_node_at_event(self, event):
        """Identify the graph node at the cursor coordinates."""
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
        """Handle mouse-down event for node selection and drag initiation."""
        if event.inaxes != self.network_ax:
            return
        self.dragged_node = self.find_node_at_event(event)
        
        # Update Selection State
        if self.dragged_node:
            self.selected_graph_node = self.dragged_node
            self.network_canvas.get_tk_widget().config(cursor="hand2")
            self.status_bar.config(text=f"Node Selected: {self.dragged_node} (Delete to remove)")
            
            # Start drag operation
            self._dragging = True
            # Lock viewport to prevent jitter during interaction
            self._drag_lock_xlim = self.network_ax.get_xlim()
            self._drag_lock_ylim = self.network_ax.get_ylim()
            
            # Context synchronization
            self.cpt_node_combo.set(self.dragged_node)
            self.load_cpt_editor() 
        else:
            self.selected_graph_node = None
            
        self.draw_network()

    def on_release(self, event):
        """Handle mouse-up event for drag termination."""
        if self.dragged_node:
            self.dragged_node = None
            self._dragging = False
            self.network_canvas.get_tk_widget().config(cursor="")
            self.draw_network()

    def on_motion(self, event):
        """Handle cursor motion event for real-time node repositioning (throttled)."""
        if self.dragged_node and event.inaxes == self.network_ax:
            # Throttle refresh rate to ~30 FPS
            current_time = time.time()
            if current_time - self._last_drag_time < 0.03:
                return
            self._last_drag_time = current_time

            if event.xdata is not None and event.ydata is not None:
                self.node_positions[self.dragged_node] = (event.xdata, event.ydata)
                self.draw_network()


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """Main execution block."""
    try:
        print("="*70)
        print("BayesQ - Quantum Bayesian Network Platform")
        print("="*70)
        print()
        print("Initializing runtime environment...")

        root = tk.Tk()
        app = QBNApp(root)

        print("✓ System initialization complete.")
        print()
        print("Launching Graphical User Interface...")
        print()

        root.mainloop()

    except (ImportError, ModuleNotFoundError) as e:
        print(f"\n✗ Critical Error: Unresolved dependency detected.")
        print(f"  {e}")
        print()
        print("Please resolve dependencies via:")
        print("  pip install qiskit qiskit-aer")
        print("  pip install networkx matplotlib psutil")
        print()
    except Exception as e:
        print(f"\n✗ Fatal Exception: {e}")
        import traceback
        traceback.print_exc()
