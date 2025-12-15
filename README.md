# BayesQ: A Visual Platform for Quantum Bayesian Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-%E2%96%B5-purple)](https://qiskit.org/)

**BayesQ** is a no-code, graphical software platform that automates the design, synthesis, and simulation of Quantum Bayesian Networks (QBNs), allowing domain experts to utilize quantum inference without writing QASM or Python code.

![QBN Designer Main Interface](assets/main_interface_screenshot.png)


---

## 🚀 Features

* **Visual Network Builder:** Drag-and-drop interface to design DAG topologies.
* **Automated Circuit Synthesis:** Translates classical probability tables (CPTs) into optimized quantum circuits using controlled rotation gates ($R_y$, $CR_y$, $MCR_y$).
* **Integrated Validation:** Real-time checking for DAG cycles, loose nodes, and probability consistency.
* **Noise Simulation:** Built-in support for Qiskit Aer noise models (Depolarizing, Thermal Relaxation) to test robustness.
* **Benchmarking Suite:** Includes headless scripts to verify algorithmic scalability.
* **Export Capabilities:** Export to OpenQASM 2.0, JSON, or CSV results.

---

## 📦 Installation

### Prerequisites
* Python 3.8 or higher
* (Optional) Docker for containerized execution

### Local Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/qbn-designer.git](https://github.com/yourusername/qbn-designer.git)
    cd qbn-designer
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 🐳 Docker Usage
To run the environment in a reproducible container 

```bash
# Build the image
docker build -t qbn-designer .

# Run the container 
docker run -it qbn-designer python benchmarks/benchmark_v2.py

```
*Note: GUI forwarding requires X11 configuration; headless benchmarks run natively.*

---

## 🖥️ Usage

### Running the Application
To launch the graphical interface:
```bash
python qbn.py
```

### Workflow
1.  **Design:** Use the **Network Builder** tab to add nodes and connect them.
2.  **Define:** Select a node and use the **CPT Editor** to input probabilities.
3.  **Build:** Press `Ctrl+B` (or Run > Build Circuit) to generate the quantum circuit.
4.  **Simulate:** Go to the **Inference** tab, set evidence (optional), and click **Run Inference**.

### Loading Case Studies
The `case_studies/` folder contains the pre-built example networks
* **Medical Diagnosis:** `case_studies/medical.qbn.json`
* **Financial Risk:** `case_studies/financial.qbn.json`
* **Environmental Monitoring:** `case_studies/environmental.qbn.json`

Use **File > Open Network** (`Ctrl+O`) to load these pre-built models.

---

## 📊 Reproducing Performance Benchmarks

Reproduce the evaluation results 

1.  **Run the Headless Benchmark:**
    Generates random DAGs of varying sizes and measures synthesis performance.
    ```bash
    python benchmarks/benchmark.py
    ```
    *Output:* `benchmark_results.csv`

2.  **Generate Plots:**
    Generates PDF plots: nodes vs. latency and nodes vs. memory usage
    ```bash
    python benchmarks/plot_results.py
    ```
    *Output:* `gentime.pdf`, `memplot.pdf`

---

## 📂 Project Structure

```text
qbn-designer/
├── assets/                 
├── benchmarks/             # Evaluation artifacts
│   ├── benchmark.py        # Data generation script
│   └── plot_results.py     # Plotting script
├── case_studies/           # Example networks
│   ├── medical.qbn.json
│   ├── financial.qbn.json
│   └── environmental.qbn.json
├── qbn.py                  # Main application entry point
├── Dockerfile              
├── requirements.txt        
├── LICENSE                 
└── README.md               

```

---

## 📄 Citation

If you use BayesQ in your research, please cite our paper:

> **A Visual Platform for Quantum Bayesian Inference and Circuit Synthesis: Software Architecture, Engineering Experience, and Evaluation** > *Abhinav Krishnan T K, Indranil Hazra* > *Special Issue: Quantum Software Development Life Cycle, 2025*

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.