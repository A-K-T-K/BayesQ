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
    git clone https://github.com/A-K-T-K/BayesQ.git
    cd BayesQ  # Match the repository name case
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
docker build -t bayesq

# Run the container 
docker run -it bayesq python benchmarks/benchmark.py
```
*Note: GUI forwarding requires X11 configuration; headless benchmarks run natively.*

---

## 🖥️ Usage

### Running the Application
To launch the graphical interface:
```bash
python bayesq.py
```

### Workflow
1.  **Design:** Use the **Network Builder** tab to add nodes and connect them.
2.  **Define:** Select a node and use the **CPT Editor** to input probabilities.
3.  **Build:** Press `Ctrl+B` (or Run > Build Circuit) to generate the quantum circuit.
4.  **Simulate:** Go to the **Inference** tab, set evidence (optional), and click **Run Inference**.

### Loading Case Studies
The `case_studies/` folder contains the pre-built example networks
* **Asia Network:** `case_studies/asia.qbn.json`
* **Cancer Network:** `case_studies/cancer.qbn.json`
* **Reliability analysis:** `case_studies/reliability.qbn.json`

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
BayesQ/
├── assets/                  # Static assets (images, figures, etc.)
├── benchmarks/              # Evaluation and benchmarking artifacts
│   ├── benchmark.py         # Data generation and benchmarking script
│   └── plot_results.py      # Benchmark results visualization
├── case_studies/            # Example Bayesian / Quantum Bayesian networks
│   ├── asia.qbn.json        # Asia network (QBN format)
│   ├── cancer.qbn.json      # Cancer network (QBN format)
│   ├── reliability.qbn.json # Reliability network (QBN format)
│   └── genie/               # Classical Bayesian networks (GeNIe format)
│       ├── asia.xdsl
│       ├── cancer.xdsl      # For classical inference in GeNIe software
│       └── reliability.xdsl
├── bayesq.py                   # Main application entry point
├── Dockerfile               # Docker configuration
├── requirements.txt         # Python dependencies
├── LICENSE                  # License information
└── README.md                # Project documentation

```

---

<!-- ## 📄 Citation

If you use BayesQ in your research, please cite our paper:

> **A Visual Platform for Quantum Bayesian Inference and Circuit Synthesis: Software Architecture, Engineering Experience, and Evaluation** > *Abhinav Krishnan T K, Indranil Hazra* > *Special Issue: Quantum Software Development Life Cycle, 2025* -->


## 📚 Acknowledgments

The example networks in `case_studies/` are based on reference Bayesian networks and research publications:

- **Asia (Chest Clinic) Network**: Originally described in Lauritzen, S. L., & Spiegelhalter, D. J. (1988). Local computations with probabilities on graphical structures and their application to expert systems. *Journal of the Royal Statistical Society: Series B (Methodological)*, 50(2), 157-224. https://doi.org/10.1111/j.2517-6161.1988.tb01721.x

- **Cancer Network**: Adapted from the bnlearn Bayesian Network Repository (https://www.bnlearn.com/bnrepository/)

- **Reliability Network**: Adapted from the fault tree structure in Xiong, S., Guo, Y., Yang, H., Zou, H., & Wei, K. (2021). Reliability study of motor controller in electric vehicle by the approach of fault tree analysis. *Engineering Failure Analysis*, 121, 105165. https://doi.org/10.1016/j.engfailanal.2020.105165

These networks are used for benchmarking and validation purposes.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
