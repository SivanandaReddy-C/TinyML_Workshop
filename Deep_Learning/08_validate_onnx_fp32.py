"""
===========================================================================
Project : TinyML Workshop - Deep Learning
File    : 08_validate_onnx_fp32.py
Author  : Dr. C. Sivananda Reddy

Description:
    Validates the exported FP32 ONNX model by

    1. Loading the ONNX model
    2. Verifying model integrity
    3. Creating ONNX Runtime session
    4. Displaying model information
    5. Generating validation report

This script will be extended in the next slides to include

    • Complete model evaluation
    • Performance benchmarking

===========================================================================
"""

import os
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort


# ==========================================================================
# Configuration
# ==========================================================================

ONNX_MODEL_PATH = "../ONNX/mlp_activity_classifier.onnx"
OUTPUT_DIR = "outputs"
REPORT_FILE = os.path.join(OUTPUT_DIR, "onnx_fp32_validation.txt")


# ==========================================================================
# Utility Functions
# ==========================================================================

def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subheader(title):
    """Print formatted subsection header."""
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def create_output_directory():
    """Create output directory if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================================
# ONNX Model Verification
# ==========================================================================

def load_and_verify_model(model_path):
    """
    Load the ONNX model and verify its integrity.

    Returns
    -------
    model : onnx.ModelProto
    """

    print_subheader("Loading ONNX Model")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ONNX model not found:\n{model_path}"
        )

    model = onnx.load(model_path)

    print("✓ Model loaded successfully.")

    print_subheader("Verifying Model Integrity")

    onnx.checker.check_model(model)

    print("✓ Model verification passed.")

    return model


# ==========================================================================
# Create ONNX Runtime Session
# ==========================================================================

def create_runtime_session(model_path):
    """
    Create ONNX Runtime inference session.
    """

    print_subheader("Creating ONNX Runtime Session")

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    print("✓ Session created successfully.")

    return session


# ==========================================================================
# Display Runtime Information
# ==========================================================================

def display_runtime_information(session):
    """Display execution provider information."""

    print_subheader("Execution Provider")

    providers = session.get_providers()

    for provider in providers:
        print(f"• {provider}")


# ==========================================================================
# Display Input Information
# ==========================================================================

def display_input_information(session):
    """Display ONNX model input information."""

    print_subheader("Model Input")

    input_tensor = session.get_inputs()[0]

    print(f"Input Name   : {input_tensor.name}")
    print(f"Input Shape  : {input_tensor.shape}")
    print(f"Input Type   : {input_tensor.type}")


# ==========================================================================
# Display Output Information
# ==========================================================================

def display_output_information(session):
    """Display ONNX model output information."""

    print_subheader("Model Output")

    output_tensor = session.get_outputs()[0]

    print(f"Output Name  : {output_tensor.name}")
    print(f"Output Shape : {output_tensor.shape}")
    print(f"Output Type  : {output_tensor.type}")


# ==========================================================================
# Display Graph Information
# ==========================================================================

def display_graph_information(model):
    """Display general graph information."""

    print_subheader("Graph Information")

    graph = model.graph

    print(f"Graph Name          : {graph.name}")
    print(f"IR Version          : {model.ir_version}")
    print(f"Producer            : {model.producer_name}")
    print(f"Producer Version    : {model.producer_version}")
    print(f"Opset Version       : {model.opset_import[0].version}")

    print(f"Number of Nodes     : {len(graph.node)}")
    print(f"Number of Inputs    : {len(graph.input)}")
    print(f"Number of Outputs   : {len(graph.output)}")
    print(f"Number of Parameters: {len(graph.initializer)}")


# ==========================================================================
# Display Operator Summary
# ==========================================================================

def display_operator_summary(model):
    """Display operator usage summary."""

    print_subheader("Operator Summary")

    operators = [node.op_type for node in model.graph.node]

    counts = Counter(operators)

    for op, count in sorted(counts.items()):
        print(f"{op:<15} : {count}")


# ==========================================================================
# Generate Validation Report
# ==========================================================================

def generate_report(model, session):
    """Generate ONNX validation report."""

    graph = model.graph

    input_tensor = session.get_inputs()[0]
    output_tensor = session.get_outputs()[0]

    operators = Counter(
        node.op_type for node in graph.node
    )

    with open(REPORT_FILE, "w") as f:

        f.write("FP32 ONNX MODEL VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("Validation Status : PASSED\n\n")

        f.write("Execution Provider\n")
        f.write("------------------\n")

        for provider in session.get_providers():
            f.write(provider + "\n")

        f.write("\n")

        f.write("Input Information\n")
        f.write("-----------------\n")

        f.write(f"Name  : {input_tensor.name}\n")
        f.write(f"Shape : {input_tensor.shape}\n")
        f.write(f"Type  : {input_tensor.type}\n\n")

        f.write("Output Information\n")
        f.write("------------------\n")

        f.write(f"Name  : {output_tensor.name}\n")
        f.write(f"Shape : {output_tensor.shape}\n")
        f.write(f"Type  : {output_tensor.type}\n\n")

        f.write("Graph Information\n")
        f.write("-----------------\n")

        f.write(f"Graph Name        : {graph.name}\n")
        f.write(f"IR Version        : {model.ir_version}\n")
        f.write(f"Producer          : {model.producer_name}\n")
        f.write(f"Producer Version  : {model.producer_version}\n")
        f.write(f"Opset Version     : {model.opset_import[0].version}\n")
        f.write(f"Nodes             : {len(graph.node)}\n")
        f.write(f"Inputs            : {len(graph.input)}\n")
        f.write(f"Outputs           : {len(graph.output)}\n")
        f.write(f"Parameters        : {len(graph.initializer)}\n\n")

        f.write("Operator Summary\n")
        f.write("----------------\n")

        for op, count in sorted(operators.items()):
            f.write(f"{op:<15} : {count}\n")

    print("\n✓ Validation report saved.")
    print(REPORT_FILE)


# ==========================================================================
# Main
# ==========================================================================

def main():

    print_header("FP32 ONNX MODEL VERIFICATION")

    create_output_directory()

    model = load_and_verify_model(ONNX_MODEL_PATH)

    session = create_runtime_session(ONNX_MODEL_PATH)

    display_runtime_information(session)

    display_input_information(session)

    display_output_information(session)

    display_graph_information(model)

    display_operator_summary(model)

    generate_report(model, session)

    print("\n✓ ONNX model verification completed successfully.")


if __name__ == "__main__":
    main()