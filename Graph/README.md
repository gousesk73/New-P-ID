# Graph Construction

## Overview
The graph is built using the NetworkX library and represents relationships between lines, symbols, and text. The final graph structure is exported as a JSON file.

## Prerequisites
Ensure you have the required Python packages installed:
```bash
pip install pandas networkx shapely
```

## Input Files
- `detected_lines.csv`: Contains line segment coordinates (`X1, Y1, X2, Y2`).
- `detected_symbols.csv`: Contains detected symbols with their bounding box coordinates (`Prediction, X1, Y1, X2, Y2`).
- `detected_text.csv`: Contains detected text with bounding box coordinates (`Detected Text, x_min, y_min, x_max, y_max`).

## Processing Steps
1. Load CSV files into Pandas DataFrames.
2. Create nodes for symbols, lines, and text using a unique ID.
3. Add edges:
   - **Line-to-symbol** and **line-to-text** edges based on proximity.
   - **Line-to-line** edges if two lines intersect.
   - **Symbol-to-symbol** edges if symbols are near each other.
4. Convert the graph to a JSON structure with nodes and edges.
5. Save the generated graph data to `graph_output.json`.

## Output
- **JSON File:** `graph_output_with_edges_and_text.json`
  - **Nodes:** Representing lines, symbols, and text elements.
  - **Edges:** Defining relationships such as `line_to_symbol`, `line_to_text`, `line_to_line`, and `symbol_to_symbol`.

## Usage
Run the script in a Python environment:
```bash
python graph.py
```
Ensure that the input CSV files exist in the working directory before executing the script.



