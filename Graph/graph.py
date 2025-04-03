import pandas as pd
import math
import networkx as nx
from shapely.geometry import LineString
import json

# Load CSV files
lines_df = pd.read_csv('detected_lines.csv')
symbols_df = pd.read_csv('detected_symbols.csv')
text_df = pd.read_csv('detected_text.csv')

# Normalize coordinates for the lines, symbols, and text detections
lines = lines_df[['X1', 'Y1', 'X2', 'Y2']].values.tolist()
symbols = symbols_df[['Prediction', 'X1', 'Y1', 'X2', 'Y2']].values.tolist()
texts = text_df[['Detected Text', 'x_min', 'y_min', 'x_max', 'y_max']].values.tolist()

# Initialize the graph
G = nx.Graph()

# Add nodes for symbols
for symbol in symbols:
    symbol_id = f"s-{symbol[0]}"  # Creating a unique ID for each symbol
    G.add_node(symbol_id, type="symbol", label=symbol[0], bbox=(symbol[1], symbol[2], symbol[3], symbol[4]))

# Add nodes for lines
for i, line in enumerate(lines):
    line_id = f"l-{i}"  # Creating a unique ID for each line
    G.add_node(line_id, type="line", start=(line[0], line[1]), end=(line[2], line[3]))

# Add nodes for text
for text in texts:
    text_id = f"t-{text[0]}"
    G.add_node(text_id, type="text", bbox=(text[1], text[2], text[3], text[4]))

# Function to calculate the Euclidean distance
def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# Function to check proximity of line point to symbol or text
def check_proximity(line_point, box):
    return distance(line_point, (box[0], box[1])) < 50  # Adjust threshold as needed

# Add edges for line-to-symbol and line-to-text
def add_line_to_elements_edges():
    for i, line in enumerate(lines):
        line_id = f"l-{i}"
        line_start = (line[0], line[1])
        line_end = (line[2], line[3])

        # Check proximity to symbols
        for j, symbol in enumerate(symbols):
            symbol_box = (symbol[1], symbol[2], symbol[3], symbol[4])
            if check_proximity(line_start, symbol_box):
                G.add_edge(line_id, f"s-{symbol[0]}")  # Connecting line to symbol
            if check_proximity(line_end, symbol_box):
                G.add_edge(line_id, f"s-{symbol[0]}")  # Connecting line to symbol

        # Check proximity to text
        for j, text in enumerate(texts):
            text_box = (texts[j][1], texts[j][2], texts[j][3], texts[j][4])
            if check_proximity(line_start, text_box):
                G.add_edge(line_id, f"t-{text[0]}")  # Connecting line to text
            if check_proximity(line_end, text_box):
                G.add_edge(line_id, f"t-{text[0]}")  # Connecting line to text

# Add edges for lines based on intersection
def add_line_to_line_edges():
    for i, line1 in enumerate(lines):
        line_id1 = f"l-{i}"
        for j, line2 in enumerate(lines):
            if i >= j:  # Skip redundant comparisons (i == j and duplicate checks)
                continue
            line_id2 = f"l-{j}"
            line1_geom = LineString([(line1[0], line1[1]), (line1[2], line1[3])])
            line2_geom = LineString([(line2[0], line2[1]), (line2[2], line2[3])])
            if line1_geom.intersects(line2_geom):
                G.add_edge(line_id1, line_id2)  # Adding edge between intersecting lines

# Add edges between symbols based on proximity
def add_symbol_to_symbol_edges():
    for i, symbol1 in enumerate(symbols):
        symbol_id1 = f"s-{symbol1[0]}"
        for j, symbol2 in enumerate(symbols):
            if i >= j:  # Skip redundant comparisons (i == j)
                continue
            symbol_id2 = f"s-{symbol2[0]}"
            # Check if symbols are close enough
            if check_proximity((symbol1[1], symbol1[2]), (symbol2[1], symbol2[2])):  # Adjust this check as needed
                G.add_edge(symbol_id1, symbol_id2)  # Adding edge between symbols

# Convert the graph to the required JSON format
def graph_to_json():
    data = {
        "nodes": [],
        "edges": []
    }

    # Add nodes for symbols, lines, and text
    for node_id, node_data in G.nodes(data=True):
        node = {
            "id": node_id,
            "type": node_data.get('type', 'unknown')  # Default to 'unknown' if 'type' is missing
        }

        if node_data['type'] == "symbol":
            node["label"] = node_data["label"]
            node["bbox"] = node_data["bbox"]
        elif node_data['type'] == "line":
            node["start"] = node_data["start"]
            node["end"] = node_data["end"]
        elif node_data['type'] == "text":
            node["bbox"] = node_data["bbox"]

        data["nodes"].append(node)

    # Add edges (relationships between nodes)
    for u, v, edge_data in G.edges(data=True):
        edge = {
            "source": u,
            "target": v
        }
        if 'relationship' in edge_data:
            edge["relationship"] = edge_data["relationship"]
        else:
            if "s-" in u and "l-" in v:
                edge["relationship"] = "line_to_symbol"
            elif "l-" in u and "t-" in v:
                edge["relationship"] = "line_to_text"
            elif "l-" in u and "l-" in v:
                edge["relationship"] = "line_to_line"
            elif "s-" in u and "s-" in v:
                edge["relationship"] = "symbol_to_symbol"
        data["edges"].append(edge)

    return data

# Add edges and generate the final JSON output
add_line_to_elements_edges()  # Add line-to-symbol and line-to-text edges
add_symbol_to_symbol_edges()  # Add symbol-to-symbol edges
add_line_to_line_edges()  # Add line-to-line edges

# Convert the graph to JSON format and save it
graph_json = graph_to_json()

# Save the JSON data to a file
json_file_path = "graph_output.json"
with open(json_file_path, 'w') as json_file:
    json.dump(graph_json, json_file, indent=4)

json_file_path  # Return the updated file path for download


# import pandas as pd
# import math
# import networkx as nx
# import matplotlib.pyplot as plt
# from shapely.geometry import LineString
# import json

# # Load CSV files
# lines_df = pd.read_csv('detected_lines.csv')
# symbols_df = pd.read_csv('detected_symbols.csv')
# text_df = pd.read_csv('detected_text.csv')

# # Normalize coordinates for the lines, symbols, and text detections
# lines = lines_df[['X1', 'Y1', 'X2', 'Y2']].values.tolist()
# symbols = symbols_df[['Prediction', 'X1', 'Y1', 'X2', 'Y2']].values.tolist()
# texts = text_df[['Detected Text', 'x_min', 'y_min', 'x_max', 'y_max']].values.tolist()

# # Initialize the graph
# G = nx.Graph()

# # Add nodes for symbols
# for symbol in symbols:
#     symbol_id = f"s-{symbol[0]}"
#     G.add_node(symbol_id, type="symbol", label=symbol[0], pos=((symbol[1] + symbol[3]) / 2, (symbol[2] + symbol[4]) / 2))

# # Add nodes for lines
# for i, line in enumerate(lines):
#     line_id = f"l-{i}"
#     G.add_node(line_id, type="line", pos=((line[0] + line[2]) / 2, (line[1] + line[3]) / 2))

# # Add nodes for text
# for text in texts:
#     text_id = f"t-{text[0]}"
#     G.add_node(text_id, type="text", pos=((text[1] + text[3]) / 2, (text[2] + text[4]) / 2))

# # Function to calculate the Euclidean distance
# def distance(p1, p2):
#     return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# # Function to check proximity of line point to symbol or text
# def check_proximity(line_point, box):
#     return distance(line_point, (box[0], box[1])) < 50  # Adjust threshold as needed

# # Add edges for line-to-symbol and line-to-text
# def add_line_to_elements_edges():
#     for i, line in enumerate(lines):
#         line_id = f"l-{i}"
#         line_start = (line[0], line[1])
#         line_end = (line[2], line[3])

#         # Check proximity to symbols
#         for symbol in symbols:
#             symbol_id = f"s-{symbol[0]}"
#             symbol_box = (symbol[1], symbol[2], symbol[3], symbol[4])
#             if check_proximity(line_start, symbol_box) or check_proximity(line_end, symbol_box):
#                 G.add_edge(line_id, symbol_id)

#         # Check proximity to text
#         for text in texts:
#             text_id = f"t-{text[0]}"
#             text_box = (text[1], text[2], text[3], text[4])
#             if check_proximity(line_start, text_box) or check_proximity(line_end, text_box):
#                 G.add_edge(line_id, text_id)

# # Add edges for lines based on intersection
# def add_line_to_line_edges():
#     for i, line1 in enumerate(lines):
#         line_id1 = f"l-{i}"
#         for j, line2 in enumerate(lines):
#             if i >= j:
#                 continue
#             line_id2 = f"l-{j}"
#             line1_geom = LineString([(line1[0], line1[1]), (line1[2], line1[3])])
#             line2_geom = LineString([(line2[0], line2[1]), (line2[2], line2[3])])
#             if line1_geom.intersects(line2_geom):
#                 G.add_edge(line_id1, line_id2)

# # Add edges between symbols based on proximity
# def add_symbol_to_symbol_edges():
#     for i, symbol1 in enumerate(symbols):
#         symbol_id1 = f"s-{symbol1[0]}"
#         for j, symbol2 in enumerate(symbols):
#             if i >= j:
#                 continue
#             symbol_id2 = f"s-{symbol2[0]}"
#             if check_proximity((symbol1[1], symbol1[2]), (symbol2[1], symbol2[2])):
#                 G.add_edge(symbol_id1, symbol_id2)

# # Generate edges
# add_line_to_elements_edges()
# add_symbol_to_symbol_edges()
# add_line_to_line_edges()

# # Draw the graph
# plt.figure(figsize=(12, 8))
# pos = nx.get_node_attributes(G, 'pos')
# nx.draw(G, pos, with_labels=True, node_size=500, font_size=8, edge_color='gray')

# # Color nodes based on type
# node_colors = []
# for node in G.nodes():
#     node_type = G.nodes[node]['type']
#     if node_type == 'symbol':
#         node_colors.append('red')
#     elif node_type == 'line':
#         node_colors.append('blue')
#     elif node_type == 'text':
#         node_colors.append('green')
#     else:
#         node_colors.append('black')

# nx.draw_networkx_nodes(G, pos, node_color=node_colors)
# plt.title("Graph Representation of Detected Elements")
# plt.show()



