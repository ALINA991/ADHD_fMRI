import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objs as go
from plotly.offline import plot
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer

def compute_graph(df, measure, thr_edge, allow_neg_edge=False):
    # Compute the similarity or correlation matrix based on the chosen measure
    if measure == 'correlation':
        similarity_matrix = df.corr(min_periods=1)
    elif measure == 'cosine':
        # Compute the cosine similarity matrix
        imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        similarity_matrix = pd.DataFrame(cosine_similarity(df_imputed.T), index=df.columns, columns=df.columns)
    else:
        raise ValueError("Invalid measure. Choose 'correlation' or 'cosine'.")
    
    G = nx.Graph()

    # Add nodes for each question
    for question in similarity_matrix.columns:
        G.add_node(question)

    # Add edges based on the threshold and whether negative edges are allowed
    for i in range(len(similarity_matrix.columns)):
        for j in range(i + 1, len(similarity_matrix.columns)):
            sim_value = similarity_matrix.iloc[i, j]
            if allow_neg_edge:
                if sim_value > thr_edge:
                    G.add_edge(similarity_matrix.columns[i], similarity_matrix.columns[j], weight=sim_value, color='blue')
                elif sim_value < -thr_edge:
                    G.add_edge(similarity_matrix.columns[i], similarity_matrix.columns[j], weight=sim_value, color='red')
            else:
                if abs(sim_value) > thr_edge:
                    G.add_edge(similarity_matrix.columns[i], similarity_matrix.columns[j], weight=sim_value, color='blue')

    return G


def plot_graph(G, question_dict, neg_edges = False):
    # Position the nodes using the spring layout
    pos = nx.spring_layout(G, seed = 42)

    # Prepare the edge traces for Plotly
    edge_trace = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        if neg_edges:
            edge_color = edge[2]['color']
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1.0, color=edge_color),
                hoverinfo='none'
        ))
        else: 
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=0.5, color='#888'),
                hoverinfo='none'
            ))
            

    # Prepare the node trace
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=[],
            size=10,
            line_width=2),
        text=[f"{node}: {question_dict[node]}" for node in G.nodes()]  # Hover text
    )

    # Create the layout
    layout = go.Layout(
        title='Interactive Network Graph with Positive and Negative Edges',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        plot_bgcolor='white'
    )

    # Create the figure
    fig = go.Figure(data=edge_trace + [node_trace], layout=layout)

    # Plot the figure
    plot(fig)
    