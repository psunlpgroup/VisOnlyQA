import src.training_dataset.alphageometry.src.graph as gh


def save_alphageometry_figure(graph: gh.Graph, output_path: str, height: float=512/100, width=512/100):
    gh.nm.draw(
        graph.type2nodes[gh.Point],
        graph.type2nodes[gh.Line],
        graph.type2nodes[gh.Circle],
        graph.type2nodes[gh.Segment],
        output_figure_path=output_path,
        imheight=height,
        imwidth=width,
    )
