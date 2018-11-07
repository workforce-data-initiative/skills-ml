from graphviz import Graph

def display_nodes(nodes):
    graph = Graph()
    for node in nodes:
        graph.node(node.identifier, node.name)
        for child in node.children:
            graph.edge(node.identifier, child.identifier)
    return graph

def display_ontology(ontology):
    graph = Graph()
    graph.attr(rankdir='TB')
    with graph.subgraph(name='cluster_competencies') as c:
        c.attr(rankdir='LR')
        c.attr(label='Competencies')
        for competency in ontology.competencies:
            c.node(competency.identifier, competency.name)
            for child in competency.children:
                c.edge(competency.identifier, child.identifier)
    with graph.subgraph(name='cluster_occupations') as c:
        c.attr(rankdir='RL')
        c.attr(label='Occupations')
        for occupation in ontology.occupations:
            c.node(occupation.identifier, occupation.name)
            for child in occupation.children:
                c.edge(occupation.identifier, child.identifier)

    for edge in ontology.edges:
        graph.edge(edge.competency.identifier, edge.occupation.identifier, style='dashed')

    return graph
