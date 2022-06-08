import networkx as nx

def maxweight(ham_dis):
    # get the pairs with the highest hamming distances
    ham_graph = nx.Graph(ham_dis)
    matching = nx.max_weight_matching(ham_graph, maxcardinality=True, weight='weight')
    return list(matching) 