import numpy
import networkx as nx
from pyvis.network import Network
import datetime

onehot = lambda y : y==numpy.unique(y)

scaler = lambda X : (X-X.mean())/X.std()

class GraphManager:
    ModelGraph = nx.Graph()
    def __init__(self):
        ...
    def add_to_graph(self,val):
        inids,out = val
        outid = id(out)
        if not hasattr(inids,'__iter__'):
            inids = [inids]
        for inid in inids :
            GraphManager.ModelGraph.add_edges_from([(inid,outid)])
        return outid,out
    def draw(engine=None,params={},outname="ModelGraph.html"):
        if not engine:
            nx.draw(GraphManager.ModelGraph,with_labels=True)
        elif engine == 'pyviz':
            pynet = Network(**params)
            pynet.from_nx(GraphManager.ModelGraph)
            pynet.barnes_hut()
            #for node in pynet.nodes:
            #    node["detail"] = ctypes.cast(node['id'], ctypes.py_object).value
            return pynet.show(outname)