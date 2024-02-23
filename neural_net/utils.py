import numpy,os
import networkx as nx
from pyvis.network import Network
import datetime

non_infinite = lambda a : a[numpy.isfinite(a)]

get_path = lambda dir : os.path.join(os.path.dirname(os.path.abspath(__file__)),*dir)

now = lambda : datetime.datetime.now().strftime('%Y%m%d%H%M%S')



onehot = lambda y : (y==numpy.unique(y))+0

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
def get_batchix(n,batch_size):  
        batch_size = batch_size or n
        batchix = list(range(0,n,batch_size))
        if batchix[-1]<n : batchix.append(n)
        batchix = [slice(low,high) for low,high in zip(batchix,batchix[1:])]
        return batchix