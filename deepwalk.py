import random
import networkx as nx 

from gensim.models import Word2Vec


class DeepWalk:
  """

  Implement DeepWalk algorithm.

  reference paper : DeepWalk: Online Learning of Social Representations

  link : https://arxiv.org/abs/1403.6652

  Using the algorithm can get graph embedding model with your network data.


  """
  def __init__(self, G=None, adjlist_path=None, edgelist_path=None):
    """

    Parameters

    G : networkx : networkx graph.
    
    adjlist_path : network file path. 

    """
    if G == adjlist_path == edgelist_path == None:
      raise ValueError('all parameter is None, please check your input.')
      

    try:
      
      if G != None:
        self.G = G
      elif adjlist_path != None:
        self.G = nx.read_adjlist(adjlist_path)
      elif edgelist_path != None:
        self.G = nx.read_edgelist(edgelist_path)

    except Exception as e:
      print(e)



  def random_walk(self, iterations, start_node=None, random_walk_times=5):
    """

    : Implement of random walk algorithm :

    Parameters
    ----------------------------------------

    iterations : int : random walk number of iteration 

    start_node : str : choose start node (random choose a node, if start_node is None)

    random_walk_times : int : random walk times.

    ----------------------------------------

    Returns

    walk_records : list of walks record


    """
    walk_records = []
    

    for i in range(iterations):
      
      if start_node is None:
        s_node = random.choice(list(self.G.nodes()))
        walk_path = [s_node]
      else:
        walk_path = [start_node]
        
      current_node = s_node
      while(len(walk_path) < random_walk_times):
        neighbors = list(self.G.neighbors(current_node))
        
        
        current_node = random.choice(neighbors)
        walk_path.append(current_node)
          
      walk_records.append(walk_path)
    
    return walk_records


  def buildWord2Vec(self, **kwargs):
    """
    
    Using gensim to build word2vec model

    Parameters
    ----------------------------------------

    **kwargs

    
    walk_path : list : random walk results
    size : int : specific embedding dimension, default : 100 dim
    window : int : specific learn context window size, default : 5
    workers : int : specific workers. default : 2

    ----------------------------------------

    Returns

    walk_records : list of walks record


    """
    
    walk_path = kwargs.get('walk_path', None)
    if walk_path is None:
      return 
    
    size = kwargs.get('size', 100)
    window = kwargs.get('window', 5)
    workers = kwargs.get('workers', 2)

    embedding_model = Word2Vec(walk_path, size=size, window=window, min_count=0, workers=workers, sg=1, hs=1)

    return embedding_model






