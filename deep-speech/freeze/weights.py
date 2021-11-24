# purpose: check that freezing the first layers is working by printing out the layer weights

import tensorflow as tf
from tensorflow.python.framework import tensor_util
import argparse

def load_pb(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        graph_nodes=[n for n in graph_def.node]
        return graph_nodes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path the the .pb model')
    parser.add_argument('--layer', help='layer weights to display e.g. layer_1/weights')
    args = parser.parse_args()

    graph_nodes = load_pb(args.path)
    wts = [n for n in graph_nodes if n.op=='Const']
    
    #for n in wts:
    #    print(n.name)

    for n in wts:
        if n.name == args.layer:
            print("Name of the node - %s" % n.name)
            print("Value - ")
            print(tensor_util.MakeNdarray(n.attr['value'].tensor))

if __name__ == '__main__':
    main()
