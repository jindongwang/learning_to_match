# A wrapper of Visdom for visualization

import visdom
import numpy as np
import time
import torch

class Visualize(object):
    def __init__(self, port=8097, env='env'):
        self.port = port
        self.env = env
        self.vis = visdom.Visdom(port=self.port, env=self.env)

    def plot_line(self, Y, global_step, title='title', legend=['legend']):
        """ Plot line
        Inputs:
            Y (list): values to plot, a list
            global_step (int): global step
        """
        y = np.array(Y).reshape((1, len(Y)))
        self.vis.line(
            Y = y, 
            X = np.array([global_step]), 
            win = title,
            opts = dict(
                title = title,
                height = 360,
                width = 400,
                legend = legend,
            ),
            update = 'new' if global_step==0 else 'append'
        )

    def heat_map(self, X, title='title'):
        self.vis.heatmap(
            X = X,
            win = title,
            opts=dict(
                title = title,
                width = 360,
                height = 400,
            )
        )

    def log(self, info, title='log_text'):
        """
        self.log({'loss':1, 'lr':0.0001})
        """

        log_text = ('[{time}] {info} <br>'.format(
                            time=time.strftime('%m%d_%H%M%S'),\
                            info=info)) 
        self.vis.text(log_text, title, append=True)  

        '''
        File name: plot-pytorch-autograd-graph.py
        Author: Ludovic Trottier
        Date created: November 8, 2017.
        Date last modified: November 8, 2017
        Credits: moskomule (https://discuss.pytorch.org/t/print-autograd-graph/692/15)
    '''
    


    def make_dot(self, var, params):
        """ Produces Graphviz representation of PyTorch autograd graph.
        
        Blue nodes are trainable Variables (weights, bias).
        Orange node are saved tensors for the backward pass.
        
        Args:
            var: output Variable
            params: list of (name, Parameters)
        """
        
        param_map = {id(v): k for k, v in params}

        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        from graphviz import Digraph
        dot = Digraph(
            filename='network', 
            format='pdf',
            node_attr=node_attr, 
            graph_attr=dict(size="12,12"))
        seen = set()
        
        def add_nodes(var):
            if var not in seen:
                
                node_id = str(id(var))
                
                if torch.is_tensor(var):
                    node_label = "saved tensor\n{}".format(tuple(var.size()))
                    dot.node(node_id, node_label, fillcolor='orange')
                    
                elif hasattr(var, 'variable'):
                    variable_name = param_map.get(id(var.variable))
                    variable_size = tuple(var.variable.size())
                    node_name = "{}\n{}".format(variable_name, variable_size)
                    dot.node(node_id, node_name, fillcolor='lightblue')
                    
                else:
                    node_label = type(var).__name__.replace('Backward', '')
                    dot.node(node_id, node_label)
                    
                seen.add(var)
                
                if hasattr(var, 'next_functions'):
                    for u in var.next_functions:
                        if u[0] is not None:
                            dot.edge(str(id(u[0])), str(id(var)))
                            add_nodes(u[0])
                            
                if hasattr(var, 'saved_tensors'):
                    for t in var.saved_tensors:
                        dot.edge(str(id(t)), str(id(var)))
                        add_nodes(t)

        add_nodes(var.grad_fn)
        
        return dot


if __name__ == '__main__':
    vvv = visdom.Visdom(port=8097, env='test')
    def test():
        vis = Visualize(env='test')
        import time
        for i in range(10):
            y = np.random.rand(1, 2)
            title = 'Two values'
            legend = ['value 1', 'value 2']
            vis.plot_line([y[0,0], y[0,1]], i, title, legend)
            time.sleep(2)
    test()