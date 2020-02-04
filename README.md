# df-autograd
This is a simple autodifferentiation experiment.  
`Node` computes first order derivatives with respect to all variables in the graph.  
I also implemented numeric differentiation to verify my results.  

Instead of a `backward()` method each Node has a `gradient()` method which returns the gradient's computation graph.  
This allows for easy computation of higher order derivatives.  
See readme.py for further details. 
