# from LayerClass import Layer
from .LayerClass import Layer

class PropagationLayer(Layer):
    """Handle all the traced layer non define as call_module or call_function/method.

    :param node: Is the actual traced node from torch.fx.
    :type node: torch.fx.node.Node
    :param trace: Is the complete trace of torch.fx of the model.
    :type trace: torch.fx.graph._node_list
    :param prev_node: Is the just previous node in the trace before the actual traced node.
    :type node: torch.fx.node.Node
    """

    def __init__(self, node, trace, prev_node):
        """Constructor."""
        super().__init__(node, trace, prev_node)

    def get_declaration(self) -> str:
        """Generate and return the full class generated for a propagation layer.
        
        What we call a propagation layer, is a layer who do no direct action on the data but propagate it further.

        Sometimes this layer have a special argument to propagate but have to return the input for the next layers.

        :return: The full declaration of a Layer containing a CallFunction
        :rtype: str
        """
        string = self.generate_class()

        task = ""
        if len(self.node.args) > 0 and self.node.op == "placeholder":
            task = str(self.node.args[0])
        else:
            task = "input"

        string += self.generate_forward(task)

        return string

    def __str__(self) -> str:
        """Allow to print easily all the information of a layer.
        
        It adds a print to inform that it is a propagation layer.
        :return: String to print
        :rtype: str
        """
        print_str = super().__str__()
        print_str += "    This layer is just a propagation one\n\n"
        return print_str
