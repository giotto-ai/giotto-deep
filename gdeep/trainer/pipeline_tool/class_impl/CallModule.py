# from LayerClass import Layer
from .LayerClass import Layer


class CallModule(Layer):
    """Handle all the traced layer define as "call_module".
    
    Modules are all the torch Module present in the file "whitelist.txt", we do not guaranty the handle of custom module.

    :param node: Is the actual traced node from torch.fx.
    :type node: torch.fx.node.Node
    :param trace: Is the complete trace of torch.fx of the model.
    :type trace: torch.fx.graph._node_list
    :param prev_node: Is the just previous node in the trace before the actual traced node.
    :type node: torch.fx.node.Node
    """

    def __init__(self, node, trace, prev_node, module_desc):
        """Constructor."""
        super().__init__(node, trace, prev_node)
        self.module_desc = module_desc

    def get_declaration(self) -> str:
        """Generate and return the full class generate for a layer containing a call method or call function.
        
        For example :
        .. code-block:: python
            @skippable...
            class {self.name}_layer(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.fc = nn.call_module
                def forward(self, input):
                    ... = yield pop...
                    ret = call module (self.fc(input))
                    yield stash ...
                    return ret

        :return: The full declaration of a Layer containing a CallModule
        :rtype: str
        """
        string = self.generate_class()
        string += self.generate_init(str(self.module_desc))

        task = f"self.fc("
        for arg in self.args:
            # This is for handle the kwargs (i.e. dim=1)
            if isinstance(arg, list):
                task += f"{arg[0]}={arg[1]}, "
            else:
                task += f"{arg}, "
        task = task[:-2]
        task += ")"

        string += self.generate_forward(task)

        return string

    def __str__(self) -> str:
        """Allow to print easily all the information of a layer.
        
        It adds a print of the torch module executed.
        :return: String to print
        :rtype: str
        """
        print_str = super().__str__()
        print_str += f"    The module description is {self.module_desc}\n\n"
        return print_str
