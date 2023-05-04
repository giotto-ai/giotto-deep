from .LayerClass import Layer
import torch


class GetAttrModule(Layer):
    """Handle all the traced layer define as "getattr".
    
    If a special self is used in the module traced, torch fx mark it as an getattr layer.
    So we have to recreate the self object used and propagate it.

    For the moment only nn.parameter.Parameter are handled.

    :param node: Is the actual traced node from torch.fx.
    :type node: torch.fx.node.Node
    :param trace: Is the complete trace of torch.fx of the model.
    :type trace: torch.fx.graph._node_list
    :param prev_node: Is the just previous node in the trace before the actual traced node.
    :type node: torch.fx.node.Node
    :param net: The network given to the pipeline tool.
    :type net: Non-specific, could be a module or a set of module.
    """

    def __init__(self, node, trace, prev_node, net):
        """Constructor."""
        super().__init__(node, trace, prev_node)
        # We have to split the node.target because he contains the name of the attribute used.
        # For example "model.pooling.layer.query" the name of the attribute is the last word, query.
        module_tmp = str(node.target).split('.')
        tmp = ""
        for name in module_tmp[0:-1]:
            tmp += f"{name}."
        tmp = tmp[:-1]
        attr_parsed = module_tmp[len(module_tmp) - 1:]

        # Once we have the attribute to found we will search in the network given by the user
        for name, module in net.named_modules():
            if str(name) == tmp:
                attr = getattr(module, attr_parsed[0])

                if isinstance(attr, torch.nn.parameter.Parameter):
                    self.module_attr_desc = "parameter.Parameter(torch.Tensor("

                    for shape in attr.shape:
                        self.module_attr_desc += f"{shape}, "

                    self.module_attr_desc = self.module_attr_desc[:-2]

                    requires_grad = getattr(attr, "requires_grad")

                    self.module_attr_desc += f"), requires_grad={requires_grad})"

    def get_declaration(self) -> str:
        """Generate and return the full class generate for a layer containing a gettatr of a module.
        
        For example :
        .. code-block:: python
            @skippable...
            class {self.name}_layer(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.fc = nn.parameter.Parameter(torch.Tensor(1, 16), requires_grad=True)
                def forward(self, input):
                    ret = self.fc
                    yield stash ...
                    return ret
        :return: The full class generated for a getattr layer
        :rtype: str
        """
        string = self.generate_class()
        string += self.generate_init(str(self.module_attr_desc))
        string += self.generate_forward("self.fc")

        return string

    def __str__(self) -> str:
        """Allow to print easily all the information of a layer.
        
        It adds a print of the parameter created in the layer.
        :return: String to print
        :rtype: str
        """
        print_str = super().__str__()
        print_str += f"    The parameter description is {self.module_attr_desc}\n\n"
        return print_str
