import torch
from ..constant import TAB

class Layer:
    """Set up of the Layer to manage every thing for creating a layer with the init and forward functions.

    This class is the base class for CallFunction, CallModule, GetAttrModule and PropagationLayer. Thus,
    this should never have called.

    :param node: Is the actual traced node from torch.fx.
    :type node: torch.fx.node.Node
    :param trace: Is the complete trace of torch.fx of the model.
    :type trace: torch.fx.graph._node_list
    :param prev_node: Is the just previous node in the trace before the actual traced node.
    :type node: torch.fx.node.Node
    """

    def __init__(self, node, trace, prev_node):
        """Constructor."""
        self.node = node
        self.name = str(node)
        self.args = []
        self.kwargs = []
        self.pop_list = []
        self.stash_list = []

        self.getitem_idx = None
        # True if there is a change of CPU at this specific layer
        self.separation_layer = False

        # Call function to handle correctly all information about the node.
        self._args_handler(trace, prev_node)
        self._kwargs_handler()
        self._pop_handler(prev_node)

    def get_node(self) -> torch.fx.node.Node:
        """Getter for the node treated.

        :return: Node of the layer
        :rtype: torch.fx.node.Node
        """
        return self.node

    def get_name(self) -> str:
        """Getter for the name of the node, just is string format.

        :return: Name of the node.
        :rtype: str
        """
        return self.name

    def set_stash(self, target_node):
        """Update the stash_list, i.e. the list of layers to which it must send the result of its forward.

        :param target_node: Node for which the result should be saved.
        :type target_node: torch.fx.node.Node
        """
        tmp = self.name + "_to_" + str(target_node)
        if tmp not in self.stash_list:
            self.stash_list.append(tmp)

    def _get_handler(self, arg, trace) -> torch.fx.node.Node:
        """Exist in the only objective to handler successive getattr and getitem call.
        
        Because those traced node should not be considered as Layer, and we have to search for the last node before them.
        :param arg: The arg to check
        :type arg: torch.fx.node.Node
        :param trace: Is the complete trace of torch.fx of the model.
        :type trace: torch.fx.graph._node_list
        :return: The last parent of the chained getitem and getattr call (if there is no chain just their direct parent)
        :rtype: torch.fx.node.Node
        """
        _arg = arg

        while str(_arg).find("getitem") >= 0 or str(_arg).find("getattr") >= 0:
            for node in trace:
                if node == _arg:
                    _arg = node.args[0]
                    break
        return _arg

    def update_arg_by_attr(self, old_arg, new_arg):
        """Allow to update argument with a specific attribute.
        
        For example t.shape
        :param old_arg: Old argument to replace.
        :type old_arg: torch.fx.node.Node
        :param new_arg: New arg containing the attribute. (For ex .shape)
        :type new_arg: str
        """
        for i, arg in enumerate(self.args):
            if arg == old_arg:
                self.args[i] = new_arg
                break

    def get_pop_parent(self) -> list:
        """Getter for the pop_list, the list of the layer who the actual layers need stashed value.

        :return: The pop list
        :rtype: list
        """
        return self.pop_list

    def _pop_handler(self, prev_node):
        """Handle the creation of the pop list.
        
        This function will go throw the args of the treated node and found if in his param he have a layer who is not directly connected to him.
        :param prev_node: The node directly before the treated node
        :type prev_node: torch.fx.node.Node
        """
        for arg in self.args:
            if isinstance(arg, list):
                tmp = arg[1]
            else:
                tmp = arg
            # If the node is not instance of torch.fx, then it's a param numeric or other..
            if isinstance(tmp, torch.fx.node.Node) and tmp != prev_node:
                # ocmon avoid to have multiple time the same pop in the pop list. Because we can have several time
                # the same value in the params.
                ocmon = False
                for pop in self.pop_list:
                    if str(pop[0]) == str(tmp):
                        ocmon = True
                        break
                if not ocmon:
                    self.pop_list.append([tmp, str(tmp) + '_to_' + self.name])

    def _kwargs_handler(self):
        """Handle the kwargs.
        
        The kwargs are specific and need to be writen as "dim=1". 
        In this this function save the name of the parameter and his value in the list of args.
        """
        for key, kwarg in self.node.kwargs.items():
            self.args.append([key, kwarg])

    def _args_handler(self, trace, prev_node):
        """Handle the args.
        
        Args can be values but also the returned value of a past layer, so we have tocheck if the args is older than the direct previous or not. 
        Also if in the args there is a getattr or getitem we have to delete it with a call to _get_handler()
        :param trace: Is the complete trace of torch.fx of the model.
        :type trace: torch.fx.graph._node_list
        :param prev_node: The node directly before the treated node
        :type prev_node: torch.fx.node.Node
        """
        for arg in self.node.args:
            arg = self._get_handler(arg, trace)

            if arg == prev_node:
                self.args.append("input")

            else:
                self.args.append(arg)

    def generate_class(self) -> str:
        """Create a string containing the full class declaration for a layer.

        For example if there is any pop or stash to make : 
        .. code-block:: python
            @skippable(stash=[..], pop=[..])
        And then the class :
        .. code-block:: python
            class self.name_layer(nn.Module)

        :return: Return the class declaration
        :rtype: str
        """
        string = ""
        if len(self.stash_list) > 0 or len(self.pop_list) > 0:
            string += f"@skippable("

            if len(self.stash_list) > 0:
                string += "stash=["
                for stash in self.stash_list:
                    string += f"'{stash}', "
                string = string[:-2]
                string += "], "

            if len(self.pop_list) > 0:
                string += "pop=["
                for pop in self.pop_list:
                    string += f"'{pop[1]}', "
                string = string[:-2]
                string += "], "

            string = string[:-2]
            string += ")\n"

        string += f"class {self.name}_layer(nn.Module):\n"
        return string

    def generate_init(self, declaration) -> str:
        """Create a string containing the full __init__ function of the layer.

        For example :
        .. code-block:: python
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.DECLARATION

        The declaration can be for example nn.Linear(in_features=4, out_features=16, bias=True) or nn.parameter.Parameter(1, 16, requires_grad=True)

        :param declaration: Is the module or parameter to initialize
        :type declaration: str
        :return: Return the class __init__
        :rtype: str
        """
        string = TAB[1] + "def __init__(self) -> None:\n"
        string += TAB[2] + "super().__init__()\n"
        string += TAB[2] + f"self.fc = nn.{declaration}\n"

        return string

    def reset_separation_layer(self):
        self.separation_layer = False

    def set_separation_layer(self):
        """Set the layer as a separation layer.
        
        This is needed because when the data change from a GPU to another he needs to be cloned.

        So this adds to the forward declaration : input = input.clone()
        """
        self.separation_layer = True

    def get_separation_layer(self) -> bool:
        """Return the separation status of the layer.
        
        :return: True if the layer is a separation layer, else False
        :rtype: bool
        """
        return self.separation_layer

    def generate_forward(self, task) -> str:
        """Create the full forward definition.
        
        He has to first, if needed, pop all the params sent from older layer. 
        Secondly clone the input if it is a separation layer. 
        Thirdly do his task and affect it to the ret variable. A task can be some different thing, for example the execution of the Module declarated in the __init__ (self.fc) or simply a function call (torch.add).
        Fourthly it will add, if defined, a getitem to the executed task.  So if the task return Tuple and only the first one is used it will add at the end a [0]. 
        Fifthly he have to stash is ret value if anyone need it further.
        And finaly return is value for the next layer.

        A full forward can look as :
        .. code-block:: python
            def forward(self, input):
                add = yield pop('add_to_{self.name}')
                input = input.clone()
                ret = torch.add(input, add)
                yield stash('{self.name}_to_another', ret)
                return ret

        :param task: Task to be done in the forward
        :type task: str
        :return: Return the full forward declaration
        :rtype: str
        """
        string = TAB[1] + "def forward(self, input):\n"
        for pop in self.pop_list:
            string += TAB[2] + f"{pop[0]} = yield pop('{pop[1]}')\n"

        if self.separation_layer:
            string += TAB[2] + "input = input.clone()\n"

        string += TAB[2] + f"ret = {task}"

        if self.getitem_idx is not None:
            string += f"[{self.getitem_idx}]"

        string += "\n"

        for stash in self.stash_list:
            string += TAB[2] + f"yield stash('{stash}', ret)\n"

        if self.node.op == "placeholder":
            string += TAB[2] + f"return input\n\n"
        else:
            string += TAB[2] + f"return ret\n\n"

        return string

    def add_stash(self, node_to):
        """Add stash information for special layer.
        
        The getattr are ignore during the parsing, so if one of those depend on layer we have to add their parent to the stash list to not lose the connection.
        :param node_to: Parent of a getattr
        :type node_to: torch.fx.node.Node
        """
        self.stash_list.append(self.name + "_to_" + str(node_to))

    def add_getitem(self, idx):
        """Allow to add a getitem to the task of the layer.
        
        Because the getitem cannot be considered as layer, we have to link the getitem to the layer with who he is linked.
        :param idx: Index of the getitem
        :type idx: int
        """
        self.getitem_idx = idx

    def __str__(self) -> str:
        """Allow to print easily all the information of a layer.

        :return: String to print
        :rtype: str
        """
        print_str = "---- Layer information ----\n"
        print_str += f"    Name : {self.name}\n"
        print_str += f"    Argument : {self.args}\n"

        if len(self.pop_list) > 0:
            print_str += f"    Pop info :\n"

            for pop, stashed_from in self.pop_list:
                print_str += f"        Argument {pop}, Stashed from {stashed_from}\n"

        if len(self.stash_list) > 0:
            print_str += "    Stash info : \n"

            for stash in self.stash_list:
                print_str += f"        ret is stashed for {stash}\n"
        if self.getitem_idx is not None:
            print_str += f"    This node will access the result of ret with [{self.getitem_idx}]\n"
        return print_str
