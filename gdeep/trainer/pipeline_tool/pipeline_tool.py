from torch.fx import symbolic_trace
import torch
import math
from pathlib import Path
from .constant import TAB

class SkippableTracing:
    """Create and sequence the model parallelism.

    He will parsed the model given by the user and generate a file that contain all the splitted Layer and divide them on the gpus.

    Example of use :
    .. code-block:: python
        trace = SkippableTracing(2, model)
        # Then just get the generated splitted model
        model = trace.get_modules()

    :param nb_gpus: Nb of gpus of work, if none given max gpus will be taken.
    :type nb_gpus: int
    :param model: Model to parsed
    :type model: a model extended of nn.Module.
    """

    def __init__(self, nb_gpus, model, input_shape, output_shape, configs_mha={}):
        """Constructor."""
        self.module_desc = {}
        self.file = ""
        self.net = None
        self.LayerLists = {}
        self.GetattrLists = {}
        self.nb_gpu = nb_gpus
        self.file_name = "layered_model.py"
        self.directory_name = "pipelinecache"
        self.configs_mha = configs_mha
        self.mha_number = len(self.configs_mha) 
        self.mha_count = 0
        self.input_shape = input_shape
        self.output_shape = output_shape

        self._verfiy_config_mha()

        self._tracer(model)

    def _verfiy_config_mha(self):
        """Verify if at least embed_dim and num_heads are present in the configuration given by the user.
        
        This two parameters are mandatory to create a MHA.
        """
        for config in self.configs_mha:
            try:
                config['embed_dim']
            except KeyError as e:
                print("You didn't provide embed_dim in one of your MHA config")
                exit()
            
            try:
                config['num_heads']
            except KeyError as e:
                print("You didn't provide num_heads in one of your MHA config")
                exit()

    def _write_in_file(self):
        """Write to the output file the generated Layers to use it from other files."""        
        dir_path = Path(__file__).resolve().parent / self.directory_name

        if not dir_path.exists():
            dir_path.mkdir(parents=True)

        file_path = dir_path / self.file_name

        with open(file_path, "w") as f:
            f.write(self.file)
            f.close()

    def get_modules(self) -> torch.nn.Sequential:
        """Allow the user to get the generated Sequential model for each GPU."""
        from .pipelinecache.layered_model import PipelinedModel

        model = PipelinedModel()
        return model.get_modules()

    def _init_file(self):
        """Add all necessary import to the file."""
        self.file += "import torch\n"
        self.file += "import torch.nn.functional as F\n"
        self.file += "import torch.nn as nn\n"
        self.file += "from torch.distributed.pipeline.sync.skip import stash, pop, skippable \n\n"

    def _generate_end_class(self):
        """Add a class at the end of the generated file to get simply the pipelined model."""
        self.file += f"class PipelinedModel(nn.Module):\n"
        self.file += TAB[1] + "def __init__(self) -> None:\n"
        self.file += TAB[2] + "super().__init__()\n"

        gpu_index = 0
        self.file += TAB[2] + f"self.s{gpu_index} = nn.Sequential("

        for layer in self.LayerLists.values():
            self.file += f"{layer.get_name()}_layer(), "

            if layer.get_separation_layer():
                self.file = self.file[:-2]
                self.file += f").cuda({gpu_index})\n"
                gpu_index = gpu_index + 1
                self.file += TAB[2] + f"self.s{gpu_index} = nn.Sequential("

        self.file = self.file[:-2]
        self.file += f").cuda({gpu_index})\n"

        self.file += TAB[1] + "def forward(self, input):\n"
        self.file += TAB[2] + f"ret = input\n"
        for gpu in range(self.nb_gpu):
            self.file += TAB[2] + f"ret = self.s{gpu}(ret.to({gpu}))\n"

        self.file += TAB[2] + "return ret\n"

        self.file += TAB[1] + "def get_modules(self):\n"
        self.file += TAB[2] + "return  nn.Sequential(*["
        for gpu in range(self.nb_gpu):
            self.file += f"nn.Sequential(*self.s{gpu}),"
        self.file = self.file[:-1]
        self.file += "])\n"

    def _create_mha(self):
        """Create MHA string declaration."""
        config = self.configs_mha[self.mha_count]

        decl = f"MultiheadAttention("

        for key, param in config.items():
            decl += f"{key}={param}, "

        decl = decl[:-2]
        decl += ")"
        self.mha_count = self.mha_count + 1

        return decl

    def _catch_module_desc(self):
        """Create a look-up dictionary to match target names with their declaration.
        
        We use the withelist.txt to know which module name we have to keep as "core" module.
        All the modules not present in the whitelist willbe digged to found their composition.

        MultiheadAttention are not parsed, so we have to do a little trick to found how is it configured, only based on giotto_deep implementation.
        """
        filename = Path(__file__).resolve().parent / 'whitelist.txt'
        whitelist = open(filename).readlines()
        whitelist = [line[:-1] for line in whitelist]  # ? use strip() instead ?

        for name, module in self.net.named_modules():
            if str(module).split('(', 1)[0] in whitelist:
                if str(module).split('(', 1)[0].find('Multi') >= 0:
                    try:
                        if self.mha_number >= 1:
                            self.module_desc[name] = self._create_mha()
                        else:
                            raise UserWarning(f"You didn't specified any MHA config, but at least one exist.")
                    except UserWarning as e:
                        print(f"Error : {e}") 
                        exit()
                    # self.module_desc[
                    #     name] = f"MultiheadAttention(embed_dim={self.net.config.hidden_size}, num_heads={self.net.config.num_attention_heads}, dropout={self.net.config.attention_probs_dropout_prob}, batch_first=True)"
                else:
                    self.module_desc[name] = module

    def allocate_layers(self, layers, values):
        num_gpus = len(layers)
        total_layers = sum(layers)
        total_values = sum(values)
        target_value = total_values / num_gpus

        if abs(total_layers - total_values) < 0.1 * total_values:
            # Les deux valeurs sont déjà proches
            return layers

        gpu_layers = [[] for _ in range(num_gpus)]
        gpu_sums = [0] * num_gpus

        # Parcours des valeurs entières et des layers par GPU dans l'ordre décroissant des valeurs entières
        for layer, value in sorted(zip(layers, values), key=lambda x: -x[1]):
            for i in range(num_gpus):
                if gpu_sums[i] + value <= target_value:
                    gpu_layers[i].append(layer)
                    gpu_sums[i] += value
                    break

        # Allouer toutes les valeurs restantes aux autres GPUs
        for i in range(num_gpus):
            if len(gpu_layers[i]) < len(values) // num_gpus:
                gpu_layers[i].extend(layers[len(gpu_layers[i]):])

        return gpu_layers

    def set_repartition(self, layer_per_gpu):
        current_layer = 0
        gpu_index = 0
        separation_layer_index = layer_per_gpu[gpu_index] - 1

        for _, layer in self.LayerLists.items():
            if current_layer >= len(self.LayerLists.items()) - 1:
                self.file += layer.get_declaration()
                break
            
            if separation_layer_index == current_layer:
                layer.set_separation_layer()
                gpu_index += 1

                separation_layer_index += layer_per_gpu[gpu_index]

            self.file += layer.get_declaration()
            current_layer += 1

        self._generate_end_class()

    def evaluate_mem(self):
        from .dataset import PipelineDataset
        model = self.get_modules()
        model = torch.distributed.pipeline.sync.Pipe(model, 2)

        dataset = PipelineDataset(1024, self.input_shape[1:], [1] if len(self.output_shape) == 1 else self.output_shape[1:])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.input_shape[0], shuffle=True)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        
        memory_peaks = [0] * self.nb_gpu

        for epoch in range(1):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                inputs = inputs.to(0)
                labels = labels.to(self.nb_gpu - 1)

                current_memory = [torch.cuda.memory_allocated(i) for i in range(self.nb_gpu)]

                outputs = model(inputs).local_value()

                # Forward pass
                loss = criterion(outputs, labels.squeeze())

                # Backward pass et mise à jour des poids
                loss.backward()
                
                memory_peak = [torch.cuda.memory_allocated(i) - current_memory[i] for i in range(self.nb_gpu)]
                for i, peak in enumerate(memory_peak):
                    if peak > memory_peaks[i]:
                        memory_peaks[i] = peak
                optimizer.step()

        return memory_peaks


    # Trace model and declare dependencies between layers (stash and pop)
    def _tracer(self, net):
        """Trace and create all the composite needed to describe correctly the models given.
        
        It will call the class of class_impl folder to generate at the end the correct file of the model splited between GPUs.
        :param net: Model to trace.
        :type net: a model extended of nn.Module.
        """
        self.net = net
        trace = symbolic_trace(net)
        self._catch_module_desc()

        prev_node = None

        from .class_impl.CallModule import CallModule
        from .class_impl.CallFunction import CallFunction
        from .class_impl.PropagationLayer import PropagationLayer
        from .class_impl.GetAttr import GetAttr
        from .class_impl.GetAttrModule import GetAttrModule

        # Iter through each node traced by torch.fx
        for node in trace.graph.nodes:
            if str(node).find("getitem") >= 0:
                for _node in trace.graph.nodes:
                    if node in _node.args:
                        if str(node.args[0]).find("getattr") >= 0:
                            self.GetattrLists[node.args[0]].add_getitem(node.args[1])
                        else:
                            self.LayerLists[node.args[0]].add_getitem(node.args[1])

            elif str(node).find("getattr") >= 0:
                self.GetattrLists[node] = GetAttr(node, trace.graph.nodes)

            else:
                if node.op == "call_module":
                    self.LayerLists[node] = CallModule(node, trace.graph.nodes, prev_node,
                                                       self.module_desc[node.target])

                elif node.op == "call_function" or node.op == "call_method":
                    self.LayerLists[node] = CallFunction(node, trace.graph.nodes, prev_node)

                elif node.op == "get_attr":
                    self.LayerLists[node] = GetAttrModule(node, trace.graph.nodes, prev_node, net)
                    pass

                else:
                    self.LayerLists[node] = PropagationLayer(node, trace.graph.nodes, prev_node)
                prev_node = node

        # For each getattr, we will update the Layer who are linked to it. If the value of the getattr need to be
        # stashed we will update the stash list of the parent of the getattr. And by default update all the argument
        # to have the correct declaration with the getattr.
        for _, getattr_item in self.GetattrLists.items():
            if getattr_item.is_stash_needed():
                self.LayerLists[getattr_item.get_parent()].add_stash(getattr_item.get_child())

            self.LayerLists[getattr_item.get_child()].update_arg_by_attr(getattr_item.get_parent(),
                                                                         getattr_item.get_attr_name())

        # As it is complicated to trace efficiently the stash we updated it with the poplist of each node.
        # So for each "pop parent" we set a stash for the current node.
        for layer in self.LayerLists.values():
            for pop_parent in layer.get_pop_parent():
                self.LayerLists[pop_parent[0]].set_stash(layer.get_node())

        self._init_file()

        # Calcul the split region for gpus.
        clone_step = math.floor((len(self.LayerLists.items())) / self.nb_gpu)
        
        # First naive repartition.
        layer_per_gpu = []
        for i in range(self.nb_gpu):
            layer_per_gpu.append(clone_step)
        self.set_repartition(layer_per_gpu)
        
        self._write_in_file()

        self.evaluate_mem()
    

