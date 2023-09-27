from torch.fx import symbolic_trace
import torch
# import math
from pathlib import Path
from .constant import TAB
# import os
import sys
# from gpu_alloc import TraceMalloc
# from dataset import PipelineDataset
# import multiprocessing
# import torch.multiprocessing as tmp
import subprocess
# from pytorch_memlab import MemReporter

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
        self.nb_gpu = nb_gpus if nb_gpus is not None else torch.cuda.device_count()
        self.file_name = "layered_model.py"
        self.directory_name = "pipelinecache"
        self.configs_mha = configs_mha
        self.mha_number = len(self.configs_mha)
        self.mha_count = 0
        self.input_shape = input_shape
        self.output_shape = output_shape

        # self.trace_gpu_alloc = TraceMalloc(nb_gpus)
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

    def _equilibration(self, layers, memory):
        """Balance the distribution of layers across GPUs based on memory usage.

        :param layers: Current distribution of layers across GPUs.
        :type layers: list
        :param memory: Memory usage for each GPU.
        :type memory: list
        :return: New balanced distribution of layers across GPUs.
        :rtype: list
        """
        repartition = layers.copy()

        n = len(layers)

        upper_idx = max(range(n), key=lambda i: memory[i])
        lower_idx = min(range(n), key=lambda i: memory[i])

        repartition[lower_idx] += 1
        repartition[upper_idx] -= 1

        return repartition

    def reset_repartition(self, layer_per_gpu):
        """Reset the distribution of layers based on the number of layers per GPU.

        :param layer_per_gpu: Number of layers per GPU.
        :type layer_per_gpu: list
        """
        current_layer = 0
        gpu_index = 0
        separation_layer_index = layer_per_gpu[gpu_index] - 1

        for _, layer in self.LayerLists.items():
            if current_layer >= len(self.LayerLists.items()) - 1:
                break

            if separation_layer_index == current_layer:
                layer.reset_separation_layer()
                gpu_index += 1
                separation_layer_index += layer_per_gpu[gpu_index]

            current_layer += 1

    def set_repartition(self, layer_per_gpu):
        """Set the distribution of layers across GPUs based on the number of layers per GPU.

        :param layer_per_gpu: Number of layers per GPU.
        :type layer_per_gpu: list
        """
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

    def _check_memory_peak(self, memory_peak):
        """Check if the memory peaks are balanced across GPUs.

        :param memory_peak: Memory peaks for each GPU.
        :type memory_peak: list
        :return: True if memory peaks are balanced, False otherwise.
        :rtype: bool
        """
        threshold = 0.2  # 20% de tolérance
        reference_value = memory_peak[0]  # Sélection d'une valeur de référence aléatoire dans le tableau
        return all(abs(value - reference_value) <= reference_value * threshold for value in memory_peak[1:])


    def _repartition(self):
        """Perform the distribution of layers across GPUs in a balanced manner based on memory usage."""
        self._init_file()
        # Save self var for remake
        file = self.file

        # Calculate first naive repartition on gpus
        clone_step = len(self.LayerLists.items()) // self.nb_gpu
        remainder = len(self.LayerLists.items()) % self.nb_gpu
        layer_per_gpu = [clone_step] * self.nb_gpu

        # Distribuer les couches restantes entre les GPU
        for i in range(remainder):
            layer_per_gpu[i] += 1

        # Initialise cloned layers
        self.set_repartition(layer_per_gpu)

        # Write in file the naive repartition
        self._write_in_file()

        # print(f"First repartition : {layer_per_gpu}")
        dir_path = Path(__file__).resolve().parent / "evaluate_mem.py"

        previous_repartitions = []

        python_exec = sys.executable

        while True:
            p = subprocess.run([python_exec, dir_path,
                            '--input_shape', str(list(self.input_shape)),
                            '--output_shape', str(list(self.output_shape)),
                            '--number_gpu', str(int(self.nb_gpu)),
                            '--number_chunks', str(2)], capture_output=True, text=True)

            result = p.stdout
            result = result.replace("[", "").replace("]", "")
            result = result.split(",")
            # print(result)
            memory_peak = [int(x.strip()) for x in result]

            # print(f"Memory peaks per GPU: {[peak for peak in memory_peak]} MB")

            if not self._check_memory_peak(memory_peak):
                # print("GPU are not equilibrated!")
                new_layer_per_gpu = self._equilibration(layer_per_gpu, memory_peak)
                # print(f"New repartition calculated : {new_layer_per_gpu}")

                if new_layer_per_gpu in previous_repartitions:
                    # print("Oscillation detected, exiting the loop.")
                    break

                previous_repartitions.append(new_layer_per_gpu)

                self.file = file
                self.reset_repartition(layer_per_gpu)
                self.set_repartition(new_layer_per_gpu)
                self._write_in_file()
                layer_per_gpu = new_layer_per_gpu

            else:
                break

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

        self._repartition()
