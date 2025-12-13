import os
import yaml
import json
import copy
import argparse
import utils.tools as tl

class Config(object):
    """
    Global config object. 
    It automatically loads from a hierarchy of config files and turns the keys to the 
    class attributes. 
    """
    def __init__(self, load=True, cfg_dict=None, cfg_level=None):
        """
        Args: 
            load (bool): whether or not yaml is needed to be loaded.
            cfg_dict (dict): dictionary of configs to be updated into the attributes
            cfg_level (int): indicating the depth level of the config
        """
        self._level = "cfg" + ("." + cfg_level if cfg_level is not None else "")
        if load:
            self.args = self._parse_args()
            print("Loading config from {}.".format(self.args.cfg_file))
            self.need_initialization = True
            cfg_base = self._initialize_cfg()
            cfg_dict = self._load_yaml(self.args)
            cfg_dict = self._merge_cfg_from_base(cfg_base, cfg_dict)
            self.cfg_dict = cfg_dict
        self._update_dict(cfg_dict)
        if load:
            tl.make_checkpoint_dir(self.OUTPUT_DIR)
            tl.make_checkpoint_dir(self.OUTPUT_DIR_SAVE)

    def _parse_args(self):
        """
        Wrapper for argument parser. 
        """
        parser = argparse.ArgumentParser(
            description="Argparser for configuring [code base name to think of] codebase"
        )
        parser.add_argument(
            "--cfg",
            dest="cfg_file",
            help="Path to the configuration file",
            default=None
        )
        return parser.parse_args()

    def _path_join(self, path_list):
        """
        Join a list of paths.
        Args:
            path_list (list): list of paths.
        """
        path = ""
        for p in path_list:
            path+= p + '/'
        return path[:-1]

    def _initialize_cfg(self):
        """
        When loading config for the first time, base config is required to be read.
        """
        if self.need_initialization:
            self.need_initialization = False
            if os.path.exists('./configs/base.yaml'):
                with open("./configs/base.yaml", 'r') as f:
                    cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return cfg
    
    def _load_yaml(self, args, file_name=""):
        """
        Load the specified yaml file.
        Args:
            args: parsed args by `self._parse_args`.
            file_name (str): the file name to be read from if specified.
        """
        assert args.cfg_file is not None
        if not file_name == "": # reading from base file
            with open(file_name, 'r') as f:
                cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        else: # reading from top file
            with open(args.cfg_file, 'r') as f:
                cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
                file_name = args.cfg_file
        return cfg
    
    def _merge_cfg_from_base(self, cfg_base, cfg_new, preserve_base=False):
    
        for k,v in cfg_new.items():
            if k in cfg_base.keys():
                if isinstance(v, dict):
                    self._merge_cfg_from_base(cfg_base[k], v)
                else:
                    cfg_base[k] = v
            else:
                if "BASE" not in k or preserve_base:
                    cfg_base[k] = v
        return cfg_base
    
    def _update_dict(self, cfg_dict):
        """
        Set the dict to be attributes of the config recurrently.
        Args:
            cfg_dict (dict): the dictionary to be set as the attribute of the current 
                config class.
        """
        def recur(key, elem):
            if type(elem) is dict:
                return key, Config(load=False, cfg_dict=elem, cfg_level=key)
            else:
                if type(elem) is str and elem[1:3]=="e-":
                    elem = float(elem)
                return key, elem
        dic = dict(recur(k, v) for k, v in cfg_dict.items())
        self.__dict__.update(dic)
    
    def get_args(self):
        """
        Returns the read arguments.
        """
        return self.args
    
    def __repr__(self):
        return "{}\n".format(self.dump())
            
    def dump(self):
        return json.dumps(self.cfg_dict, indent=2)

    def deep_copy(self):
        return copy.deepcopy(self)
    
if __name__ == '__main__':
    cfg = Config(load=True)
    print(cfg.DATA)
    