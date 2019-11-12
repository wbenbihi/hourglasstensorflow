import os
import configparser
from configparser import InterpolationMissingOptionError


class ConfigSection:
    def __init__(self, name, section):
        self.name = name
        self.section = section
        self._set_items()

    def keys(self):
        return list(self.section.keys())

    def items(self):
        return [
            (key, value) for key, value in self.__dict__.items() if key in self.keys()
        ]

    def _set_items(self):
        for key in self.section.keys():
            try:
                setattr(self, key, eval(self.section.get(key, raw=True)))
            except:
                setattr(self, key, self.section.get(key, raw=True))


class ConfigObject:
    def __init__(self, name, config_file):
        self.name = name
        self.config_file = config_file
        self.cfg = configparser.ConfigParser()
        self.cfg.read(self.config_file)
        self._set_section_attributes()

    def _set_section_attributes(self):
        for name, section in list(self.cfg.items()):
            setattr(self, name, ConfigSection(name, section))


class Config:
    def __init__(
        self,
        config_files: dict = {
            "main": "main_config.ini",
            "processor": "processor_config.ini",
        },
    ):
        self.config_files = config_files
        self._set_configparser_attributes()
        self._define_project_structure()

    def _set_configparser_attributes(self):
        for name, cfg_file in self.config_files.items():
            setattr(self, name, ConfigObject(name, cfg_file))

    def _define_project_structure(self):
        self.ROOT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


CFG = Config(
    {
        "default": os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "default","model_config.cfg"
        ),
    }
)
