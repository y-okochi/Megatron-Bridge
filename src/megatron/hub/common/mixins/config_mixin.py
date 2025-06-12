import json
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type, TypeVar, Union

import yaml
from omegaconf import OmegaConf

from megatron.hub.core.utils.instantiate_utils import InstantiationMode, instantiate
from megatron.hub.core.utils.yaml_utils import safe_yaml_representers

# For TOML support
try:
    import toml

    HAS_TOML = True
except ImportError:
    HAS_TOML = False


T = TypeVar("T", bound="ConfigMixin")
ConfigFormat = Literal["yaml", "json", "toml"]


class ConfigMixin:
    """
    A mixin class that adds from_pretrained and save_pretrained methods
    similar to HuggingFace's interface. Integrates with OmegaConf and
    the existing ConfigContainer/instantiate infrastructure.

    Supports JSON, YAML, and TOML formats (JSON and TOML are converted to YAML).
    """

    # Default filename for saving/loading configuration
    CONFIG_NAME = "config"  # Will have .yaml, .json, or .toml extension
    DEFAULT_CONFIG_FORMAT: ConfigFormat = "json"

    @classmethod
    def from_pretrained(
        cls: Type[T],
        pretrained_model_name_or_path: Union[str, Path],
        trust_remote_code: bool = False,
        mode: InstantiationMode = InstantiationMode.LENIENT,
        **kwargs,
    ) -> T:
        """
        Load a pretrained model configuration from a directory or file.

        Args:
            pretrained_model_name_or_path: Path to a directory containing a config file,
                                          or direct path to a config file (yaml/json/toml)
            trust_remote_code: Whether to trust and execute code references (classes/functions)
                              found in the configuration. Required to be True if the config
                              contains any class or function references. Default: False
            mode: Instantiation mode (STRICT or LENIENT) for the instantiate function
            **kwargs: Additional keyword arguments to override loaded configuration

        Returns:
            Instance of the class with loaded configuration

        Example:
            ```python
            # Load from directory (looks for config.yaml, config.json, or config.toml)
            model = MyModel.from_pretrained("./saved_model/")

            # Load from specific file
            model = MyModel.from_pretrained("./saved_model/config.yaml")

            # With code references
            model = MyModel.from_pretrained("./saved_model/", trust_remote_code=True)

            # Override configuration values
            model = MyModel.from_pretrained("./saved_model/", temperature=0.8)
            ```
        """
        path = Path(pretrained_model_name_or_path)

        # Determine the config file path
        if path.is_dir():
            # Look for config files in order of preference
            config_file = None
            for ext in [".yaml", ".yml", ".json", ".toml"]:
                candidate = path / f"{cls.CONFIG_NAME}{ext}"
                if candidate.exists():
                    config_file = candidate
                    break

            if config_file is None:
                raise FileNotFoundError(
                    f"No configuration file found in {path}. "
                    f"Expected {cls.CONFIG_NAME}.yaml, {cls.CONFIG_NAME}.json, or {cls.CONFIG_NAME}.toml"
                )
        else:
            config_file = path

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_file}")

        # Load the configuration based on file extension
        file_ext = config_file.suffix.lower()

        if file_ext in [".yaml", ".yml"]:
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        elif file_ext == ".json":
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        elif file_ext == ".toml":
            if not HAS_TOML:
                raise ImportError(
                    "TOML support requires the 'toml' package. Install it with: pip install toml"
                )
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = toml.load(f)
        else:
            raise ValueError(
                f"Unsupported file format: {file_ext}. Supported formats: .yaml, .yml, .json, .toml"
            )

        # Check for trust_remote_code requirement
        if not trust_remote_code and cls._contains_code_references(config_dict):
            raise ValueError(
                "This configuration contains class or function references. "
                "Loading it requires trust_remote_code=True to prevent arbitrary code execution."
            )

        # Convert to OmegaConf for compatibility with instantiate
        omega_conf = OmegaConf.create(config_dict)

        # Merge with kwargs
        if kwargs:
            override_conf = OmegaConf.create(kwargs)
            omega_conf = OmegaConf.merge(omega_conf, override_conf)

        # Add _target_ if not present
        if "_target_" not in omega_conf:
            omega_conf["_target_"] = f"{cls.__module__}.{cls.__qualname__}"

        # Convert back to container for instantiate
        final_config = OmegaConf.to_container(omega_conf, resolve=True)

        # Use instantiate to create the object
        return instantiate(final_config, mode=mode)

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        config_format: ConfigFormat | None = None,
        config_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Save the model configuration to a directory.

        Args:
            save_directory: Directory where to save the configuration
            config_format: Format to save in ("yaml", "json", or "toml"). Default: "yaml"
            config_name: Optional custom name for the config file (without extension)
            **kwargs: Additional metadata to save alongside the configuration

        Example:
            ```python
            # Save as YAML (default)
            model.save_pretrained("./saved_model/")

            # Save as JSON
            model.save_pretrained("./saved_model/", config_format="json")

            # Save with custom name
            model.save_pretrained("./saved_model/", config_name="my_config")
            ```
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        if not config_format:
            config_format = self.DEFAULT_CONFIG_FORMAT

        # Determine file extension
        format_to_ext = {"yaml": ".yaml", "yml": ".yaml", "json": ".json", "toml": ".toml"}

        config_format = config_format.lower()
        if config_format not in format_to_ext:
            raise ValueError(
                f"Unsupported format: {config_format}. "
                f"Supported formats: {list(format_to_ext.keys())}"
            )

        if config_format == "toml" and not HAS_TOML:
            raise ImportError(
                "TOML support requires the 'toml' package. Install it with: pip install toml"
            )

        config_name = config_name or self.CONFIG_NAME
        config_file = save_path / f"{config_name}{format_to_ext[config_format]}"

        # Get the configuration dictionary
        config_dict = self._to_dict()

        # Add any additional metadata
        if kwargs:
            config_dict.update(kwargs)

        # Save based on format
        if config_format in ["yaml", "yml"]:
            with safe_yaml_representers():
                with open(config_file, "w", encoding="utf-8") as f:
                    yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif config_format == "json":
            # First convert to YAML string to use the custom representers
            with safe_yaml_representers():
                yaml_str = yaml.safe_dump(config_dict, default_flow_style=False)
            # Then parse and save as JSON
            yaml_dict = yaml.safe_load(yaml_str)
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(yaml_dict, f, indent=2, ensure_ascii=False)
        elif config_format == "toml":
            # First convert to YAML string to use the custom representers
            with safe_yaml_representers():
                yaml_str = yaml.safe_dump(config_dict, default_flow_style=False)
            # Then parse and save as TOML
            yaml_dict = yaml.safe_load(yaml_str)
            with open(config_file, "w", encoding="utf-8") as f:
                toml.dump(yaml_dict, f)

        print(f"Configuration saved to {config_file}")

    def _to_dict(self) -> Dict[str, Any]:
        """
        Convert the instance to a dictionary representation.

        Returns:
            Dictionary representation of this instance
        """
        # Check if this is a ConfigContainer (has to_dict method)
        if (
            hasattr(self, "to_dict")
            and callable(self.to_dict)
            and self.__class__.__name__ != "ConfigMixin"
        ):
            return self.to_dict()

        # Otherwise, build dict from dataclass fields or attributes
        result = {}
        result["_target_"] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"

        if is_dataclass(self):
            # Handle dataclass
            for field in dataclass_fields(self):
                if field.name.startswith("_"):
                    continue
                value = getattr(self, field.name)
                result[field.name] = self._convert_value_to_dict(value)
        else:
            # Handle regular class
            for key, value in self.__dict__.items():
                if not key.startswith("_"):
                    result[key] = self._convert_value_to_dict(value)

        return result

    def _convert_value_to_dict(self, value: Any) -> Any:
        """
        Recursively convert a value to a dictionary representation.

        Args:
            value: The value to convert

        Returns:
            The converted value
        """
        # Use ConfigContainer's conversion if available
        if (
            hasattr(self.__class__, "_convert_value_to_dict")
            and self.__class__.__name__ != "ConfigMixin"
        ):
            return self.__class__._convert_value_to_dict(value)

        # Otherwise, basic conversion
        if hasattr(value, "_to_dict"):
            return value._to_dict()
        elif hasattr(value, "to_dict") and callable(value.to_dict):
            return value.to_dict()
        elif is_dataclass(value) and not isinstance(value, type):
            # Handle regular dataclasses
            result = {"_target_": f"{value.__class__.__module__}.{value.__class__.__qualname__}"}
            for field in dataclass_fields(value):
                if not field.name.startswith("_"):
                    result[field.name] = self._convert_value_to_dict(getattr(value, field.name))
            return result
        elif isinstance(value, (list, tuple)):
            return [self._convert_value_to_dict(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._convert_value_to_dict(v) for k, v in value.items()}
        else:
            return value

    @staticmethod
    def _contains_code_references(config_dict: Dict[str, Any]) -> bool:
        """
        Check if a configuration dictionary contains code references.

        Args:
            config_dict: The configuration dictionary to check

        Returns:
            True if code references are found, False otherwise
        """
        if isinstance(config_dict, dict):
            for key, value in config_dict.items():
                # Check for _target_ that's not a built-in type
                if key == "_target_" and isinstance(value, str):
                    # Consider it a code reference if it's not a basic type
                    if not value.startswith(
                        ("builtins.", "str", "int", "float", "bool", "list", "dict", "tuple")
                    ):
                        return True
                # Check for _call_ = False which indicates a code reference
                if key == "_call_" and value is False:
                    return True
                # Recursively check nested structures
                if ConfigMixin._contains_code_references(value):
                    return True
        elif isinstance(value, (list, tuple)):
            for item in value:
                if ConfigMixin._contains_code_references(item):
                    return True

        return False
