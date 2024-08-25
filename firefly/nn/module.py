from __future__ import annotations
from typing import Any
from firefly.nn.parameter import Parameter
from warnings import warn


class Module:

    _parameters: (
        dict[str, Parameter | list[Parameter], tuple[Parameter, ...]] | None
    ) = None
    _modules: dict[str, Module | list[Module] | tuple[Module, ...]] | None = None

    def _setup(self):
        if self._parameters is None:
            object.__setattr__(self, "_parameters", {})
        if self._modules is None:
            object.__setattr__(self, "_modules", {})

    def __setattr__(self, key, value):
        self._setup()
        if key in {"_parameters", "_modules"}:
            raise ValueError(f"Cannot set attribute {key}.")
        if key in self._parameters:
            del self._parameters[key]
        if key in self._modules:
            del self._modules[key]
        if isinstance(value, (list, tuple)):
            n_params = len([i for i in value if isinstance(i, Parameter)])
            n_modules = len([i for i in value if isinstance(i, Module)])
            if n_params == len(value):
                self._parameters[key] = value
            elif n_modules == len(value):
                self._modules[key] = value
            elif n_modules > 0 or n_params > 0:
                warn(f"Mixed types in `{key}`. Parameters won't be registered.")
        if isinstance(value, Parameter):
            self._parameters[key] = value
        elif isinstance(value, Module):
            self._modules[key] = value
        object.__setattr__(self, key, value)

    def get_parameters(self, discard_non_trainable: bool = True) -> list[Parameter]:
        self._setup()
        parameters = []
        for parameter in self._parameters.values():
            if not isinstance(parameter, Parameter):
                for p in parameter:
                    if not discard_non_trainable or p.trainable:
                        parameters.append(p)
            elif not discard_non_trainable or parameter.trainable:
                parameters.append(parameter)
        for module in self._modules.values():
            if not isinstance(module, Module):
                for m in module:
                    parameters.extend(m.get_parameters(discard_non_trainable))
            else:
                parameters.extend(module.get_parameters(discard_non_trainable))
        return list(set((parameters)))

    def _gather_state_dict(self, prefix: str = "") -> dict:
        self._setup()
        state_dict = {}
        for key, parameter in self._parameters.items():
            full_key = f"{prefix}{key}"
            if isinstance(parameter, Parameter):
                state_dict[full_key] = parameter.get_state()
            else:
                for i, p in enumerate(parameter):
                    state_dict[f"{full_key}.{i}"] = p.get_state()
        for key, module in self._modules.items():
            full_key = f"{prefix}{key}."
            if isinstance(module, Module):
                state_dict.update(module._gather_state_dict(full_key))
            else:
                for i, m in enumerate(module):
                    state_dict.update(m._gather_state_dict(f"{full_key}{i}."))
        return state_dict

    def load_state_dict(self, state_dict: dict, prefix: str = ""):
        self._setup()
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue
            subkey = key[len(prefix) :]
            if "." in subkey:
                main_key, sub_subkey = subkey.split(".", 1)
                if main_key in self._modules:
                    module = self._modules[main_key]
                    if isinstance(module, Module):
                        module.load_state_dict(state_dict, f"{prefix}{main_key}.")
                    else:
                        index, sub_subkey = sub_subkey.split(".", 1)
                        module[int(index)].load_state_dict(
                            state_dict, f"{prefix}{main_key}.{index}."
                        )
                elif main_key in self._parameters:
                    param = self._parameters[main_key]
                    if isinstance(param, Parameter):
                        param.set_state(value)
                    else:
                        index = int(sub_subkey)
                        param[index].set_state(value)
            else:
                if subkey in self._parameters:
                    param = self._parameters[subkey]
                    if isinstance(param, Parameter):
                        param.set_state(value)
                    else:
                        for p in param:
                            p.set_state(value)

    def state_dict(self) -> dict:
        return self._gather_state_dict()

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Forward method not implemented.")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def to(self, backend: str):
        for param in self.get_parameters(False):
            param.tensor.to(backend)
        return self

    def train_mode(self):
        for param in self.get_parameters(True):
            param.tensor.requires_grad = True

    def eval_mode(self):
        for param in self.get_parameters(True):
            param.tensor.requires_grad = False
