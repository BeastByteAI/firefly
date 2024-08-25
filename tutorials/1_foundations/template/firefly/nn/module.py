from __future__ import annotations
from firefly.nn.parameter import Parameter


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

    def forward(self, *ars, **kwargs) -> any:
        raise NotImplementedError("Forward method is not implemented.")

    def __call__(self, *args):
        return self.forward(*args)

    def __setattr__(self, key, value):
        # TODO
        raise NotImplementedError("This method needs to be implemented.")

    def get_parameters(self, discard_non_trainable: bool = True) -> list[Parameter]:
        # TODO
        raise NotImplementedError("This method needs to be implemented.")
    

    ### The following methods are optional and are not further used in the tutorial ###

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
                    state_dict.update(m._gather_state_dict(f"{full_key}.{i}."))
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

    def train_mode(self):
        for param in self.get_parameters(True):
            param.tensor.requires_grad = True

    def eval_mode(self):
        for param in self.get_parameters(True):
            param.tensor.requires_grad = False
