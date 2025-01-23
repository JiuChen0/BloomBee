import os
import sys
import importlib


_replace_modules = {}


class Empty:
    def __init__(self, *args):
        pass


class CustomModuleLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):

    def __init__(self):
        self._cache = {}
        self._original_finders = sys.meta_path.copy()

    def find_spec(self, fullname, path, target=None):
        if fullname in _replace_modules:
            replace_module_path = _replace_modules[fullname]
            if not os.path.exists(replace_module_path):
                raise ImportError(
                    f"Replacement module not found: {replace_module_path}"
                )
            return importlib.util.spec_from_file_location(fullname, replace_module_path)
        return None

    def exec_module(self, module):
        """Execute module"""
        if module.__spec__.name in _replace_modules:
            with open(module.__spec__.origin, "r", encoding="utf-8") as f:
                code = compile(f.read(), module.__spec__.origin, "exec")
                exec(code, module.__dict__)
            self._cache[module.__spec__.name] = module
        else:
            # Fallback to original import mechanism
            finder = sys.meta_path.pop(0)
            try:
                module = importlib.import_module(module.__spec__.name)
                self._cache[module.__spec__.name] = module
            finally:
                sys.meta_path.insert(0, finder)

    def load_module(self, fullname):
        """Load module with enhanced error handling"""
        try:
            if fullname in self._cache:
                return self._cache[fullname]

            spec = self.find_spec(fullname, None)
            if spec is None:
                raise ImportError(f"Module {fullname} not found")

            module = importlib.util.module_from_spec(spec)

            sys.modules[fullname] = module
            self.exec_module(module)
            return module

        except Exception as e:
            raise ImportError(f"Failed to load module {fullname}: {str(e)}") from e
