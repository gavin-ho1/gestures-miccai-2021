from pathlib import Path
from typing import List, Union

import torch


def sglob(
        path: Union[Path, str],
        pattern: str = '**/*'
        ) -> List[Path]:
    path = Path(path)
    return sorted(list(path.glob(pattern)))


def torch_load(path, **kwargs):
    if 'weights_only' not in kwargs:
        parts = torch.__version__.split('.')
        major, minor = int(parts[0]), int(parts[1])
        if major >= 3 or (major == 2 and minor >= 6):
            kwargs['weights_only'] = False
    return torch.load(path, **kwargs)
