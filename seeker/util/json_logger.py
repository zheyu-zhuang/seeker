"""JSON-lines logging helpers for training metrics.

License note:
This module is adapted from the open-source Diffusion Policy codebase
(MIT License), with local project-specific modifications.
"""

from typing import Optional, Callable, Any, Sequence
import os
import copy
import json
import numbers
import pandas as pd


def read_json_log(path: str, 
        required_keys: Sequence[str]=tuple(), 
        **kwargs) -> pd.DataFrame:
    """
    Read json-per-line file, with potentially incomplete lines.
    kwargs passed to pd.read_json
    """
    lines = list()
    with open(path, 'r') as f:
        while True:
            # one json per line
            line = f.readline()
            if len(line) == 0:
                # EOF
                break
            elif not line.endswith('\n'):
                # incomplete line
                break
            is_relevant = False
            for k in required_keys:
                if k in line:
                    is_relevant = True
                    break
            if is_relevant:
                lines.append(line)
    if len(lines) < 1:
        return pd.DataFrame()  
    json_buf = f'[{",".join([line for line in (line.strip() for line in lines) if line])}]'
    df = pd.read_json(json_buf, **kwargs)
    return df

class JsonLogger:
    def __init__(self, path: str, 
            filter_fn: Optional[Callable[[str,Any],bool]]=None):
        if filter_fn is None:
            filter_fn = lambda k,v: isinstance(v, numbers.Number)

        # default to append mode
        self.path = path
        self.filter_fn = filter_fn
        self.file = None
        self.last_log = None
    
    def start(self):
        # use line buffering
        try:
            self.file = file = open(self.path, 'r+', buffering=1)
        except FileNotFoundError:
            self.file = file = open(self.path, 'w+', buffering=1)

        # Recover the last complete JSONL record and discard a partial trailing record.
        pos = file.seek(0, os.SEEK_END)

        while pos > 0 and file.read(1) != "\n":
            pos -= 1
            file.seek(pos, os.SEEK_SET)
        last_line_end = file.tell()
        
        pos = max(0, pos-1)
        file.seek(pos, os.SEEK_SET)
        while pos > 0 and file.read(1) != "\n":
            pos -= 1
            file.seek(pos, os.SEEK_SET)
        last_line_start = file.tell()

        if last_line_start < last_line_end:
            last_line = file.readline()
            self.last_log = json.loads(last_line)
        
        file.seek(last_line_end)
        file.truncate()
    
    def stop(self):
        self.file.close()
        self.file = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def log(self, data: dict):
        filtered_data = dict(
            filter(lambda x: self.filter_fn(*x), data.items()))
        # save current as last log
        self.last_log = filtered_data
        for k, v in filtered_data.items():
            if isinstance(v, numbers.Integral):
                filtered_data[k] = int(v)
            elif isinstance(v, numbers.Number):
                filtered_data[k] = float(v)
        buf = json.dumps(filtered_data)
        # ensure one line per json
        buf = buf.replace('\n','') + '\n'
        self.file.write(buf)
    
    def get_last_log(self):
        return copy.deepcopy(self.last_log)
