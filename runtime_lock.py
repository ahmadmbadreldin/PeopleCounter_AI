"""Process-wide single-instance lock for one counter per branch."""

import ctypes
import os

from branch_config import safe_branch_id


class BranchInstanceLock:
    def __init__(self, branch_id):
        self.handle = None
        self.acquired = True
        if os.name != "nt":
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_bool
        name = f"Local\\ColorsBeautyCounter_{safe_branch_id(branch_id)}"
        self.handle = kernel32.CreateMutexW(None, False, name)
        self.acquired = bool(self.handle) and ctypes.get_last_error() != 183
        if self.handle and not self.acquired:
            kernel32.CloseHandle(self.handle)
            self.handle = None

    def close(self):
        if os.name == "nt" and self.handle:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_bool
            kernel32.CloseHandle(self.handle)
            self.handle = None
