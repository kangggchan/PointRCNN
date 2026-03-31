import os, sys
_tools = os.path.dirname(os.path.abspath(__file__))
_root  = os.path.dirname(_tools)
sys.path.insert(0, _tools)                      # so 'import _init_path' works from any cwd
sys.path.insert(0, _root)                       # so 'import lib.*' works
sys.path.insert(0, os.path.join(_root, 'lib/datasets'))
sys.path.insert(0, os.path.join(_root, 'lib/net'))
