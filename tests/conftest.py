import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(root, "python"), os.path.join(root, "examples")]
