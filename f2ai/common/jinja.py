import os
from jinja2 import Environment, FileSystemLoader, select_autoescape

jinja_env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "../templates")), autoescape=select_autoescape)