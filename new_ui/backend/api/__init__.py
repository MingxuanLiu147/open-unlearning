from flask import Blueprint

config_bp = Blueprint('config', __name__)
runner_bp = Blueprint('runner', __name__)
results_bp = Blueprint('results', __name__)
skills_bp = Blueprint('skills', __name__)
agent_bp = Blueprint('agent', __name__)
data_bp = Blueprint('data', __name__)

from . import config, runner, results, skills, agent, data  # noqa: E402, F401
