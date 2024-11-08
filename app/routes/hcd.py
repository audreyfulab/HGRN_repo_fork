from flask import Blueprint, request, render_template, jsonify, send_file
import io
import networkx as nx



hcd_blueprint = Blueprint('hcd', __name__)

@hcd_blueprint.route('/')
def index():
    # return render_template('index.html')
    return render_template('index.html')

@hcd_blueprint.route('/query', methods=['POST'])
def query():
    """_summary_
    """