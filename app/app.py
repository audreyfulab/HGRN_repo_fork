from flask import Flask
from config import Config
from routes.hcd import hcd_blueprint

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    app.register_blueprint(hcd_blueprint)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000,host='0.0.0.0',)