from flask import Flask
from routes.face_recognition import face_recognition_bp
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Flask
app = Flask(__name__)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH'))

# Register the blueprint
app.register_blueprint(face_recognition_bp)

if __name__ == '__main__':
    host = os.getenv('FLASK_HOST')
    port = int(os.getenv('FLASK_PORT'))
    debug = os.getenv('DEBUG')
    app.run(debug=debug, host=host, port=port)
