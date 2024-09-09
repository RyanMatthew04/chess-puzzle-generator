from flask import Flask, jsonify,send_from_directory
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/generate_fen', methods=['GET'])
def generate_fen():
    # Run your Python script and get the FEN string
    result = subprocess.run(['python', 'generate.py'], capture_output=True, text=True)
    fen_string = result.stdout.strip()
    
    return jsonify({'fen': fen_string})

if __name__ == '__main__':
    app.run(debug=True,port=5500)
    