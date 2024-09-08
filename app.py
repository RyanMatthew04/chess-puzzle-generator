from flask import Flask, jsonify
import subprocess

app = Flask(__name__)

@app.route('/generate_fen', methods=['GET'])
def generate_fen():
    # Run your Python script and get the FEN string
    result = subprocess.run(['python', 'generate.py'], capture_output=True, text=True)
    fen_string = result.stdout.strip()
    print(fen_string)
    
    return jsonify({'fen': fen_string})

if __name__ == '__main__':
    app.run(debug=True,port=5500)
    