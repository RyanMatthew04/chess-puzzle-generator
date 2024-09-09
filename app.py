from flask import Flask, jsonify,send_from_directory
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/generate_fen', methods=['GET'])
def generate_fen():
    # Run your Python script and get the FEN string
    try:
        result = subprocess.run(['python', 'generate.py'], capture_output=True, text=True)
        fen_string = result.stdout.strip()

        if not fen_string:
                raise ValueError("Received empty FEN from generate.py")
            
        return jsonify({'fen': fen_string})
    
    except subprocess.CalledProcessError as e:
        print(f"Subprocess error: {e}")
        return jsonify({'fen': e}), 500
    except ValueError as e:
        print(f"Value error: {e}")
        return jsonify({'fen': e}), 500
    
    

if __name__ == '__main__':
    app.run(debug=True)
    