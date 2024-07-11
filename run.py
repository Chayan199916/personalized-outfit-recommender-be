from app import create_app
from pyngrok import ngrok
import os

app = create_app()

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    # Set up ngrok
    public_url = ngrok.connect(5000)
    print(' * Tunnel URL:', public_url)
    app.run(port=5000)
