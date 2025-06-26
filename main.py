import os
from dotenv import load_dotenv
from flask import Flask

from genkit.ai import Genkit
from genkit.plugins.flask import genkit_flask_handler
from genkit.plugins.google_genai import GoogleAI
from genkit.plugins.google_genai import googleai_name

load_dotenv() # loads .env into shell
GEMINI_API_KEY = os.environ.get('GEMINI_APP_KEY', "") # retrieve it

ai = Genkit(
    plugins=[GoogleAI(api_key=GEMINI_API_KEY)],
    model=googleai_name('gemini-2.5-flash'),
)

app = Flask(__name__)

@app.post('/joke')
@genkit_flask_handler(ai)
@ai.flow()
async def joke(name: str, ctx):
    return await ai.generate(
        on_chunk=ctx.send_chunk,
        prompt=f'tell a medium sized joke about {name}',
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))