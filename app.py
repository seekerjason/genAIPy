import os
from dotenv import load_dotenv
import json
from pydantic import BaseModel, Field
from genkit.ai import Genkit
from genkit.plugins.google_genai import GoogleAI

load_dotenv() # loads .env into shell
GEMINI_API_KEY = os.environ.get('GEMINI_APP_KEY', None) # retrieve it

ai = Genkit(
    plugins=[GoogleAI(api_key=GEMINI_API_KEY)],
    model='googleai/gemini-2.5-flash',
)

class RpgCharacter(BaseModel):
    name: str = Field(description='name of the character')
    back_story: str = Field(description='back story')
    abilities: list[str] = Field(description='list of abilities (3-4)')

@ai.flow()
async def generate_character(name: str):
    result = await ai.generate(
        prompt=f'generate an RPG character named {name}',
        output_schema=RpgCharacter,
    )
    return result.output

'''
async def main() -> None:
    reps=await generate_character('Goblorb')

    print(json.dumps(reps, indent=2))

ai.run_main(main())
'''