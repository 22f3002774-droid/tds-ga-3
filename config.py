# Put your OpenAI API key here, OR (recommended) set it as an environment
# variable named OPENAI_API_KEY on your hosting platform instead of editing
# this file, so you don't commit a secret key to GitHub.
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-REPLACE-ME")
