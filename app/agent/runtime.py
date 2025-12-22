"""Agent runtime initialization - singleton Runner and SessionService."""
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from .agent import root_agent

# Initialize session service (singleton)
session_service = InMemorySessionService()

# Initialize Runner with root agent (singleton)
runner = Runner(
    app_name="fashion-image-api",
    agent=root_agent,
    session_service=session_service
)
