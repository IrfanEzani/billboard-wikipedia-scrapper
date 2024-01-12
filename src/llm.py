import box             # Used for easy dictionary access
import yaml            # Used for parsing YAML files
from dotenv import load_dotenv  # Used for loading environment variables from .env file
from langchain_openai import ChatOpenAI  # Importing the ChatOpenAI class

# Open and read the configuration file
with open("config.yaml", "r", encoding="utf8") as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))  # Parsing the YAML file into a Box object for easy access

load_dotenv(dotenv_path=cfg.ENVDIR, verbose=True)  # ENVDIR is expected to be defined in config.yaml

# instantiate LLM (ChatOpenAI)
llm = ChatOpenAI(
    model=cfg.MODEL_NAME,               # Model name from the config file
    temperature=cfg.TEMPERATURE,        # Temperature setting from the config file
    model_kwargs={"seed": cfg.SEED}     # Additional model parameters, like seed, from the config file
)
