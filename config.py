import yaml

config = None


def load_config():
    global config

    with open("./config.yml", "r") as file:
        config = yaml.safe_load(file)
