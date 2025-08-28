from ManureCouplingStrategies.core import *
from ManureCouplingStrategies.utils import *
import os


def main(project_root=None):
    if project_root is None:
        try:
            from dotenv import load_dotenv, find_dotenv

            _ = load_dotenv(find_dotenv())
            project_root = os.getenv("PROJECT_ROOT")
        except ImportError:
            assert "dotenv not found, "
    print("Hello from manurecouplingstrategies!")


def manure_transport_all():

    print("Manure transport function called!")


if __name__ == "__main__":

    main()
