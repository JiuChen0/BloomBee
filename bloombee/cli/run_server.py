from petals.cli.run_server import main as petals_run_server
from bloombee.core.base_runner import run_service


def run_server(*args):
    """Run Server service"""
    run_service(petals_run_server, "Server", *args)


if __name__ == "__main__":
    run_server()
