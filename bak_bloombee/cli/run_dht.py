from petals.cli.run_dht import main as petals_run_dht
from bloombee.core.base_runner import run_service


def run_dht(*args):
    """Run DHT service"""
    run_service(petals_run_dht, "DHT", *args)


if __name__ == "__main__":
    run_dht()
