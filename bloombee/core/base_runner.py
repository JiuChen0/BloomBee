import sys
from typing import Callable
from bloombee.logger import get_logger

logger = get_logger()


def run_service(main_func: Callable, service_name: str, *args):
    """
    Base function for running services

    Args:
        main_func: The main function to call (e.g. petals_run_dht)
        service_name: Name of the service for logging
        *args: Arguments to pass to the service
    """
    if not args:
        args = sys.argv[1:]

    sys_argv = [f"bloombee_{service_name.lower()}"] + list(args)
    sys.argv = sys_argv

    logger.info(f"Starting BloomBee {service_name} service...")
    try:
        main_func()
    except Exception as e:
        logger.error(f"{service_name} service failed: {str(e)}")
        raise
    finally:
        logger.info(f"{service_name} service stopped")
