# utils/logging_utils.py
import logging
import logstash

# Define the Logstash server and port
LOGSTASH_HOST = 'logstash'  # Docker container name
LOGSTASH_PORT = 5003

def log_deployment_event(event_message, log_level='info'):
    """
    Logs deployment events such as success, failure, or other status updates.

    Parameters:
    - event_message: A string message describing the deployment event.
    - log_level: The level of logging ('info', 'warning', 'error'). Defaults to 'info'.
    """
    # Initialize the logger (this can also be configured externally via logging config)
    logger = logging.getLogger("deployment_logger")
    
    # Add Logstash handler to send logs to Logstash server
    logstash_handler = logstash.TCPLogstashHandler(LOGSTASH_HOST, LOGSTASH_PORT, version=1)
    logger.addHandler(logstash_handler)
    
    # Set the log level based on input
    if log_level == 'info':
        logger.info(event_message)
    elif log_level == 'warning':
        logger.warning(event_message)
    elif log_level == 'error':
        logger.error(event_message)
    else:
        logger.info(event_message)  # Default to info if no proper log_level is given
    
def log_monitoring_event(event_message, log_level='info'):
    """
    Logs monitoring events such as resource usage, performance metrics, etc.

    Parameters:
    - event_message: A string message describing the monitoring event.
    - log_level: The level of logging ('info', 'warning', 'error'). Defaults to 'info'.
    """
    # Initialize the logger (this can also be configured externally via logging config)
    logger = logging.getLogger("monitoring_logger")
    
    # Add Logstash handler to send logs to Logstash server
    logstash_handler = logstash.TCPLogstashHandler(LOGSTASH_HOST, LOGSTASH_PORT, version=1)
    logger.addHandler(logstash_handler)
    
    # Set the log level based on input
    if log_level == 'info':
        logger.info(event_message)
    elif log_level == 'warning':
        logger.warning(event_message)
    elif log_level == 'error':
        logger.error(event_message)
    else:
        logger.info(event_message)  # Default to info if no proper log_level is given