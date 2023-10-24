# Description: utilities used to handle api rate limit.

import random
import time
from typing import Tuple, List, Optional

import openai

CHARACTER_LIMIT_PER_MINUTE = 9000

character_count = 0
last_reset_time = time.time()


def reset_character_count():
    global character_count
    global last_reset_time
    character_count = 0
    last_reset_time = time.time()


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError,),
):
    """
    Retry a function with exponential backoff for OpenAI API rate limit.
    """

    def wrapper(*args, **kwargs):
        global character_count
        global last_reset_time

        num_retries = 0
        delay = initial_delay

        while True:
            try:
                current_time = time.time()
                if (current_time - last_reset_time) >= 60:
                    reset_character_count()

                message_to_send = kwargs.get("message", "")
                character_count += len(message_to_send)

                if character_count > CHARACTER_LIMIT_PER_MINUTE:
                    time_to_wait = 60 - (current_time - last_reset_time)
                    time.sleep(time_to_wait)
                    reset_character_count()

                response = func(*args, **kwargs)

                return response

            except errors as e:
                num_retries += 1

                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                delay *= exponential_base * (1 + jitter * random.random())

                time.sleep(delay)

            except Exception as e:
                raise e

    return wrapper
