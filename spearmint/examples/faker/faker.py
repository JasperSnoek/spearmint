import sys
import math
import time
import random


# A fake task that sleeps and returns a random result so we can test and debug
# spearmint without wasting CPU and energy computing real functions.

def main(job_id, params):
    time.sleep(random.random() * 2)
    return random.random() * 100
