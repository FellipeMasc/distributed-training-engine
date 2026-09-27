import time

def timming(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        print(f"it took {end - start} seconds to run")
        return value
    return wrapper

@timming
def add(a, b):
    return a + b

value = add(1,2)
print(value)