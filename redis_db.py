import redis
import numpy as np


class Memory:
    def __init__(self, host='localhost', port=6379, decode_responses=True):
        self.host = host
        self.port = port
        self.decode_responses = decode_responses
        self.connection = None  # Initialize connection as None
        self.people_names = []
        self.people_encodings = []

    def open_connection(self):
        if self.connection is None:
            self.connection = redis.Redis(
                host=self.host, port=self.port, decode_responses=self.decode_responses
            )

    def add_people(self, items: dict):
        self.open_connection()
        r = self.connection
        for i, k in items.items():
            r.hset(
                f"client:{i}",
                mapping=k,
            )

        # self.close_connection()
        return True

    def add_person(self, person: str, **kwargs):
        self.open_connection()
        r = self.connection
        for key, value in kwargs.items():
            try:
                r.hset(person, key, value)
            except:
                print(f"{key}: {value}")
        return True

    def get_field(self, name: str, field: str):
        self.open_connection()
        r = self.connection
        return r.hget(name, key=field)

    def update_person(self, person: str, **kwargs):
        self.open_connection()
        r = self.connection

        # Iterate through the key-value pairs in the kwargs dictionary
        for key, value in kwargs.items():
            # Check if the value is an integer or a numeric increment
            if ((isinstance(value, int) or isinstance(value, float) or (isinstance(value, str) and value.isdigit())) and
                    key != 'stay_time'):
                r.hincrby(name=person, key=key, amount=value)
            else:
                r.hset(person, key, value)
        return True

    def get_all_field(self, name: str):
        self.open_connection()
        r = self.connection
        return r.hgetall(name)

    def get_all_people(self, field1: str, field2: str):
        self.open_connection()
        r = self.connection
        hash_keys = r.keys('client:*')

        for key in hash_keys:
            field1_value = r.hget(key, field1)
            field2_value = r.hget(key, field2)
            if field1_value:
                self.people_names.append(field1_value)
            if field2_value:
                self.people_encodings.append(np.array(field2_value.strip('[]').split(), dtype=np.float64))
        return self.people_names, self.people_encodings
