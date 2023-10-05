import psycopg2


class Database:
    def __init__(self, host="localhost", database="smart_cam", user="postgres", password="abdu3421"):
        self.host = host
        self.database = database
        self.user = user
        self.password = password

    @property
    def connection(self):
        return psycopg2.connect(host=self.host, database=self.database, user=self.user, password=self.password)

    def execute(self, sql: str, parameters: tuple = None, fetchone=False, fetchall=False, commit=False):
        if not parameters:
            parameters = ()
        connection = self.connection
        cursor = connection.cursor()
        data = None
        cursor.execute(sql, parameters)

        if commit:
            connection.commit()
        if fetchall:
            data = cursor.fetchall()
        if fetchone:
            data = cursor.fetchone()
        connection.close()
        return data

    def create_table_client(self):
        sql = """
        CREATE TABLE IF NOT EXISTS client (
            name VARCHAR(255),
            is_client BOOLEAN,
            created_time TIMESTAMP,
            last_time TIMESTAMP,
            last_enter_time TIMESTAMP,
            last_leave_time TIMESTAMP,
            enter_count INTEGER,
            leave_count INTEGER,
            stay_time INTEGER,
            image VARCHAR(255),
            last_image VARCHAR(255)
            )
"""
        self.execute(sql, commit=True)

    @staticmethod
    def format_args(sql, parameters: dict):
        sql += " AND ".join([
            f"{item} = %s" for item in parameters
        ])
        return sql, tuple(parameters.values())

    def add_person(self, name: str, is_client: bool, created_time, last_time, last_enter_time, last_leave_time,
                   enter_count: int, leave_count: int, stay_time: int, image: str, last_image: str):

        sql = """
        INSERT INTO client(name, is_client, created_time, last_time, last_enter_time, 
        last_leave_time, enter_count, leave_count, stay_time, image, last_image) 
        VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        self.execute(sql, parameters=(name, is_client, created_time, last_time, last_enter_time, last_leave_time,
                                      enter_count, leave_count, stay_time, image, last_image), commit=True)

    def select_person(self, name):
        # SQL_EXAMPLE = "SELECT * FROM Users where id=1 AND Name='John'"
        sql = "SELECT * FROM client WHERE name = %s"
        # sql, parameters = self.format_args(sql, kwargs)

        return self.execute(sql, parameters=(name,), fetchone=True)

    def select_param(self, param: str, name: str):
        # SQL_EXAMPLE = "SELECT * FROM Users where id=1 AND Name='John'"
        sql = f"SELECT {param} FROM client WHERE name= %s"

        return self.execute(sql, parameters=(name, ), fetchone=True)

    def count_people(self):
        return self.execute("SELECT COUNT(*) FROM client;", fetchone=True)

    def update_person(self, name: str, **kwargs):
        # Initialize the SQL query and parameter list
        sql = "UPDATE client SET "
        parameters = []

        # Iterate through the key-value pairs in the kwargs dictionary
        for key, value in kwargs.items():
            # Check if the value is an integer or a numeric increment
            if isinstance(value, int) or isinstance(value, float) or (isinstance(value, str) and value.isdigit()):
                sql += f"{key} = {key} + %s, "
                parameters.append(value)
            else:
                sql += f"{key} = %s, "
                parameters.append(value)

        # Remove the trailing comma and space
        sql = sql.rstrip(', ')

        # Add the WHERE clause
        sql += " WHERE name = %s"
        parameters.append(name)

        # Execute the SQL query
        return self.execute(sql, parameters=tuple(parameters), commit=True)

    def delete_users(self):
        self.execute("DELETE FROM client WHERE TRUE", commit=True)
