import sys
import subprocess
from typing import Callable, Dict, Any, List

from rq import Retry, Queue


def connect_to_mysql(usr: str, psw: str, host: str) -> None:
    """function to connect to mysql database

    Args:
        usr:
            the username for mySQL server
        psw:
            the password for the user
        host:
            IP of the host. Port is 3306
    """
    import mysql.connector

    mydb = mysql.connector.connect(host=host, user=usr, password=psw)

    print(mydb)
    mycursor = mydb.cursor()

    mycursor.execute("CREATE DATABASE IF NOT EXISTS example")

    mycursor.execute("SHOW DATABASES")

    for x in mycursor:  # type: ignore
        print(x)


# testing the working of RQ
def test_fnc(text: str) -> int:
    """testing function

    Args:
        text:
            a string

    Returns:
        int:
            the length of the input string
    """
    return len(text)


def run_hpo_in_parallel(q: Queue, fnc: Callable, args: List[Any], number: int) -> None:
    """this function allows to automatically enqueue the jobs

    Args:
        q:
            the redis queue
        fnc:
            the function containing the computations
        args:
            the arguments of the function ``fnc``
        number:
            the number of times you wanto to enqueue the job
    """
    for _ in range(number):
        q.enqueue(fnc, args=args, retry=Retry(max=3))
