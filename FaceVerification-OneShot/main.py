from utils import *

distance, door_open_flag = verify("images\Kha.jpg", "KHA", database, model)


print("(", distance, ",", door_open_flag, ")")

