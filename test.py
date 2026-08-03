import os

file_path = "Makefile"
mod_time = os.path.getmtime(file_path)
print(mod_time)