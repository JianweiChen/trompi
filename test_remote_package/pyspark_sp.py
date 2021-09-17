from trompi import *
Trompi.import_("forest.py", import_member=True)

print(colored("hoho", 'red'))
def goo():
    print(type(inspect.stack()[0][1].f_globals))