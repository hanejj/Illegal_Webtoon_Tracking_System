__version__ = '0.4.4'

class Notes:
    def __init__(self):
        self.show = True

    def print_notes(self):
        if self.show:
            print(f'''
loading...
            ''')
            self.close()

    def close(self):
        self.show = False

bw_notes = Notes()
