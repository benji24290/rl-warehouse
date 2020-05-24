class Logger():
    def __init__(self):
        self.show_debug = False
        self.show_info = False
        self.show_warn = False
        self.show_error = True
        self.show_type = True
        self.filter = ""  # "handle_arrival:"

    def info(self, *arg):
        if self.show_info and self._show(arg[0]):
            if self.show_type:
                print("INFO:", *arg)
            else:
                print(*arg)

    def debug(self, *arg):
        if self.show_debug and self._show(arg[0]):
            if self.show_type:
                print("DEBUG:", *arg)
            else:
                print(*arg)

    def warn(self, *arg):
        if self.show_warn and self._show(arg[0]):
            if self.show_type:
                print("WARN:", *arg)
            else:
                print(*arg)

    def error(self, *arg):
        if self.show_error and self._show(arg[0]):
            if self.show_type:
                print("ERROR:", *arg)
            else:
                print(*arg)

    def _show(self, prefix):
        if(self.filter == "" or prefix == self.filter):
            return True
        return False


log = Logger()
