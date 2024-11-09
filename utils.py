
class ANSIColor:
    RESET = "\033[0m"

    COLORS = {
        'black': "\033[30m",
        'red': "\033[31m",
        'green': "\033[32m",
        'yellow': "\033[33m",
        'blue': "\033[34m",
        'magenta': "\033[35m",
        'cyan': "\033[36m",
        'white': "\033[37m",
    }

    BG_COLORS = {
        'black': "\033[40m",
        'red': "\033[41m",
        'green': "\033[42m",
        'yellow': "\033[43m",
        'blue': "\033[44m",
        'magenta': "\033[45m",
        'cyan': "\033[46m",
        'white': "\033[47m",
    }

    @staticmethod
    def text(text, color=None, bg_color=None):
        color_code = ANSIColor.COLORS.get(color, "")
        bg_color_code = ANSIColor.BG_COLORS.get(bg_color, "")
        return f"{color_code}{bg_color_code}{text}{ANSIColor.RESET}"
