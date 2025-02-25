import rich

_log_styles = {
    "ActiveSplat": "bold green",
    "GUI": "bold magenta",
    "Debug": "bold yellow",
    "War": "bold yellow",
    "Eval": "bold red",
}


def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="ActiveSplat"):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)