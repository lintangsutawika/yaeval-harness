# TODO: Function to fill in results to a csv or google spreadsheet


def simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {
        k: handle_arg_string(v) for k, v in [arg.split("=") for arg in arg_list]
    }
    return args_dict


def handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg


def calculate_tokens(tokens):
    if isinstance(tokens, Tuple):
        tokens = [tokens]

    num_tokens = 0
    for _tokens in tokens:
        i_tokens, o_tokens = _tokens
        num_tokens = len(i_tokens + o_tokens)

    return num_tokens