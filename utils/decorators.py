def conditional_decorator(dec, condition, error_msg=""):
    def decorator(func):
        if not condition:
            print(error_msg)
        else:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)
    return decorator