import matplotlib.pyplot as plt

def _get_axis(ax = None):
    if ax is None:
        fig, ax = plt.subplots()
        return ax
    else:
        return ax

def _adjust_kwargs(var_name: str, default_kwargs: dict = {}, user_kwargs: dict = {}):
    '''
    Merges default kwargs with user input kwargs
    This will override any existing default kwarg with user kwarg and add user kwarg for any missing default kwarg
    '''
    # Search through all user kwargs (this ensures any kwarg not in defaultKwargs will be added)
    for p in user_kwargs:
        # If user specifies a dict for the kwarg (based off varName, then get the variable specific kwarg)
        if isinstance(user_kwargs[p], dict):
            # If the dict doesnt have the varName, then we don't add/override default kwargs
            if var_name in user_kwargs[p]:
                default_kwargs[p] = user_kwargs[p][var_name]
        else:
            default_kwargs[p] = user_kwargs[p]
    return default_kwargs