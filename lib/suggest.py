import numpy as np

def calc_users_sims(pred, genres, users_prefs):
    '''
    Calculates similarity between pred_genre_vec and users preferences

    Arguments:
        pred -- genre vector of new music
        genres -- features names of the ohe of genres. As list
        users_prefs -- list of list of tuples containing music preferences
            for each user. Form of [[(genre, val),...],
                                    [(genre, val),...]
                                    ...]
    Returns:
        sims -- similarity between users preferences and genre vector of new music
            As np.array, length equal to number of users
    '''
    ohe_users_prefs = np.zeros([len(users_prefs), len(genres)])
    for i in range(len(users_prefs)):
        ohe = dict(zip(genres, np.zeros(len(genres))))
        for genre, val in users_prefs[i]:
            if genre in genres:
                ohe[genre] = val
            else:
                print(genre+' is not in ohe features vec, genres.')
        ohe_users_prefs[i] = list(ohe.values())

    sims = np.dot(pred, ohe_users_prefs.T)/np.dot(pred, pred)

    return sims

def suggest(sims, names, title, threshold=0.5):
    '''
    Prints string saying who should listen to the music
    
    Arguments:
        sims -- similarity vector from calc_users_sims
        names -- names of the users, as list of strings
        title -- title of song, string
        Optional:
            threshold -- threshold for suggesting.
                If 0, threshold low, everyone should listen
    '''
    if sum(sims>threshold) == 0:
        print('No one might like the song '+title)
        return

    for i in range(len(sims)):
        if sims[i]>threshold:
            print(names[i]+' might like the song '+title)

    return