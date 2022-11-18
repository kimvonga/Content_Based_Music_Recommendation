import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

plt.rcParams.update({'font.size': 16})

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(2*np.pi, 0, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)


                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def plot_pred(df, genres, ind, lim=5, frame='circle', color='C0'):
    '''
    Creates radar plot of song's genre vector
    
    Arguments:
        df -- dataframe containing genre_vec under column name 'pred_genre_vec'
              also contains song title under 'title' and 'artist_name'
        genres -- ohe of genre_vec to return genre names. As list
        ind -- index of song, index starting at 0
        Optional:
            lim -- limit number of genres shown
            frame -- draw radar plot as 'circle' or 'polygon'
            color -- color of radar plot
    '''
    imp_ind = np.argsort(-df['pred_genre_vec'].iloc[ind]).astype('int')[:lim]
    tags = genres[imp_ind]
    vals = df['pred_genre_vec'].iloc[ind][imp_ind]
    title = df['title'].iloc[ind]+' by '+df['artist_name'].iloc[ind]
    
    theta = radar_factory(lim, frame=frame)
    
    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)

    line = ax.plot(theta, vals, color=color)
    ax.fill(theta, vals,  alpha=0.25, color=color)
    ax.set_varlabels(tags)

    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_title(title, weight='bold', size='medium', position=(0.5,1.1),
                horizontalalignment='center', verticalalignment='center')

    plt.show()
    return

def create_user(user, lim=5, frame='circle'):
    '''
    Creates radar plot of user music preferences
    
    Arguments:
        user -- tuple of form (name, color, [(genre, val),...])
        Optional:
            lim -- Limit number of genres shown
            frame -- draw radar plot as 'circle' or 'polygon'
    '''
    name = user[0]
    color = user[1]
    tags = np.array([x[0] for x in user[2]])
    vals = np.array([x[1] for x in user[2]])
    imp_ind = np.argsort(-vals).astype('int')[:lim]
    tags = tags[imp_ind]
    vals = vals[imp_ind]
    
    theta = radar_factory(lim, frame=frame)
    
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)

    line = ax.plot(theta, vals, color=color)
    ax.fill(theta, vals,  alpha=0.25, color=color)
    ax.set_varlabels(tags)
    
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_title(name, weight='bold', size='medium', position=(0.5,1.1),
                horizontalalignment='center', verticalalignment='center')

    plt.show()
    return

def create_users(users, lim=5, frame='circle'):
    '''
    Creates radar plots of users' music preferences
    
    Arguments:
        users -- list of tuples of form [(name, color, [(genre, val),...]),
                                         (name, color, [(genre, val),...])]
        Optional:
            lim -- Limit number of genres shown
            frame -- draw radar plot as 'circle' or 'polygon'
    '''
    names = np.array([x[0] for x in users])
    colors = np.array([x[1] for x in users])
    tups = np.array([x[2] for x in users])
    tags = np.array([[y[0] for y in x] for x in tups])
    vals = np.array([[float(y[1]) for y in x] for x in tups])

    num_users = len(users)

    theta = radar_factory(lim, frame=frame)

    fig, axs = plt.subplots(figsize=(5.5*num_users, 4), nrows=1, ncols=num_users,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    for i in range(num_users):
        # imp_ind = np.argsort(-vals[i]).astype('int')[:lim]
        t = tags[i]
        v = vals[i]

        axs[i].plot(theta, v, color=colors[i])
        axs[i].fill(theta, v, facecolor=colors[i], alpha=0.25)

        axs[i].set_varlabels(t)
        axs[i].set_rgrids([0.2, 0.4, 0.6, 0.8])
        axs[i].set_title(names[i], weight='bold', size='medium', position=(0.5,1.1),
                        horizontalalignment='center', verticalalignment='center')

    plt.show()
    return