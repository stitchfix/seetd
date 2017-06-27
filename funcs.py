"""

Created by: 6/1/17
On:jesseclark

"""

import pandas as pd
import numpy as np
from itertools import combinations
import argparse
import logging
import copy

logging.info('Starting logger for...')
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def invert_dict(dict_in, append_list=False):
    """
    Invert the key:values of a dict
    :param dict_in: dict to invert
    :param append_list: append to a list? (for non-uniqueness)
    :return: inverted dict
    """
    if not append_list:
        return {val:key for key,val in dict_in.items()}
    else:
        dict_out = {val:[] for key,val in dict_in.items()}
        for key, val in dict_in.items():
            dict_out[val].append(key)
        return dict_out


def load_data(fname='SeatTest_New.xlsx'):
    """
    Load the xlsx using pandas.
    :param fname: string location of the file to load
    :return: pandas object
    """
    return pd.ExcelFile(fname)


def get_names_teams_cur_seats(file_in, names='names'):
    """
    Process the xlsx sheet, extracting the names and seats
    :param file_in: pandas object for the file, use load_data(fname)
    :param names: the name of the tab that contains the names
    :return: list of names, names:seats dict, names:teams dict
    """

    # load the names portion of the sheet
    names_df = file_in.parse(names)

    # rename if nec
    if 'Full Name' in names_df.columns:
        names_df = names_df.rename(columns={'Full Name':'Names'})

    if 'Seat' in names_df.columns:
        names_df = names_df.rename(columns={'Seat':'current seat'})

    if 'Team' in names_df.columns:
        names_df = names_df.rename(columns={'Team':'Teams'})

    # we sort here to set an order for constructing the X and dij
    names = sorted(names_df.Names.values.tolist())

    # make the init names_Seats_dict
    names_seats_dict = {row[1]['Names']: row[1]['current seat'] for row in
                        names_df[['Names', 'current seat']].reset_index(drop=True).iterrows()}

    # get the names-teams dict
    names_teams_dict = {row[1]['Names']: str(row[1]['Teams']) for row in
                        names_df[['Names', 'Teams']].reset_index(drop=True).iterrows()}

    return names, names_seats_dict, names_teams_dict


def create_seating_graph(seats_arr, excludes=('nan',0), inc_self=True):
    """
    Create the graph of seats from the layout.
    :param seats_arr: np array of physical seating arrangement
    :param excludes: ignore entries in seats_arr that take these values (i.e. use 0 or nan for aisles)
    :param inc_self: include the seat number as a neighbiour to itself?
    :return: seats graph as a dict
    """
    # which chairs are neighbours
    ni, nj = seats_arr.shape
    seats_graph = {}
    # loop through each seat
    for indi in range(ni):
        for indj in range(nj):

            # get current seat
            seat = seats_arr[indi, indj]

            if seat not in excludes:
                # now get the neighbours of the seat
                # here we consider the diagonals to be a neighbour
                ii = np.array([-1, 0, 1]) + indi
                jj = np.array([-1, 0, 1]) + indj
                # keep the indices within the bounds
                ii = ii[(ii >= 0) & (ii < ni)]
                jj = jj[(jj >= 0) & (jj < nj)]

                # loop through the indices
                inds = [(i, j) for i in ii for j in jj]
                neighbours = [seats_arr[ind] for ind in inds if seats_arr[ind] not in [seat]+list(excludes)]

                if inc_self:
                    neighbours.append(seat)

                seats_graph[seat] = neighbours

    return seats_graph


def get_seat_locations(file_in, seats='seat_map', more_connected=True):
    """
    Process the seat locations portion of the xlsx sheet
    :param file_in: pandas object for the file, use load_data(fname)
    :param seats: name of the tab that contains the sheets
    :param more_connected: bool, remove aisles from seats when constructing connection graph?
    :return: list of seats, dict of seat locations (tuples), graph of seat connections,
            dict of seat-seat distances, numpy array of seat map
    """

    # get seat locations from map and also all available seats (not just those occupied)
    seat_map_df = file_in.parse(seats, header=None)

    seats_arr = np.nan_to_num(np.array(seat_map_df, dtype=float)).astype(int)

    seats = sorted(list(seats_arr[np.where(seats_arr.astype(float) != 0)]))

    # a dict of the seat number and location
    seat_locations = {seat: (np.where(seats_arr == seat)[0][0], np.where(seats_arr == seat)[1][0]) for seat in seats}

    # we can make the seats have more neighbours by removing the aisles
    if more_connected:
        _seats_arr = seats_arr[np.where(seats_arr.sum(1) != 0), :].squeeze()
        _seats_arr = _seats_arr[:, np.where(_seats_arr.sum(0) != 0)].squeeze()

    else:
        _seats_arr = seats_arr

    seats_graph = create_seating_graph(_seats_arr, inc_self=False)

    # we want the distance from each seat to every other seat
    seat_distances = {}
    for seat1 in seats:
        distances = {}
        for seat2 in seats:
            p1 = np.array(seat_locations[seat1])
            p2 = np.array(seat_locations[seat2])
            distances[seat2] = abs(p1 - p2).sum()
        seat_distances[seat1] = distances

    return seats, seat_locations, seats_graph, seat_distances, seats_arr


def get_person_person_distance(names_seats_dict, seat_distances, names):
    """
    get person to person distances
    used for getting the cost of how far people have moved from each other
    can get names_seats_dict = X_to_names_seats_dict(X, names, seats)
    :param names_seats_dict: names:seats dict
    :param seat_distances: dict of seat-seat distances, indexed by seat name
    :param names: list of names
    :return: numpy array of person-person distances
    """

    pij = np.zeros((len(names), len(names)))

    # loop through people and seats
    # use the sorted names list, this dictates the ordering for the matrix
    for ind1, name1 in enumerate(names):

        # get seat
        seat1 = names_seats_dict[name1]

        # get distances to all other people
        for ind2, name2 in enumerate(names):
            seat2 = names_seats_dict[name2]

            pij[ind1, ind2] = seat_distances[seat1][seat2]

    return pij


def calc_dij(names, seats, seat_distances, names_seats_dict):
    """
    calc the person-seat distances.
    :param names: list of names
    :param seats: list of seats
    :param seat_distances: dict of seat-seat distances
    :param names_seats_dict: names:seats dict of current arrangement
    :return: numpy array of person-seat distances
    """
    # calc the dij matrix - ordered! 0 important
    dij = np.zeros((len(names), len(seats)))

    for ind1, name in enumerate(names):
        # get cur seat for name
        cur_seat = names_seats_dict[name]
        # get the distance to all other seats
        dists = [seat_distances[cur_seat][ind] for ind in seats]
        dij[ind1, :] = dists

    return dij


def names_seats_dict_to_X(names_seats_dict, names=None, seats=None):
    """
    Convert the dictionary of names:seats into allocation matrix Xij
    :param names_seats_dict: names:seats dict
    :param names: list of names, ordering dictates ordering of X, if none provided, defaults to sorted keys
    :param seats: list of seats
    :return: numpy array allocating person i to seat j Xij
    """

    if names is None:
        names = sorted(names_seats_dict.keys())
    if seats is None:
        seats = sorted(names_seats_dict.values())

    X = np.zeros((len(names), len(seats)))

    for ind1, name in enumerate(names):
        # get the index of the name in seat
        ind2 = seats.index(names_seats_dict[name])
        X[ind1, ind2] = 1

    return X


def X_to_names_seats_dict(X, names, seats):
    """
    Inverse operation of names_seats_dict_to_X, create a names:seats dict from numpy allocation array Xij
    :param X: numpy array allocating person i to seat j Xij
    :param names: list of names, required for keys
    :param seats: list of seats
    :return: names:seats dict for current allocation
    """

    # use normal dict so we still get key errors
    names_seats_dict = {}

    for ind1, name in enumerate(names):
        cur_seat_ind = np.where(X[ind1, :] == 1)[0][0]
        cur_seat = seats[cur_seat_ind]

        names_seats_dict[name] = cur_seat

    return names_seats_dict


def cost_distance(X, normalize=True, eps=.10, pen_same_seat=True):
    """
    Cost function for maximising distance from current position
    :param X: Data object
    :param normalize: normalize by the number of people?
    :param eps: project onto box of height eps (avoid / 0)
    :param pen_same_seat: bool, only penalize same seat (rather than adding to all)
    :return: scalar cost of current config
    """

    Xdij = X.X*X.dij

    # add additional penalty for same seat
    def add_eps(Xdij):
        Xdij[np.where(Xdij == 0)] = eps
        return Xdij

    # project onto box
    if pen_same_seat:
        Xdij = add_eps(Xdij)
        eps = 0

    if normalize:
        return 1./(eps+(Xdij).sum()/X.X.sum())
    return 1./(eps+(Xdij).sum())


def cost_same_team_by_distance(X, normalize=True, team_normalize=True, p=1.):
    """
    cost of current config pertaining to team-member by distance
    :param X: Data object of current config
    :param normalize: normalize by number of teams
    :param team_normalize: normalize by number of people in a team
    :param p: power scaling of distance term (u in blog post)
    :return: scalar of the cost for this config
    """

    allocated_seats = X.names_seats_dict

    team_names = X.team_names

    cost_team = 0

    for team in team_names:

        cost_subteam = 0
        # now get all combinations of pairs
        team_combs = list(combinations(X.teams_names_dict[team], 2))
        for name1,name2 in team_combs:

            # get occupied seats for team members
            seat1 = allocated_seats[name1]
            seat2 = allocated_seats[name2]

            dist = (X.seat_distances[seat1][seat2])**p

            cost_subteam += dist

        # normalize by team size
        cc = 1.0
        if team_normalize:
            cc = 1.0*len(team_combs)

        if cc != 0:
            inc_cost = X.team_weights[team] / (cost_subteam / cc)
        else:
            inc_cost = 0

        cost_team += inc_cost


    fact = 1.0
    if normalize:
        fact = 1.0 / len(team_names)

    return cost_team * fact


def cost_previous_neighbour_by_distance(X, normalize=True, p=1.0, eps=.5):
    """
    cost of current config for neighbours by distance
    :param X: Data object
    :param normalize: take average or sum?
    :param p: power scaling
    :param eps: softening param
    :return: cost scaler
    """
    pij_current = X.pij
    pij_orig = X.pij_orig

    if normalize:
        return 1. / (eps + (np.abs(pij_current - pij_orig) ** p).mean())
    return 1. / (eps + (np.abs(pij_current - pij_orig) ** p).sum())


def generate_random_assignment(names, seats):
    # generate a new random arrangement by shuffling seats
    return {name:new_seat for name,new_seat in zip(names,list(np.random.choice(seats, size=len(names), replace=False)))}


def get_cost_weights(X, n_trials=1000):
    """
    Get the relative weights for the cost terms based on their values for random arrangements.
    :param X: Data object
    :param n_trials: the number of random configurations to generate for scaling calculation.
    :return: list of cost weights
    """

    names = X.names
    seats = X.seats

    c1,c2,c3 = [],[],[]

    rand_X = X.copy()

    # def get_average_cost_random(ntrials=100, seats, names):
    for ind in range(n_trials):
        # make a new random arrangement
        new_names_seats_dict = generate_random_assignment(names, seats)
        # get the new X
        _rand_X = names_seats_dict_to_X(new_names_seats_dict, names, seats)

        # copy and update to random
        rand_X.update(_rand_X)

        # max this - hence -ve sign
        _c1 = cost_distance(rand_X)

        # min this
        _c2 = cost_same_team_by_distance(rand_X)

        # max this, hence -ve sign
        # pij_current = get_person_person_distance(new_names_seats_dict, seat_distances, names)
        _c3 = cost_previous_neighbour_by_distance(rand_X)

        c1.append(_c1)
        c2.append(_c2)
        c3.append(_c3)

    lambda_1 = np.mean(c1)
    lambda_2 = np.mean(c2)
    lambda_3 = np.mean(c3)

    lambdas = [lambda_1, lambda_2, lambda_3]
    tot_lambda = sum(abs(lam) for lam in lambdas)

    cost_weights = [1. / (lam / tot_lambda) for lam in lambdas]
    return tuple([cost / sum(cost_weights) for cost in cost_weights])


def cost_total(X, cost_weights=(1.0, 1.0, 1.0)):
    """
    cost of the three terms, weighted by cost_weights
    :param X: Data object
    :param cost_weights: tuple (list) of weights
    :return: scaler
    """
    return cost_weights[0] * cost_distance(X) + \
           cost_weights[1] * cost_same_team_by_distance(X) + \
           cost_weights[2] * cost_previous_neighbour_by_distance(X, normalize=True)


def fill_empty_seats(seats_names_dict, seats):
    """
    Fill empty seats with names 'empty'
    :param seats_names_dict: seats:names dict
    :param seats: list of seats that should be in the dict
    :return: seats:names dict
    """
    for seat in seats:
        if seat not in seats_names_dict:
            seats_names_dict[seat] = 'empty'

    return seats_names_dict


def swap_neighbour(X, p1=None, p2=None):
    """
    Swap two seats for a config
    :param X: Data object
    :param p1: position 1, if None, randomly chosen
    :param p2: position 2, if None, randomly chosen from neighbours of p1
    :return: Data object with updated allocation
    """

    # get current seating
    allocated_seats = X.names_seats_dict
    allocated_seats_inv = fill_empty_seats(invert_dict(allocated_seats), X.seats)

    # get which one to swap
    if p1 is None:
        p1 = np.random.choice(X.seats, 1, replace=True)[0]

    # choose randomly from its neighbours
    if p2 is None:
        p2 = np.random.choice(X.seats_graph[p1], 1, replace=True)[0]

    # swap
    allocated_seats_inv[p1], allocated_seats_inv[p2] = allocated_seats_inv[p2], allocated_seats_inv[p1]

    return names_seats_dict_to_X(invert_dict(allocated_seats_inv), names=X.names, seats=X.seats)


def local_optimize(X, nrounds=1, cost_weights=(1.,1.,1.), use_all=True, use_subset=False):
    """
    Do local optimization by switching to neighbouring states and accept any change that improves cost
    :param X: Data object
    :param nrounds: how many times to cycle
    :param cost_weights: weighting for cost terms
    :param use_all: consider everyone a neighbour?
    :param use_subset: if use_all, randomly sample a sub-portion?
    :return: Data object, old cost, new cost, intermediate states
    """

    old_cost = cost_total(X, cost_weights=cost_weights)

    orig_cost = old_cost

    configs = [X.names_seats_dict.copy()]

    # loop through number of rounds
    for ind1 in range(nrounds):

        # loop through randomly
        for ind in np.random.choice(range(len(X.seats)), len(X.seats), replace=True):
            # print(old_cost)
            # get a candidate

            if not use_all:
                neighbours_to_check = np.random.choice(X.seats_graph[X.seats[ind]],len(X.seats_graph[X.seats[ind]]), False)
            else:
                if use_subset:
                    neighbours_to_check = [st for st in X.seats if np.random.rand() > .5]
                else:
                    neighbours_to_check = X.seats

            for ind_n in neighbours_to_check:

                X_new = X.copy()
                X_new.update(swap_neighbour(X_new, p1=X.seats[ind], p2=ind_n))

                new_cost = cost_total(X_new, cost_weights=cost_weights)

                # accept if better
                if new_cost < old_cost:
                    old_cost = new_cost
                    configs.append(X.names_seats_dict)
                    X = X_new.copy()

    return X, orig_cost, old_cost, configs


def current_temperature(iter_no, s_amp, s_cent, s_width, trig=False):
    # temp for annealing schedule
    if not trig:
        return float(s_amp) / (1.0 + np.exp(1.0*(iter_no - s_cent) / float(s_width)))
    else:
        return s_amp*(np.sin((iter_no+s_cent)*2*np.pi/float(s_width))+1.0)


def accept_new_candidate(old_cost, new_cost, T, verbose=True):
    """
    Accept a new candidate?
    :param old_cost: old cost value
    :param new_cost: new cost value
    :param T: temp param
    :param verbose: print stuff
    :return: bool for acceptance
    """

    if new_cost <= old_cost:
        return True
    else:
        delta = new_cost - old_cost

        prob_accept = np.exp(-delta / float(T))

        if verbose:
            LOGGER.info("T = {}, p = {}".format(T, prob_accept))

        if prob_accept > np.random.uniform(0, 1):
            return True  # ,prob_accept
        else:
            return False  # ,prob_accept


def swap_pixels(X, p1=None, p2=None):
    """
    Swap two allocations
    :param X: numpy array for allocation
    :param p1: posn 1 to swap
    :param p2: posn 2 to swap
    :return: numpy array of updated allocation
    """

    if p1 is None or p2 is None:
        # get two people
        p1, p2 = np.random.choice(range(X.shape[0]), 2, replace=True)

    # get location of seats
    s1, s2 = np.argmax(X[p1, :]), np.argmax(X[p2, :])

    # set both to 0
    X[p1, :] = 0
    X[p2, :] = 0

    # switch seats
    X[p1, s2] = 1
    X[p2, s1] = 1

    return X


def change_X(X):
    """
    Update the data object by swapping two positions
    :param X: Data object
    :return: updated Data object
    """
    _X = swap_pixels(X.X)

    X.update(_X)

    return X


class Data:
    """
    Class to put all the stuff related to the data. Bit janky but it will do.
    """

    def __init__(self, fname, fdir):

        # name and dir of file
        self.fname = fname
        self.fdir = fdir

        # name of the tabs in the sheets
        self.names_tab = None
        self.seats_tab = None

        # placeholders
        self.names = None
        self.names_seats_dict = None
        self.seats_names_dict = None
        self.names_teams_dict = None
        self.teams_names_dict = None
        self.team_names = None

        self.names_seats_dict_orig = None

        self.seats = None
        self.seat_locations = None
        self.seats_graph = None
        self.seat_distances = None
        self.seats_arr = None

        self.dij = None
        self.pij = None
        self.pij_orig = None
        self.X = None
        self.X_orig = None
        self.X_prev = None

        self.team_weights = None

        # load the data
        self.xl = load_data(fname)

    def pre_process(self):
        # load and process everything

        self.get_names_teams_cur_seats()
        self.get_seat_locations()
        self.calc_dij()
        self.calc_pij()
        self.calc_X()

    def update(self, X):
        # update the Data based on a new arrangement

        # store prev arrangement
        self.X_prev = copy.deepcopy(X)

        # reset to current
        self.X = X

        # update dict of allocations and its inverse
        self.names_seats_dict = X_to_names_seats_dict(X, self.names, self.seats)
        self.seats_names_dict = invert_dict(self.names_seats_dict)

        # update person-person matrix
        self.calc_pij()

    def revert(self):
        # revert to previous configuration
        self.update(self.X_prev)

    def update_team_weights(self, team_weights=None):
        # add/update team weights
        if team_weights is None:
            self.team_weights = {nm: 1.0 for ind, nm in enumerate(self.team_names)}
        else:
            self.team_weights = team_weights

    def get_names_teams_cur_seats(self, names_tab='names'):

        self.names_tab = names_tab

        self.names, self.names_seats_dict_orig, \
        self.names_teams_dict = get_names_teams_cur_seats(self.xl, names=names_tab)

        self.names_seats_dict = copy.deepcopy(self.names_seats_dict_orig)

        # invert the dict
        self.seats_names_dict = invert_dict(self.names_seats_dict)
        self.teams_names_dict = invert_dict(self.names_teams_dict, append_list=True)

        self.team_names = list(set(self.teams_names_dict.keys()))

    def get_seat_locations(self, seats_tab='seat_map'):

        self.seats_tab = seats_tab

        self.seats, self.seat_locations, self.seats_graph, \
        self.seat_distances, self.seats_arr = get_seat_locations(self.xl, seats=seats_tab)

    def calc_dij(self):
        self.dij = calc_dij(self.names, self.seats, self.seat_distances, self.names_seats_dict)

    def calc_pij(self):
        if self.pij_orig is None:
            self.pij_orig = get_person_person_distance(self.names_seats_dict_orig, self.seat_distances, self.names)

        self.pij = get_person_person_distance(self.names_seats_dict, self.seat_distances, self.names)

    def calc_X(self):
        if self.X_orig is None:
            self.X_orig = names_seats_dict_to_X(self.names_seats_dict_orig, names=self.names, seats=self.seats)
        self.X = names_seats_dict_to_X(self.names_seats_dict, names=self.names, seats=self.seats)

    def copy(self):
        return copy.deepcopy(self)


def teams_to_seat_arr(teams, seats_arr, allocated_seats):

    if isinstance(teams.values()[0], int):
        # plot the team dist
        teams_seats_arr = np.zeros(seats_arr.shape)
    else:
        teams_seats_arr = np.chararray(seats_arr.shape)

    for person, seat in allocated_seats.iteritems():
        # get location for the seat
        y, x = np.where(seats_arr == seat)
        # now get the team for the
        team = teams[person]
        teams_seats_arr[y, x] = team

    return teams_seats_arr


def names_to_seat_arr(teams, seats_arr, allocated_seats, numeric=False, names=None):
    # plot the team dist
    ny,nx =  seats_arr.shape

    if not numeric:
        names_seats_arr = np.array([['  ']*nx]*ny, dtype=object)
    else:
        names_seats_arr = np.zeros((ny, nx))

    for person, seat in allocated_seats.iteritems():
        # get location for the seat
        #print(person)
        y, x = np.where(seats_arr == seat)
        # now get the team for the
        if not numeric:
            names_seats_arr[y, x] = person
        else:
            names_seats_arr[y, x] = names.index(person)

    return names_seats_arr


