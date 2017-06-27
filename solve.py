"""

Created by: 6/27/17
On:jesseclark

"""

from funcs import *

def get_default_params():

    # file locations
    save_dir = ''
    save_name = 'results.csv'

    fname = 'seetd_example.xlsx'
    fdir = ''

    # algo params
    auto_calc_weights = True

    cost_weights_in = (1., 1., 1.)

    n_starts = 1
    number_of_iterations = 15
    starting_iteration = 0
    s_amp = 1.
    s_width = 10.
    s_cent = 0

    fully_connected = True

    return save_dir, save_name, fname, fdir, auto_calc_weights, cost_weights_in, n_starts, number_of_iterations, \
           starting_iteration, s_amp, s_width, s_cent, fully_connected

def parse_args(parser):

    parser.add_argument('--save_dir', help="directory to save to", default=save_dir)
    parser.add_argument('--save_name', help="saved filename", default=save_name)

    parser.add_argument('--file_dir', help="directory to load file from", default=fdir)
    parser.add_argument('--file_name', help="filename", default=fname)

    parser.add_argument('--auto_calc_weights', help="automatically calc the scaling of cost terms?", default='True')

    parser.add_argument('--cost_weights', help="weights for each of the cost terms", default=cost_weights_in, nargs='+')

    parser.add_argument('--iterations', help="number of iterations", default=number_of_iterations, type=int)

    parser.add_argument('--fully_connected', help="should everyone be considered a neighbour for local opt?", default='True')

    return parser.parse_args()


if __name__ == "__main__":

    # pre-define params for defaults - easier to debug with an IDE
    save_dir, save_name, fname, fdir, auto_calc_weights, cost_weights_in, n_starts, number_of_iterations, \
    starting_iteration, s_amp, s_width, s_cent, fully_connected = get_default_params()

    # parsed args
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    # remap to params
    save_dir = args.save_dir
    save_name = args.save_name

    fname = args.file_name
    fdir = args.file_dir

    number_of_iterations = args.iterations

    # algo params
    auto_calc_weights = args.auto_calc_weights == 'True'

    fully_connected = args.fully_connected == 'True'

    cost_weights_in = map(float, args.cost_weights)

    # summary of inputs
    LOGGER.info("{}{}".format(fdir, fname))
    LOGGER.info("{}{}".format(save_dir, save_name))
    LOGGER.info("{}".format(auto_calc_weights))
    LOGGER.info("{}".format(cost_weights_in))
    LOGGER.info("{}".format(number_of_iterations))
    LOGGER.info("{}".format(fully_connected))


    # start from here
    X = Data(fname, fdir)
    X.pre_process()
    X.update_team_weights({ind: 1 if ind != 1 else -1 for ind in X.team_names})

    # init some things
    n_changes = 0
    temps = []
    global_configs = []

    # params
    # cost_weights = (distance weight, team weight, neighbour weight)
    if auto_calc_weights:
        LOGGER.info("Auto-calculating cost weights...")
        cost_weights = get_cost_weights(X)
    else:
        LOGGER.info("Using provided cost weights...")
        cost_weights = cost_weights_in
    LOGGER.info("(distance weight, team weight, neighbour weight) = {}".format([float(str(cw)[:5]) for cw in cost_weights]))


    # start calculating
    old_cost = cost_total(X, cost_weights=cost_weights)
    best_cost = old_cost

    best_costs = [best_cost]
    current_costs = [old_cost]

    # debug
    LOGGER.info("Team names: {}".format(X.team_names))

    LOGGER.info("Initial cost [{}]".format(old_cost))

    for i in range(number_of_iterations):

        # current itno
        iter_no = starting_iteration + i
        # current temperature
        T = current_temperature(iter_no, s_amp, s_cent, s_width, trig=False)

        # change the curret one
        X_candidate = change_X(X.copy())
        cand_cost = cost_total(X_candidate, cost_weights=cost_weights)

        # aceptance criteria
        accept = accept_new_candidate(old_cost, cand_cost, T)

        if accept:
            X, orig_cost, cand_cost, configs = local_optimize(X_candidate, cost_weights=cost_weights, use_all=fully_connected)
            old_cost = cand_cost

        # store copy of the best so far
        if cand_cost <= best_cost:
            best_cost = cand_cost
            X_best = X.copy()

        best_costs.append(best_cost)
        current_costs.append(old_cost)

        LOGGER.info("[{}] Current [{}], Best [{}]".format(i, cand_cost, best_cost))

    LOGGER.info("Finished iterations...")
    LOGGER.info("Saving to {}...".format(save_dir+save_name))
    team_arr = teams_to_seat_arr(X_best.names_teams_dict, X_best.seats_arr, X_best.names_seats_dict)
    names_arr = names_to_seat_arr(X_best.names_teams_dict, X_best.seats_arr, X_best.names_seats_dict)

    # output a results dataframe
    results_df = pd.DataFrame(X_best.names_seats_dict_orig.items(), columns=['name', 'old_seat']).merge(
        pd.DataFrame(X_best.names_seats_dict.items(), columns=['name', 'new_seat']), on=['name']).merge(
        pd.DataFrame(X_best.names_teams_dict.items(), columns=['name', 'team']), on=['name'])

    results_df = results_df[['name','team','old_seat','new_seat']]
    results_df.sort_values(by='name').to_csv(save_dir+save_name)
    LOGGER.info("Done...")

    LOGGER.info("cost went from [{}] -> [{}]".format(current_costs[0], best_cost))

    print(team_arr)
    print(names_arr)