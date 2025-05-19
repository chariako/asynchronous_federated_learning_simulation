import numpy as np

def generate_global_clock(num_clients, rates, T_train, Delta, mode):
    # Generate local clocks for each client
    local_clocks = []
    for client_id in range(num_clients):
        samples_i = int(np.ceil(rates[client_id] * T_train))
        local_times = np.cumsum(np.random.exponential(
            scale=1/rates[client_id], size=samples_i))
        local_clocks.extend([(time, [client_id]) for time in local_times])

    # Sort the global clock by time stamps
    global_clock = sorted(local_clocks, key=lambda x: x[0])
    
    # filter global_clock
    for t in range(len(global_clock)-1,-1,-1):
        if global_clock[t][0] > T_train:
            global_clock.pop(t)
        else:
            break

    if mode == 'async':

        return global_clock

    elif mode == 'sync':

        new_clock = []
        T = len(global_clock)
        sampled = set(np.random.choice(
            range(num_clients), Delta, replace=False))
        found = set()
        t = 0

        while found != sampled and t < T:
            time_stamp, [client_id] = global_clock[t]
            if client_id in sampled - found:
                found.add(client_id)

            t += 1

            if found == sampled:
                new_clock.append((time_stamp, list(sampled)))
                sampled = set(np.random.choice(
                    range(num_clients), Delta, replace=False))
                found = set()

        return new_clock
