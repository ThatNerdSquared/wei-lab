from recomb_markov_chain import RecombMarkovChain


def main():
    rmc = RecombMarkovChain(n_states=4)
    rmc.reset_to_poisson(0.5)
    print(rmc)
    print(rmc.get_p_cos(0))
    print(rmc.get_p_cos(1))
    print(rmc.get_p_cos(2))
    print(rmc.get_p_cos(3))
    rmc.perturb_transition_probability(1, 2, -0.2)
    print(rmc)
    print(rmc.get_p_cos(0))
    print(rmc.get_p_cos(1))
    print(rmc.get_p_cos(2))
    print(rmc.get_p_cos(3))
    # print(rmc.get_p_cos(4))
    # print(rmc.get_p_cos(5))
    # print(rmc.get_p_cos(6))
    # print(rmc.get_p_cos(7))
    # print(rmc.get_p_cos(8))


if __name__ == '__main__':
    main()
