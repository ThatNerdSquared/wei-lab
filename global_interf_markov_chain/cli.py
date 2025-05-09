from recomb_markov_chain import RecombMarkovChain

def main():
    rmc = RecombMarkovChain(n_states=4)
    rmc.reset_to_poisson(0.8)
    print(rmc)


if __name__ == "__main__":
    main()
