# todo 1. import your solution from folder submissions with a unique id
from submissions.sergeys_solution_example import solve_sequencial as sergeys_solve_sequential

if __name__ == "__main__":
    # todo 2. add your solver to dictionary
    solver_dict = {
        'sergeys_test':{
            'solver': sergeys_solve_sequential,
            'prev_solution': None}
    }
