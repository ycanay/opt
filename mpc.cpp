#include"mpc.h"
/**
 * @brief TODO
 * 
 * @param params 
 */
void MPController::update_optimizer(int params[]){
    delete(this->optimizer);
    this->optimizer = new Optimizer();
    current_step = 0;
}

int* MPController::step_optimizer(){
    int *return_val = optimizer->get_solution_copy();
    delete(optimizer);
    update_optimizer(return_val);
    last_state = return_val;
    current_step++;
    return return_val;
}

int** MPController::get_mpc_result(){
    int** result = new int*[max_steps];
    for(int iterator = 0; iterator < max_steps; iterator++){
        result[iterator] = step_optimizer();
    }
    return result;
}

void MPController::update_max_step(int number_of_steps){
    max_steps = number_of_steps;
}