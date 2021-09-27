#include"optimization.h"

class MPController
{
private:
    Optimizer* optimizer;
    int max_steps;
    int* last_state;
    int current_step;

public:
    MPController(int number_of_steps);
    ~MPController();
    void update_optimizer(int params[]);
    int* step_optimizer();
    int** get_mpc_result();
    void update_max_step(int number_of_steps);
};

MPController::MPController(int number_of_steps)
{
    optimizer = new Optimizer(false);
    this->max_steps = number_of_steps;
}

MPController::~MPController()
{
    delete optimizer;
}

