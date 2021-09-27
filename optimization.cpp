#include<optimization.h>
#include<iostream>
#include <cassert>
#include <math.h>

#define W_GX  1.00
#define W_GY  1.00
#define W_V  1.0
#define W_ALPHA  1.0
#define W_THETA 1.0
#define W_Y 1.0
#define W_J 1.0
#define W_H 1.0

#define X_D 100.0
#define Y_D 100.0
#define V_D 10.0
#define THETA_D 0.0

#define NUMBER_OF_STATES 6

Optimizer::Optimizer(bool printiterate
) : printiterate_(printiterate)
{
    internal_state = new int[NUMBER_OF_STATES];
}

Optimizer::~Optimizer(){
    
}

bool Optimizer::get_nlp_info(
    Index&                   n,
    Index&                   m,
    Index&                   nnz_jac_g,
    Index&                   nnz_h_lag,
    IndexStyleEnum&          index_style){
    // The problem has 6 variables, x[0] through x[5]
    n = NUMBER_OF_STATES;
    
    // 4 equality kinematic constraint
    m = 4;
    
    // in this example the jacobian is dense and contains 24 nonzeros
    nnz_jac_g = 24;
    
    // the Hessian is also dense and has 36 total nonzeros, but we
    // only need the lower left corner (since it is symmetric)
    nnz_h_lag = 21;
    
    // use the C style indexing (0-based)
    index_style = TNLP::C_STYLE;
    
    return true;
}

bool Optimizer::get_bounds_info(
    Index   n,
    Number* x_l,
    Number* x_u,
    Index   m,
    Number* g_l,
    Number* g_u){
    // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
    // If desired, we could assert to make sure they are what we think they are.
    assert(n == NUMBER_OF_STATES);
    assert(m == 4);
    
    // the variables have lower bounds
    x_l[0] = - 300. ;
    x_l[1] = - 300. ;
    x_l[2] = 0. ;
    x_l[3] = - 5. ;
    x_l[4] = - 2. ;
    x_l[5] = - M_PI / 10 ;
    
    // the variables have upper bounds
    x_u[0] = +300. ;
    x_u[1] = +300. ;
    x_u[2] = M_PI * 2 ;
    x_u[3] = 30. ;
    x_u[4] = 2. ;
    x_u[5] = M_PI / 10 ;
    
    // the first constraint g0 has a equality which is x_0[0] = 0
    g_l[0] = g_u[0] = 0.;
    // the first constraint g1 has NO upper bound, here we set it to 2e19.
    // Ipopt interprets any number greater than nlp_upper_bound_inf as
    // infinity. The default value of nlp_upper_bound_inf and nlp_lower_bound_inf
    // is 1e19 and can be changed through ipopt options.
    
    // the 2nd constraint g1 has a equality which is x_0[1] = 0
    g_l[1] = g_u[1] = 0.;

    // the 3rd constraint g1 has a equality which is x_0[2] = 0
    g_l[2] = g_u[2] = 0.;

    // the 4th constraint g1 has a equality which is x_0[3] = 0
    g_l[3] = g_u[3] = 0.;

    return true;
}

bool Optimizer::get_starting_point(
      Index   n,
      bool    init_x,
      Number* x,
      bool    init_z,
      Number* z_L,
      Number* z_U,
      Index   m,
      bool    init_lambda,
      Number* lambda){
    // Here, we assume we only have starting values for x, if you code
    // your own NLP, you can provide starting values for the dual variables
    // if you wish
    assert(init_x == true);
    assert(init_z == false);
    assert(init_lambda == false);
    
    // initialize to the given starting point
    x[0] = 0.;
    x[1] = 0.;
    x[2] = 0.;
    x[3] = 0.;
    
    return true;
}
bool Optimizer::eval_f(
    Index         n,
    const Number* x,
    bool          new_x,
    Number&       obj_value){

    assert(n == NUMBER_OF_STATES);
    
    obj_value = W_GX * (X_D - x[0]) * (X_D - x[0]) + W_GY * (Y_D - x[1]) * (Y_D - x[1])+ 
                W_V * (V_D -x[3]) * (V_D -x[3]) + W_ALPHA * (x[4] * x[4]) + 
                W_Y * (x[5] * x[5]) + W_H * (THETA_D - x[2]) * (THETA_D - x[2]);
    
    return true;
}

bool Optimizer::eval_grad_f(
    Index         n,
    const Number* x,
    bool          new_x,
    Number*       grad_f){

    assert(n == NUMBER_OF_STATES);
    
    grad_f[0] = 2 * W_GX * x[0] - 2 * W_GX * X_D;
    grad_f[1] = 2 * W_GY * x[1] - 2 * W_GY * Y_D;
    grad_f[2] = 2 * W_H * x[2] - 2 * W_H * THETA_D;
    grad_f[3] = 2 * W_V * x[3] - 2 * W_V * V_D;
    grad_f[4] = 2 * W_ALPHA * x[4];
    grad_f[5] = 2 * W_V * x[3] - 2 * W_V * V_D;
    
    return true;
}

bool Optimizer::eval_g(
    Index         n,
    const Number* x,
    bool          new_x,
    Index         m,
    Number*       g){
    
    assert(n == NUMBER_OF_STATES);
    assert(m == 4);
    
    g[0] = x[0] - (x[3] * cos(x[2]) / 10);
    g[1] = x[1] - (x[3] * sin(x[2]) / 10);
    g[2] = x[2] - (x[5] / 10);
    g[3] = x[3] - (x[4] / 10);
    
    return true;
}

bool Optimizer::eval_jac_g(
    Index         n,
    const Number* x,
    bool          new_x,
    Index         m,
    Index         nele_jac,
    Index*        iRow,
    Index*        jCol,
    Number*       values){

    assert(m == 4);
    
    if( values == NULL )
    {
        // return the structure of the Jacobian
    
        // this particular Jacobian is dense
        iRow[0] = 0;
        jCol[0] = 0;

        iRow[1] = 0;
        jCol[1] = 1;
        
        iRow[2] = 0;
        jCol[2] = 2;
        
        iRow[3] = 0;
        jCol[3] = 3;
        
        iRow[4] = 0;
        jCol[4] = 4;
        
        iRow[5] = 0;
        jCol[5] = 5;
/////////////////////////////////////////
        iRow[6] = 1;
        jCol[6] = 0;
        
        iRow[7] = 1;
        jCol[7] = 1;
        
        iRow[8] = 1;
        jCol[8] = 2;
        
        iRow[9] = 1;
        jCol[9] = 3;
        
        iRow[10] = 1;
        jCol[10] = 4;
        
        iRow[11] = 1;
        jCol[11] = 5;
/////////////////////////////////////////
        iRow[12] = 2;
        jCol[12] = 0;
        
        iRow[13] = 2;
        jCol[13] = 1;
        
        iRow[14] = 2;
        jCol[14] = 2;
        
        iRow[15] = 2;
        jCol[15] = 3;
        
        iRow[16] = 2;
        jCol[16] = 4;
        
        iRow[17] = 2;
        jCol[17] = 5;
/////////////////////////////////////////
        iRow[18] = 3;
        jCol[18] = 0;
        
        iRow[19] = 3;
        jCol[19] = 1;
        
        iRow[20] = 3;
        jCol[20] = 2;
        
        iRow[21] = 3;
        jCol[21] = 3;
        
        iRow[22] = 3;
        jCol[22] = 4;
        
        iRow[23] = 3;
        jCol[23] = 5;
    }
    else
    {
        // return the values of the Jacobian of the constraints
    
        values[0] = 1; // 0,0
        values[1] = 0; // 0,1
        values[2] = sin(x[2]) * x[4]; // 0,2
        values[3] = -cos(x[2]); // 0,3    
        values[4] = 0; // 0,4
        values[5] = 0; // 0,5

        values[6] = 0; // 1,0
        values[7] = 1; // 1,1
        values[8] = -cos(x[2]) * x[4]; // 1,2
        values[9] = sin(x[2]); // 1,3    
        values[10] = 0; // 1,4
        values[11] = 0; // 1,5

        values[12] = 0; // 2,0
        values[13] = 0; // 2,1
        values[14] = 1; // 2,2
        values[15] = 0; // 2,3    
        values[16] = 0; // 2,4
        values[17] = -0.1; // 2,5

        values[18] = 0; // 3,0
        values[19] = 0; // 3,1
        values[20] = 0; // 3,2
        values[21] = 1; // 3,3    
        values[22] = -0.1; // 3,4
        values[23] = 0; // 3,5
    }
    
    return true;
}

bool Optimizer::eval_h(
    Index         n,
    const Number* x,
    bool          new_x,
    Number        obj_factor,
    Index         m,
    const Number* lambda,
    bool          new_lambda,
    Index         nele_hess,
    Index*        iRow,
    Index*        jCol,
    Number*       values
){
    assert(n == NUMBER_OF_STATES);
    assert(m == 4);
    
    if( values == NULL )
    {
        // return the structure. This is a symmetric matrix, fill the lower left
        // triangle only.
    
        // the hessian for this problem is actually dense
        Index idx = 0;
        for( Index row = 0; row < NUMBER_OF_STATES; row++ )
        {
            for( Index col = 0; col <= row; col++ )
            {
                iRow[idx] = row;
                jCol[idx] = col;
                idx++;
            }
        }
    
        assert(idx == nele_hess);
    }
    else
    {
        // return the values. This is a symmetric matrix, fill the lower left
        // triangle only
    
        // fill the objective portion
        values[0] = obj_factor * (2 * W_GX);    // 0,0
    
        values[1] = 0.;                         // 1,0
        values[2] = obj_factor * (2 * W_GY);    // 1,1
    
        values[3] = 0.;                         // 2,0
        values[4] = 0.;                         // 2,1
        values[5] = obj_factor * (2 * W_H);     // 2,2
    
        values[6] = 0.;                         // 3,0
        values[7] = 0.;                         // 3,1
        values[8] = 0.;                         // 3,2
        values[9] = obj_factor * (2 * W_V);     // 3,3
    
        values[10] = 0.;                        // 4,0
        values[11] = 0.;                        // 4,1
        values[12] = 0.;                        // 4,2
        values[13] = 0.;                        // 4,3
        values[14] = obj_factor * (2 * W_ALPHA);// 4,4

        values[15] = 0.;                        // 5,0
        values[16] = 0.;                        // 5,1
        values[17] = 0.;                        // 5,2
        values[18] = 0.;                        // 5,3
        values[19] = 0.;                        // 5,4
        values[20] = obj_factor * (2 * W_Y);    // 5,5
    
        // add the portion for the first constraint
        values[5] -= lambda[0] * cos(x[2]) * x[3] * 0.1;    // 2,2
        values[8] += lambda[0] * sin(x[2]) * 0.1;           // 3,2

        // add the portion for the 2nd constraint
        values[5] -= lambda[2] * sin(x[2]) * x[3] * 0.1;    // 2,2
        values[8] -= lambda[0] * cos(x[2]) * 0.1;           // 3,2
        
        //others are zero
    }
    
    return true;
}

void Optimizer::finalize_solution(
    SolverReturn               status,
    Index                      n,
    const Number*              x,
    const Number*              z_L,
    const Number*              z_U,
    Index                      m,
    const Number*              g,
    const Number*              lambda,
    Number                     obj_value,
    const IpoptData*           ip_data,
    IpoptCalculatedQuantities* ip_cq){

    // here is where we would store the solution to variables, or write to a file, etc
    // so we could use the solution.
    
    // For this example, we write the solution to the console
    for( Index i = 0; i < n; i++ )
    {
        internal_state[i] = x[i];
    }
    std::cout << std::endl << std::endl << "Solution of the primal variables, x" << std::endl;
    for( Index i = 0; i < n; i++ )
    {
        std::cout << "x[" << i << "] = " << x[i] << std::endl;
    }
    /**
    std::cout << std::endl << std::endl << "Solution of the bound multipliers, z_L and z_U" << std::endl;
    for( Index i = 0; i < n; i++ )
    {
        std::cout << "z_L[" << i << "] = " << z_L[i] << std::endl;
    }
    for( Index i = 0; i < n; i++ )
    {
        std::cout << "z_U[" << i << "] = " << z_U[i] << std::endl;
    }
    
    std::cout << std::endl << std::endl << "Objective value" << std::endl;
    std::cout << "f(x*) = " << obj_value << std::endl;
    
    std::cout << std::endl << "Final value of the constraints:" << std::endl;
    for( Index i = 0; i < m; i++ )
    {
        std::cout << "g(" << i << ") = " << g[i] << std::endl;
    }*/
}

bool Optimizer::intermediate_callback(
    AlgorithmMode              mode,
    Index                      iter,
    Number                     obj_value,
    Number                     inf_pr,
    Number                     inf_du,
    Number                     mu,
    Number                     d_norm,
    Number                     regularization_size,
    Number                     alpha_du,
    Number                     alpha_pr,
    Index                      ls_trials,
    const IpoptData*           ip_data,
    IpoptCalculatedQuantities* ip_cq
){
    return true;
}