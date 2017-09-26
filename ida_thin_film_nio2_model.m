% vim: set nospell nowrap textwidth=0 wrapmargin=0 formatoptions-=t:
function [overall_XZ_residual_vector, flag, new_data] = ida_thin_film_nio2_model(t,XZ,XZp,user_data_struct) % x_tot contains both x (differential states), z (algebraic variables) and derivates of states
% This function aims to return the residual of the combined diff+alg states residual.
% residual. This code will be repeatedly called by IDA in a time-stepping loop.
% Note. X: vector of differential (time-derivative) states, Z: vector of algebraic states

    %% dummy variables for IDA
    flag     = 0;  % These two variables are not used but required by IDA(s) solver.
    new_data = []; % These two variables are not used but required by IDA(s) solver.

    %% Unpack data from the 'UserData' structure into various fields
    model_params = user_data_struct.model_params;
    n_diff       = user_data_struct.n_diff;

    X  = XZ(1:n_diff); % state vector (differential variables only)
    Z  = XZ(n_diff+1:end);            % Build the array of algebraic variables
    Xp = XZp(1:n_diff);                         % retain only the first n_diff components of the combined derivative vector. Only these are required in the model equations below

    %% Assemble the overall augmented residual vector of the system [n_diff+n_alg x 1] column vector (the first n_diff components are residuals of differential variables and the rest of the components are residuals of algebraic variables)
    W = model_params.W;
    rho = model_params.rho;
    V = model_params.V;

    i0_1    = model_params.i0_1;
    phi_eq1 = model_params.phi_eq1;
    F       = model_params.F;
    R       = model_params.R;
    T       = model_params.T;

    y1      = X(1);
    y2      = Z(1);

    j1 = i0_1*(2*(1-y1)*exp(0.5*F*(y2-phi_eq1)/(R*T)) - 2*y1*exp(-0.5*F*(y2-phi_eq1)/(R*T)));

    rhs_of_state_eqn = (W/(rho*V*F))*j1;
    
    res_X_dot = Xp - rhs_of_state_eqn;
    res_Z = algEqns_thin_film_nioh2(Z,X,model_params);

    overall_XZ_residual_vector = [res_X_dot;res_Z];
end
