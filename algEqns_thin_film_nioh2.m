function [algebraiceqn_residual] = algEqns_thin_film_nioh2(Z,X,model_params)

    i0_1    = model_params.i0_1;
    i0_2    = model_params.i0_2;
    phi_eq1 = model_params.phi_eq1;
    phi_eq2 = model_params.phi_eq2;
    F       = model_params.F;
    R       = model_params.R;
    T       = model_params.T;
    i_app   = model_params.i_app;

    y1      = X(1);
    y2      = Z(1);

    j1 = i0_1*(2*(1-y1)*exp(0.5*F*(y2-phi_eq1)/(R*T)) - 2*y1*exp(-0.5*F*(y2-phi_eq1)/(R*T)));
    j2 = i0_2*(exp(F*(y2-phi_eq2)/(R*T)) - exp(-F*(y2-phi_eq2)/(R*T)));

    algebraiceqn_residual = j1 + j2 - i_app;
end