function h = outputFunction_thin_film_nio2(XZ,user_data_struct) % h is a 'vector valued' function.

    Z = XZ(user_data_struct.n_diff + 1:end);
    h = Z(1);
end