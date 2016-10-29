radias = 1;
u_deltaLength = 0.2;
f_deltaLength = 0.3;
fileName = 'sphere_2_3.mat';
origin = [0, 0, 0];
u = [1, 0, 0, 0, 0, 0];     % [vx, vy, vz, wx, wy, wz]

fd=@(p) dsphere(p,origin(1), origin(2), origin(3), radias);
[f_nodes, f_mesh]=distmeshsurface(fd, @huniform, ...
                f_deltaLength, 1.1 * radias * [-1, -1, -1 ; 1, 1, 1]);
     
[u_nodes, u_mesh] = distmeshsurface(fd, @huniform, ...
                u_deltaLength, 1.1 * radias * [-1, -1, -1 ; 1, 1, 1]);

U = u_nodes; 
U(:, 1) = u(1) + u(5) * (u_nodes(:, 3) - origin(3)) - u(6) * (u_nodes(:, 2) - origin(2));
U(:, 2) = u(2) + u(6) * (u_nodes(:, 1) - origin(1)) - u(4) * (u_nodes(:, 3) - origin(3));
U(:, 3) = u(3) + u(4) * (u_nodes(:, 2) - origin(2)) - u(5) * (u_nodes(:, 1) - origin(1));

save(fileName, 'f_nodes', 'u_nodes', 'U', 'origin')