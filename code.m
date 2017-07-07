%% Numerical Solution to the Monge's optimal transport problem on a unit square
%% which is a 2-D Monge-Ampere PDE where the densities are images.

% For this function the source image and target image are in vector form. 

% Example input for X = source_image, Y = target_image 
% Wasserstein_distance = paper_1(X, Y, 20, 1e-05, 1, 1)

function [Wasserstein] = paper_1(source_image, target_image, max_iterations, tol, dampening, epsilon)

row = 28; col = 28; % dimensions of the image

h = 1/(col-1); k = 1/(row-1); % step size is equal to 1/(dimension-1) on a unit square

% Create mesh grid for numerical solution

[x1,x2] = meshgrid(0:h:1, 0:k:1);
x2 = flipud(x2);

%% Numerically normalise images such that \int F(x) dx = \int G(y) dy = 1, and ensure they are non-zero

F = source_image + epsilon;
F = F/(trapz(0:k:1,trapz(0:h:1,reshape(F,[row,col]),2)));
fpic = reshape(F,[row,col]);

G = target_image + epsilon;
G = G/(trapz(0:k:1,trapz(0:h:1,reshape(G,[row,col]),2)));
G = reshape(G,[row,col]); gpic = G;

%% Turn images into a surface using cubic spline interpolation 

[Gy1, Gy2] = gradient(G,h,k); Gy2 = -Gy2;

[grid1, grid2] = ndgrid(0:h:1, 0:k:1);

Gi = griddedInterpolant(grid1, grid2, G, 'spline');
Gi1 = griddedInterpolant(grid1, grid2, Gy1, 'spline');
Gi2 = griddedInterpolant(grid1, grid2, Gy2, 'spline');

%% Set up derivative operators for finite difference scheme

D1  = gallery('tridiag',col, -ones(1,col-1), zeros(1,col), ones(1,col-1));  % e.g. u*Dx1
D2 = gallery('tridiag',row, -ones(1,row-1), zeros(1,row), ones(1,row-1));    % e.g. Dx2*u
D11 = gallery('tridiag',col, ones(1,col-1), -2*ones(1,col), ones(1,col-1));    % e.g. u*Dx1x1
D22 = gallery('tridiag',row, ones(1,row-1), -2*ones(1,row), ones(1,row-1));    % e.g. Dx2x2*u

Right = (sparse(2:col,1:col-1,ones(1,col-1),col,col))'; Right(1,2) = 2;
Left = sparse(2:col,1:col-1,ones(1,col-1),col,col); Left(col,col-1) = 2;
Up = (sparse(2:row,1:row-1,ones(1,row-1),row,row))'; Up(1,2) = 2;
Down = sparse(2:row,1:row-1,ones(1,row-1),row,row); Down(row,row-1) = 2;

% Impose purely homogeneous Neumann BCs

D11(1,2) = 2*D11(1,2); D11(col,col-1) = 2*D11(col,col-1);
D22(1,2) = 2*D22(1,2); D22(row,row-1) = 2*D22(row,row-1);
D2(1,2) = 0; D2(row,row-1) = 0;
D1(1,2) = 0; D1(col,col-1) = 0;

D1 = D1'; D11 = D11';

%% Set up derivative operators using the Kronecker product

Dx1 = kron(D1',speye(row)); 
Dx2 = -kron(speye(col),D2);
Dx1x1 = kron(D11',speye(row));
Dx2x2 = kron(speye(col), D22);
Dx1x2 = -kron(D1',D2);

%% Initialise parameters for damped Newton scheme

error = 1;
iteration = 1;

u = zeros(row,col);
u = u(:);

%% Nonlinear solver using damped Newton iteration

while(iteration < max_iterations && abs(error) > tol)

    % Derivatives of the function u(x1, x2)
      
    Ux1 = Dx1*u(:)/(2*h); Ux1 = reshape(Ux1,[row,col]);
    Ux2 = Dx2*u(:)/(2*k); Ux2 = reshape(Ux2,[row,col]);
    Ux1x1 = Dx1x1*u(:)/h^2; 
    Ux2x2 = Dx2x2*u(:)/k^2; 
    Ux1x2 = Dx1x2*u(:)/(4*h*k); 
    
    x_Ux1 = x1 - Ux1;
    x_Ux2 = x2 - Ux2;
    
    % Density g(y) and its derivatives
    
    G = Gi(1-x_Ux2, x_Ux1);
    G = G(:);
    
    G1 = Gi1(1-x_Ux2, x_Ux1);
    G1 = G1(:);
    
    G2 = Gi2(1-x_Ux2, x_Ux1);
    G2 = G2(:);
    
    % Refer to paper for details of coefficients to the numerical scheme
    
    A =  1 - Ux1x1 - Ux2x2 - Ux1x2.^2 + Ux1x1.*Ux2x2 ;
    B =  Ux2x2 - 1;
    C =  Ux1x1 - 1;
    D = -2*Ux1x2; 
    
    % Error as range between absolute value between the maximum and minimum residual on the domain
    
    Residual = ((1-Ux1x1).*(1-Ux2x2) - Ux1x2.^2).*G - F;
    error = abs(max(Residual) - min(Residual));
    
    % Set up sparse coefficient matrix for solver
    
    W = bsxfun(@times, speye(row*col), -2*G.*(B/h^2 + C/k^2)) + ... 
        bsxfun(@times, kron(speye(row),Right), B.*G/h^2 - A.*G1/(2*h)) + ...
        bsxfun(@times, kron(speye(row),Left), B.*G/h^2 + A.*G1/(2*h)) + ...
        bsxfun(@times, kron(Up,speye(col)), C.*G/k^2 - A.*G2/(2*k)) + ...
        bsxfun(@times, kron(Down,speye(col)), C.*G/k^2 + A.*G2/(2*k)) + ...
        bsxfun(@times, -kron(D1',D2), G.*D/(4*h*k));
 
    W = [W ones(row*col,1); ones(1,row*col+1)]; W(row*col+1,row*col+1) = 0;
    
    B = F - G.*A; B  = [B;0]; 
    
    % Solve linear system

    ew = W\B;   

    % Update Newton step 
    
    u = u + dampening*ew(1:row*col);

    iteration = iteration + 1;

end

%% plotting for visualisation of the mapping once the solution has converged

u = reshape(u,[row,col]);

Ux1 = Dx1*u(:)/(2*h);
Ux1 = reshape(Ux1, [row,col]);
Ux2 = Dx2*u(:)/(2*k); 
Ux2 = reshape(Ux2, [row,col]);

% Optimal mapping is s(x) = x - u_x(x), plotted with mesh generation 

sx = x1 - Ux1;
sy = x2 - Ux2;

% Relaxed Wasserstein metric 

Wasserstein = 1000*sqrt(trapz(0:k:1,trapz(0:h:1,(Ux1.^2+Ux2.^2).*fpic,2)));

figure(iteration)
    subplot(2,2,1), surf(x1,x2,fpic), title('Source Distribution'), xlabel('x1'), ylabel('x2');
    subplot(2,2,2), surf(x1,x2,gpic), title('Target Distribution'), xlabel('y1'), ylabel('y2');
    subplot(2,2,3), surf(x1,x2,u), title(sprintf('Solution u(x1,x2), Residual %e',error)), xlabel('x1'), ylabel('x2');;  
    subplot(2,2,3), plot(sx(1:4:end,:)', sy(1:4:end,:)', 'r.-', sx(1:4:end,:), sy(1:4:end,:), 'b.-'), title('Optimal Mapping'), axis equal;
figure(iteration+1); 
plot(sx(1:2:end,:)', sy(1:2:end,:)', 'r.-', sx(1:2:end,:), sy(1:2:end,:), 'b.-'), title('Optimal Mapping'), axis equal;

end
