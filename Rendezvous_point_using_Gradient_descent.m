clear; clc;

% Convert ASV size from mm to meters
ASV_length = 1.2; % 1200 mm
ASV_width  = 0.5; % 500 mm

% Initial ASV positions
ASVs = [0, 0;
        30, 40;
        50, 15];

% Polygonal obstacles (distributed)
obstacles = {
    [40 15;40 10;45 10];
    [30 30; 25 30; 26 19]           % Triangle top left
};

% Parameters
lambda = 10;
P = 500;
k = 10;
alpha = 0.05;
maxIter = 500;
eps = 1e-6;

% Cost function
function c = costFcn(x, ASVs, obstacles, lambda, P, k)
    diffs = ASVs - x;
    dists = sqrt(sum(diffs.^2, 2));
    c = (1/lambda) * log(sum(exp(lambda*dists)));
    for obs = 1:length(obstacles)
        poly = obstacles{obs};
        inside = inpolygon(x(1), x(2), poly(:,1), poly(:,2));
        dmin = inf;
        for i = 1:size(poly,1)
            j = mod(i, size(poly,1)) + 1;
            a = poly(i,:); b = poly(j,:);
            v = b - a; w = x - a;
            cproj = dot(w,v);
            if cproj <= 0
                dist = norm(x - a);
            elseif cproj >= dot(v,v)
                dist = norm(x - b);
            else
                t = cproj / dot(v,v);
                proj = a + t*v;
                dist = norm(x - proj);
            end
            dmin = min(dmin, dist);
        end
        d_signed = dmin * (~inside*2 - 1);
        c = c + P * exp(-k * d_signed);
    end
end

% Phase 1: Local minima
localMins = zeros(3,2);
for i = 1:3
    x = ASVs(i,:);
    for iter = 1:maxIter
        grad = zeros(1,2);
        c0 = costFcn(x, ASVs, obstacles, lambda, P, k);
        for j = 1:2
            x_eps = x; x_eps(j) = x_eps(j) + eps;
            c1 = costFcn(x_eps, ASVs, obstacles, lambda, P, k);
            grad(j) = (c1 - c0) / eps;
        end
        x = x - alpha * grad;
        if norm(grad) < 1e-3, break; end
    end
    localMins(i,:) = x;
end

% Phase 2: Global rendezvous
costs = zeros(3,1);
for i = 1:3
    costs(i) = costFcn(localMins(i,:), ASVs, obstacles, lambda, P, k);
end
[~, bestIdx] = min(costs);
bestRendezvous = localMins(bestIdx,:);

% Move to goal with obstacle avoidance
function path = moveToGoal(start, goal, obstacles, P, k, alpha, maxIter, eps)
    x = start;
    path = x;
    for iter = 1:maxIter
        grad = zeros(1,2);
        c0 = norm(x - goal);
        for obs = 1:length(obstacles)
            poly = obstacles{obs};
            inside = inpolygon(x(1), x(2), poly(:,1), poly(:,2));
            dmin = inf;
            for ii = 1:size(poly,1)
                jj = mod(ii, size(poly,1)) + 1;
                a = poly(ii,:); b = poly(jj,:);
                v = b - a; w = x - a;
                cproj = dot(w,v);
                if cproj <= 0
                    dist = norm(x - a);
                elseif cproj >= dot(v,v)
                    dist = norm(x - b);
                else
                    t = cproj / dot(v,v);
                    proj = a + t*v;
                    dist = norm(x - proj);
                end
                dmin = min(dmin, dist);
            end
            d_signed = dmin * (~inside*2 - 1);
            c0 = c0 + P * exp(-k * d_signed);
        end
        for j = 1:2
            x_eps = x; x_eps(j) = x_eps(j) + eps;
            c1 = norm(x_eps - goal);
            for obs = 1:length(obstacles)
                poly = obstacles{obs};
                inside = inpolygon(x_eps(1), x_eps(2), poly(:,1), poly(:,2));
                dmin = inf;
                for ii = 1:size(poly,1)
                    jj = mod(ii, size(poly,1)) + 1;
                    a = poly(ii,:); b = poly(jj,:);
                    v = b - a; w = x_eps - a;
                    cproj = dot(w,v);
                    if cproj <= 0
                        dist = norm(x_eps - a);
                    elseif cproj >= dot(v,v)
                        dist = norm(x_eps - b);
                    else
                        t = cproj / dot(v,v);
                        proj = a + t*v;
                        dist = norm(x_eps - proj);
                    end
                    dmin = min(dmin, dist);
                end
                d_signed = dmin * (~inside*2 - 1);
                c1 = c1 + P * exp(-k * d_signed);
            end
            grad(j) = (c1 - c0) / eps;
        end
        x = x - alpha * grad;
        path(end+1,:) = x;
        if norm(grad) < 1e-3 && norm(x - goal) < 0.1
            break;
        end
    end
end

% Plotting
colors = {'b', 'g', 'm'};
figure; hold on; axis equal; grid on;
axis([-5 55 -5 55]);
title('ASV Paths with Obstacles and Rectangular Hulls');
xlabel('X'); ylabel('Y');

% Plot obstacles
for obs = 1:length(obstacles)
    patch(obstacles{obs}(:,1), obstacles{obs}(:,2), 'r', 'FaceAlpha', 0.3);
end

% ASV paths + hulls
for i = 1:3
    path1 = moveToGoal(ASVs(i,:), localMins(i,:), obstacles, P, k, alpha, maxIter, eps);
    path2 = moveToGoal(localMins(i,:), bestRendezvous, obstacles, P, k, alpha, maxIter, eps);
    full_path = [path1; path2];
    plot(full_path(:,1), full_path(:,2), '-', 'Color', colors{i}, 'LineWidth', 1.8);
    plot(ASVs(i,1), ASVs(i,2), 'ko', 'MarkerFaceColor', colors{i});
    
    % Plot ASV rectangle
    rect_x = ASVs(i,1) - ASV_length/2;
    rect_y = ASVs(i,2) - ASV_width/2;
    rectangle('Position', [rect_x, rect_y, ASV_length, ASV_width], ...
              'EdgeColor', colors{i}, 'LineWidth', 1.2);
end

% Final rendezvous
plot(bestRendezvous(1), bestRendezvous(2), 'kp', 'MarkerFaceColor','c', 'MarkerSize', 12);


% Save all ASV paths to CSV
sampled_points = 25;

for i = 1:3
    path1 = moveToGoal(ASVs(i,:), localMins(i,:), obstacles, P, k, alpha, maxIter, eps);
    path2 = moveToGoal(localMins(i,:), bestRendezvous, obstacles, P, k, alpha, maxIter, eps);
    full_path = [path1; path2];

    % Sample approximately 25–30 points
    indices = round(linspace(1, size(full_path,1), sampled_points));
    sampled_path = full_path(indices, :);

    % Save each ASV's path to a CSV file
    filename = sprintf('ASV%d_path.csv', i);
    csvwrite(filename, sampled_path);
end
