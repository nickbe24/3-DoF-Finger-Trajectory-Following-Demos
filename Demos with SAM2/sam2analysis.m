
%% Square Demos
clear
points_per_edge = 2; 
points_between = points_per_edge - 1; 
path_pts = [];
corners = [0 60 60; 0 70 60; 0 70 50; 0 60 50]; % YZ plane (no abduction)
%corners = [-10 60 60; 10 60 60; 10 60 40; -10 60 40]; % ZX plane (abductino)
%corners =  [-10 60 20; -10 80 20; 10 80 20; 10 60 20]; %XY plane
%corners = [0 50 -20; 0 50 -30; 0 60 -30; 0 60 -20]; % YZ lower workspace
for i = 1:4
    p1 = corners(i, :);
    p2 = corners(mod(i,4)+1, :);
    for j = 0:(points_between - 1)  % exclude last point
        t = j / points_between;
        pt = (1 - t) * p1 + t * p2;
        path_pts(end+1, :) = pt;
    end
end
% Append the first corner to close the loop
path_pts(end+1, :) = corners(1, :);
%for i = 1:size(path_pts, 1)
%    fprintf('%.2f %.2f %.2f\n', path_pts(i, :));
%end
%% Circle demo YZ No abduction
clear
r = 10;
center = [0, 70, 20];
N = 20;  % number of points
% Angles evenly spaced around circle
theta = linspace(0, 2*pi, N + 1);
% Compute points
x_px = zeros(1, N+1);
y_px = r * cos(theta) + center(2);
z = r * sin(theta) + center(3);
path_pts = [x_px', y_px', z'];

%% Step path
clear
path_pts = [0 50 20; 0 53 20; 0 53 23; 0 56 23; 0 56 26; 0 59 26; 0 59 29; 0 62 29; 0 62 32; 0 50 20];

%% Compare input path with SAM extracted path 
%Get mm / pix from paint

cube_length = 3; s_cube = .025; % mm
x_px = 3; s_xp_x = 2/sqrt(3); %px
y_px = 78; s_yp_x = 4/sqrt(3); %px
r = hypot(x_px,y_px);
mpp = cube_length/r;
s_mpp = sqrt((s_cube/r)^2 + (cube_length*x_px*s_xp_x/r^3)^2 + (cube_length*y_px*s_yp_x/r^3)^2);


% load data
data = load('square_set_02/square_5mms_01_coords.txt'); % ==========================================================
path_px = data - data(1,:); %convert to relative

%format data
path_px = [path_px(:,1) -path_px(:,2)]; %invert y so + is up
%path_mm = path_mm(1:end-1,:); %clip last point if mask got messed up

s_ypx2 = 1/sqrt(3); % px
s_zpx2 = 1/sqrt(3); % px

%convert these to vector operations or integrate into trajectory forloop


%xymm_cor = xmm*ymm*s_mpp;
%norm_mm = hypot(xmm, ymm); % needs to be made relative to the trajectory measurement

% Data to save
fps = 30;
v = 5; % mm/s
ds = v/fps; % mm per frame

P = path_pts(:, 2:3);         
dP = diff(P, 1, 1); % segment vectors
segL = sqrt(sum(dP.^2, 2)); % segment lengths
M = size(dP, 1); % number of segments 

N = size(path_px, 1); % number of frames

% recorded errors
err_y = zeros(N,1); s_y_mms = zeros(N,1);
err_z = zeros(N,1); s_z_mms = zeros(N,1);
err_norm = zeros(N,1);
s_err_norms = zeros(N,1);
path_mm = zeros(N,2);

seg = 1; % current segment
pos = P(1,:);                 
rem = segL(1);

for i = 1:N

    y_px = path_px(i, 1);       
    z_px = path_px(i, 2);   
    y_mm = y_px*mpp;  s_y_mm = sqrt((s_ypx2/y_px)^2 + (s_mpp/mpp)^2) * abs(y_mm); s_y_mms(i) = s_y_mm;
    z_mm = z_px*mpp;  s_z_mm = sqrt((s_zpx2/z_px)^2 + (s_mpp/mpp)^2) * abs(z_mm); s_z_mms(i) = s_z_mm;
    norm_mm = hypot(y_mm,z_mm);
    yz_mm_cor = y_mm*z_mm*s_mpp;
    if norm_mm < 1e-6
        s_err_norms(i) = 0;
    else
        s_err_norms(i) = sqrt((y_mm/norm_mm)^2*s_y_mm^2 + (z_mm/norm_mm)^2*s_z_mm^2 + 2*yz_mm_cor*(y_mm*z_mm)/norm_mm^2);
    end

    path_mm(i,:) = [y_mm+path_pts(1,2), z_mm+path_pts(1,3)];
    ay = path_mm(i,1);
    az = path_mm(i,2);
    ey = pos(1) - ay;
    ez = pos(2) - az;

    err_y(i)    = ey;
    err_z(i)    = ez;
    err_norm(i) = hypot(ey, ez);
    travel = ds;

    while travel > 0 && seg <= M
        if travel < rem - 1e-12
            t = travel / segL(seg);
            pos = pos + t * dP(seg, :);
            rem = rem - travel;
            travel = 0;
        else
            t = rem / segL(seg);
            pos = pos + t * dP(seg, :);
            travel = travel - rem;
            seg = seg + 1;
            if seg <= M
                rem = segL(seg);
            end
        end
    end

    % if frames over flows use the last position
    if seg > M
        pos = P(end, :); 
    end

end



h = figure;
h_input = plot(path_pts(:,2), path_pts(:,3), '-', 'DisplayName','Input Path'); 
hold on;
plot(path_mm(:,1), path_mm(:,2), '-', 'DisplayName','Actual Path');
%plot(path_mm_low(:,1), path_mm_low(:,2))
%plot(path_mm_high(:,1), path_mm_high(:,2))

% plot points at input points 
plot(path_pts(:,2), path_pts(:,3), '.', 'MarkerSize', 12, 'Color', h_input.Color, 'DisplayName','Input Points');

% plot circle at actual path start point
startY = path_mm(1,1);  startZ = path_mm(1,2);
plot(startY, startZ, 'o', 'MarkerSize', 10, 'LineWidth', 1.5, 'MarkerFaceColor','none', 'Color', 'black', 'DisplayName','Start (Actual)');

% plot circle at actual path end point
endY   = path_mm(end,1); endZ   = path_mm(end,2);
plot(endY,   endZ,   'o', 'MarkerSize', 10, 'LineWidth', 1.5, 'MarkerFaceColor','none', 'Color', 'red', 'DisplayName','End (Actual)');

title("Square Path @5mm/s 01")  % ==========================================================
xlabel('Y-axis (mm)')
ylabel('Z-axis (mm)')
lgd = legend('location','northeast');
axis([58 78 48 68])
xticks(58:1:78);
yticks(48:1:68);
grid on

h.UserData.end_pt_diff = norm([endY-startY, endZ-startZ]);
end_pt_diff = norm([endY-startY, endZ-startZ])
h.UserData.avg_norm_error = mean(err_norm);
avg_norm_error = mean(err_norm)
h.UserData.max_norm_error = max(err_norm);
max_norm_error = max(err_norm)
%savefig(h, fullfile("tracked_graphs_02", 'square_5mms_01.fig')); % ===================================================

%% for ad

%Get mm / pix from paint
mm_per_pix = 3/norm([88 15]); 
%mm_per_pix = .037;

data = load('ad_square_set_02/ad_square_20mms_05_coords.txt');  %====================================================================
path_px = data - data(1,:);

path_mm = path_px * mm_per_pix;
% path_mm_low = path_px*mm_per_pix_lower;
% path_mm_high = path_px*mm_per_pix_upper;

path_mm = path_mm - path_mm(1,:);
path_mm = [path_mm(:,1) -path_mm(:,2)];
path_mm = path_mm + [path_pts(1,1) path_pts(1,3)];
%path_mm = path_mm(1:end-1,:);

% path_mm_low = path_mm_low - path_mm_low(1,:);
% path_mm_low = [path_mm_low(:,1) -path_mm_low(:,2)];
% path_mm_low = path_mm_low + [path_pts(1,2) path_pts(1,3)];
% path_mm_low = path_mm_low(1:end-1,:);
% 
% path_mm_high = path_mm_high - path_mm_high(1,:);
% path_mm_high = [path_mm_high(:,1) -path_mm_high(:,2)];
% path_mm_high = path_mm_high + [path_pts(1,2) path_pts(1,3)];
% path_mm_high = path_mm_high(1:end-1,:);

h = figure;
hold on;

% Plot Input Path (line)
h_input = plot(path_pts(:,1), path_pts(:,3), '-', 'DisplayName','Input Path'); 
hold on;

% Plot Actual Path (line)
h_actual = plot(path_mm(:,1), path_mm(:,2), '-', 'DisplayName','Actual Path');

% Points at each Input Path waypoint, same color as line
plot(path_pts(:,1), path_pts(:,3), '.', ...
     'MarkerSize', 12, ...
     'Color', h_input.Color, ...
     'DisplayName','Input Points');

% Start circle on Actual Path (same color as Actual Path line)
startX = path_mm(1,1);  startZ = path_mm(1,2);
plot(startX, startZ, 'o', 'MarkerSize', 10, 'LineWidth', 1.5, ...
     'MarkerFaceColor','none', 'Color', 'black', ...
     'DisplayName','Start (Actual)');

% End circle on Actual Path (different color, e.g., red)
endX   = path_mm(end,1); endZ   = path_mm(end,2);
plot(endX,   endZ,   'o', 'MarkerSize', 10, 'LineWidth', 1.5, ...
     'MarkerFaceColor','none', 'Color', [1 0 0], ... % RGB for red
     'DisplayName','End (Actual)');

title("Ad Square Path @20mm/s 05") % =======================================================================
xlabel('X-axis (mm)')
ylabel('Z-axis (mm)')

lgd = legend('location','northeast');

axis([-12 22 37 72])
xticks(-12:1:22);
yticks(37:1:72);
grid on

% mm/px note near legend
lgdPos = lgd.Position;
text(lgdPos(1)+.04, lgdPos(2) - 0.01, sprintf('mm/px: %.3f', mm_per_pix), ...
     'Units', 'normalized', 'FontSize', 10);


% Data to save
fps = 30;
v = 20; % mm/s
ds = v/fps; % mm per frame

P = [path_pts(:, 1) path_pts(:,3)];         
dP = diff(P, 1, 1); % segment vectors
segL = sqrt(sum(dP.^2, 2)); % segment lengths
M = size(dP, 1); % number of segments 

N = size(path_mm, 1); % number of frames

% recorded errors
err_y = zeros(N,1);
err_z = zeros(N,1);
err_norm = zeros(N,1);

seg = 1; % current segment
pos = P(1,:);                 
rem = segL(1);

for i = 1:N

    ay = path_mm(i, 1);   
    az = path_mm(i, 2);  
    ey = pos(1) - ay;
    ez = pos(2) - az;

    err_y(i)    = ey;
    err_z(i)    = ez;
    err_norm(i) = hypot(ey, ez);
    travel = ds;

    while travel > 0 && seg <= M
        if travel < rem - 1e-12
            t = travel / segL(seg);
            pos = pos + t * dP(seg, :);
            rem = rem - travel;
            travel = 0;
        else
            t = rem / segL(seg);
            pos = pos + t * dP(seg, :);
            travel = travel - rem;
            seg = seg + 1;
            if seg <= M
                rem = segL(seg);
            end
        end
    end

    % if frames over flows use the last position
    if seg > M
        pos = P(end, :); 
    end

end

error = [err_y err_z err_norm];
h.UserData.end_pt_diff = norm([endX-startX, endZ-startZ]);
end_pt_diff = norm([endX-startX, endZ-startZ])
h.UserData.avg_norm_error = mean(err_norm);
avg_norm_error = mean(err_norm)
h.UserData.max_norm_error = max(err_norm);
max_norm_error = max(err_norm)

savefig(fullfile("tracked_graphs_02", 'ad_square_20mms_05.fig')); % ================================================


%% Compare input path with SAM extracted paths for 5 runs

% Base filename 
basefile = 'square_set_02/square_5mms_';
nSets = 5; % number of sets

h = figure;
h_input = plot(path_pts(:,2), path_pts(:,3), '-', 'DisplayName','Input Path'); 
hold on;

% Plot points at input path
plot(path_pts(:,2), path_pts(:,3), '.', 'MarkerSize', 12, ...
    'Color', h_input.Color, 'DisplayName','Input Points');

% Loop through datasets
for k = 1:nSets
    % Construct filename with padded index
    fname = sprintf('%s%02d_coords.txt', basefile, k);
    if k == 1
        mm_per_pix = 3/norm([3 78]);
        elseif k == 2
            mm_per_pix = 3/norm([2 76]);
        elseif k == 3
            mm_per_pix = 3/norm([3 70]);
        elseif k == 4
            mm_per_pix = 3/norm([5 93]);
        else
            mm_per_pix = 3/norm([4 90]);
    end
    % Load data
    data = load(fname);
    path_px = data - data(1,:);
    path_mm = path_px * mm_per_pix;
    path_mm = path_mm - path_mm(1,:);
    path_mm = [path_mm(:,1) -path_mm(:,2)];
    path_mm = path_mm + [path_pts(1,2) path_pts(1,3)];
    path_mm = path_mm(1:end,:);
    
    % Plot actual path
    plot(path_mm(:,1), path_mm(:,2), '-', 'DisplayName',sprintf('Actual Path %02d',k));
    
    % Start marker
    startY = path_mm(1,1);  startZ = path_mm(1,2);
    plot(startY, startZ, 'o', 'MarkerSize', 8, 'LineWidth', 1.2, ...
        'MarkerFaceColor','none', 'Color','black', 'HandleVisibility','off');
    
    % End marker
    endY   = path_mm(end,1); endZ   = path_mm(end,2);
    plot(endY, endZ, 'o', 'MarkerSize', 8, 'LineWidth', 1.2, ...
        'MarkerFaceColor','none', 'Color','red', 'HandleVisibility','off');
end

% Final plot settings
title("Square Path @5mm/s (All Runs)")
xlabel('Y-axis (mm)')
ylabel('Z-axis (mm)')
lgd = legend('location','northeast');
axis([58 78 48 68])
xticks(58:1:78);
yticks(48:1:68);
grid on
