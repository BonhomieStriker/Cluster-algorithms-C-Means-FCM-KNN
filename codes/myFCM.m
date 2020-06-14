function [center, U, obj_fcn] = myFCM(data,c,options)
% Input£º
% data n*m n: num of samples | m: num of features 
% c: num of centers
% according to the fcm function in MATLAB
%options(1): membership matrix U param expo£¬>1(default: 2.0)
%options(2): maxmimum iteration time max_t(default: 100)
%options(3): minimum change of membership e(dafault: 1e-5)
%options(4): print the iteration info(default: 1)
% Output£º
% U: membership matrix
% center: centers of clusters 
% obj_fcn: objective function 


if nargin ~= 2 && nargin ~= 3
    error('Too many or too few input argument! ');
end
data_n = size(data, 1);
data_m = size(data, 2);


default_options = [2; 100; 1e-5; 1];


if nargin == 2
    options = default_options;
else 
    if length(options) < 4
        temp = default_options;
        temp(1:length(options)) = options;
        options = temp;
    end
    nan_index = find(isnan(options) == 1);
    options(nan_index) = default_options(nan_index);
    if options(1) <= 1 
        error('The exponent should be greater than 1 !');
    end

end

expo = options(1);%fuzzy parameter 
max_t = options(2);% maximum iteration
e = options(3);%condition of end
display = options(4);%disp

obj_fcn = zeros(max_t, 1);
U = initfcm(c, data_n);

for i = 1 : max_t
    [U, center, obj_fcn(i)] = stepfcm(data, U, c, expo);
    if display
        fprintf('FCM:Iteration count = %d, obj_fcn = %f\n',i,obj_fcn(i));
    end
    if i > 1
        if abs(obj_fcn(i) - obj_fcn(i-1)) < e
            break;
        end
    end

end
iter_n = i;
obj_fcn(iter_n + 1 : max_t) = [];
U = U';
end
