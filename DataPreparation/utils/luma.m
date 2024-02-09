function output = luma(input)

    a = 17.554;
    b = 826.81;
    c = 0.10013;
    d = -884.17;
    e = 209.16;
    f = -731.28;
    yl = 5.6046;
    yh = 10469;
    
    output = input;

    % y < yl
    output(input < yl) = a * input(input < yl);
    % yl <= y < yh
    output((input >= yl) & (input < yh)) = b * input((input >= yl) & (input < yh)).^c + d;
    % y >= yh
    output(input >= yh) = e * log(input(input >= yh)) + f;
    
    output = output ./ 4096;

end

