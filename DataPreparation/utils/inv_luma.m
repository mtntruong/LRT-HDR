function output = inv_luma(input)

    a = 0.056968;
    b = 7.3014e-30;
    c = 9.9872;
    d = 884.17;
    e = 32.994;
    f = 0.0047811;
    ll = 98.381;
    lh = 1204.7;

    input = input .* 4096;
    output = input;

    % y < yl
    output(input < ll) = a * input(input < ll);
    % yl <= y < yh
    output((input >= ll) & (input < lh)) = b * (input((input >= ll) & (input < lh)) + d).^c;
    % y >= yh
    output(input >= lh) = e * exp(f * input(input >= lh));

end

