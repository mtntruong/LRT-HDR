clear; %clc;

ppd = 30;

addpath('hdrvdp-2.2.2')
addpath('pu_log_psnr')

gt_dirs = './HDM/Test/HDR/';
gtNames = dir(fullfile(gt_dirs,'*.hdr'));

algo_list = dir('../Out_HDM/');

for al = 3 : length(algo_list)
    algo_name = algo_list(al).name;
    disp(algo_name)
    
    hdr_dirs = ['../Out_HDM/' algo_name '/'];
    imageNames = dir(fullfile(hdr_dirs,'*.hdr'));
    if ~exist([hdr_dirs 'Results/'], 'dir')
        mkdir([hdr_dirs 'Results/'])
    end

    mu_psnr  = zeros(1,length(gtNames));
    pu2_psnr = zeros(1,length(gtNames));
    pu2_msssim  = zeros(1,length(gtNames));
    Q_score = zeros(1,length(gtNames));
    P_score = zeros(1,length(gtNames));

    parfor i = 1 : length(gtNames)
        inp = hdrread([gt_dirs gtNames(i).name]) * 65535; inp = inp(51:930,51:1770,:);
        out = hdrread([hdr_dirs imageNames(i).name]) * 65535;

        mu_inp = inp / 65535;
        mu_out = out / 65535;
        t_inp = log(1 + 5000 * mu_inp) / log(1 + 5000);
        t_out = log(1 + 5000 * mu_out) / log(1 + 5000);
        mu_psnr(i) = psnr( t_out, t_inp );

        I_context = get_luminance( inp );
        res = hdrvdp( out, inp, 'rgb-bt.709', ppd );
        Q_score(i) = res.Q;
        P_score(i) = mean(res.P_map,'all');
        img = hdrvdp_visualize( res.P_map, I_context );

        inp_png = GammaTMO(ReinhardTMO(inp, 0.18), 2.2, 0, 0);
        out_png = GammaTMO(ReinhardTMO(out, 0.18), 2.2, 0, 0);

        % PSNR
        pu2_psnr(i) = qm_pu2_psnr( inp, out );
        pu2_msssim(i) = qm_pu2_msssim( inp, out );

        imwrite(img, [hdr_dirs 'Results/' gtNames(i).name '_MAP.png'])
        imwrite(out_png, [hdr_dirs 'Results/' gtNames(i).name '_LDR.png'])
    end

    mean(mu_psnr)
    mean(pu2_psnr)
    mean(pu2_msssim)
    mean(Q_score)
    mean(P_score)
    save(['./Result_MATs/HDM_' algo_name],'mu_psnr','pu2_psnr','pu2_msssim','Q_score','P_score')
end