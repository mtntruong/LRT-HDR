function omega_out = create_omega(warped, warped_balanced, ref)

	ldr = warped;
	omega_rad = zeros(size(ldr,1),size(ldr,2),3);
	omega_rad(ldr >= 0.10 & ldr <= 0.99) = 1;

	ldr = warped_balanced;
	omega_warp = zeros(size(ldr,1),size(ldr,2),3);
	for c = 1 : 3
		channel = ldr(:,:,c);
		ref_channel = ref(:,:,c);
		[~, map] = ssim(channel, ref_channel);
		omega_warp(:,:,c) = imbinarize(max(map,0),0.9);
	end
	omega_out = omega_rad .* omega_warp;

end