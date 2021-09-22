function LR_img = gen_LR_img( HR_img, scale, scale2 )

step = cal_step(scale);
step2 = cal_step(scale2);

% crop borders
[h, w, n]=size(HR_img);
h = round(floor(h / step / scale) * step * scale);
w = round(floor(w / step2 / scale2) * step2 * scale2);
HR_img_crop =HR_img(1:h,1:w,:);

% bicubic downsampling
LR_img= imresize(HR_img_crop, [round(h/scale), round(w/scale2)], 'bicubic');

end

function step = cal_step(scale)

if abs(scale - round(scale)) < 0.001
    step = 1;
elseif abs(scale * 2 - round(scale * 2)) < 0.001
    step = 2;
elseif abs(scale * 5 - round(scale * 5)) < 0.001
    step = 5;
elseif abs(scale * 10 - round(scale * 10)) < 0.001
    step = 10;
elseif abs(scale * 20 - round(scale * 20)) < 0.001
    step = 20;
end

end