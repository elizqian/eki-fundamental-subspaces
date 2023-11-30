function ensemblevis(problem,samples,ref,fignum,colorscheme)

if nargin < 5
    colorscheme = 'default';
end

if nargin < 4
    figure;
else
    figure(fignum); clf
end

if nargin < 3
    ref = 'LS';
end

switch colorscheme
    case 'default'
        statecolor=[0 0.447 0.741 0.1];
        meascolor =[255 154 113 25.5]/255;
        measdots  = [0.3 0.3 0.3 0.5];
        trcolor_meas   = [1 0 0];
        trcolor_st = [1 0 0];
        defline   = [0 0 0];
        bgcolor   = [1 1 1];
        tkcolor   = [0 0 0];
    case 'darkslides'
        statecolor = [108/255 173/255 220/255 0.3];
        meascolor = [255 154 113 25.5]/255;
        measdots  = [1 0 0 0.5];
        trcolor_meas   = [255 180 0]/255;
        trcolor_st = [122 198 255]/255;
        defline   = [231 230 231 150]/255;
        bgcolor   = [52 63 79]/255;
        tkcolor   = [231 230 231]/255;
end

subplot(2,1,1)
s0 = plot(problem.x,samples,'Color',statecolor); hold on
s1 = plot(problem.x,problem.truth,'Color',defline); hold on
switch ref
    case 'LS'
        s2 = plot(problem.x,problem.muLS,':','Color',trcolor_st);
        legend([s0(1),s1,s2],{'$u_i^{(j)}$','$u^\dagger$','$u_{\rm LS}$'},'interpreter','latex','location','southwest','orientation','horizontal','TextColor',tkcolor);
    case 'pos'
        s2 = plot(problem.x,problem.mupos,':','Color',trcolor_st);
        legend([s0(1),s1,s2],{'$u_i^{(j)}$','$u^\dagger$','$\mu_{\rm pos}$'},'interpreter','latex','location','southwest','orientation','horizontal','TextColor',tkcolor);
end
legend boxoff
title('State space $R^d$','interpreter','latex','Color',tkcolor,'FontSize',14)
set(gca,'XColor',tkcolor, 'YColor',tkcolor,'Color',bgcolor);
subplot(2,1,2)
m0 = plot(1:problem.n,problem.G*samples,'Color',meascolor); hold on
m1 = plot(1:problem.n,problem.G*problem.truth,'Color',defline);
m2 = plot(1:problem.n, problem.meas,'+','Color',measdots,'MarkerSize',4);
switch ref
    case 'LS'
        m3 = plot(1:problem.n,problem.G*problem.muLS,':','Color',trcolor_meas);
        legend([m0(1),m1,m2,m3],{'$Gu_i^{(j)}$','$Gu^\dagger$','$m$','$Gu_{\rm LS}$'},...
            'interpreter','latex','location','south','orientation','horizontal','TextColor',tkcolor);
    case 'pos'
        m3 = plot(1:problem.n,problem.G*problem.mupos,':','Color',trcolor_meas);
        legend([m0(1),m1,m2,m3],{'$Gu_i^{(j)}$','$Gu^\dagger$','$m$','$G\mu_{\rm pos}$'},...
            'interpreter','latex','location','south','orientation','horizontal','TextColor',tkcolor);
end
legend boxoff
xlim([1 problem.n])
title('Measurement space $R^n$','interpreter','latex','Color',tkcolor,'FontSize',14)
set(gca,'XColor',tkcolor, 'YColor',tkcolor,'Color',bgcolor);
set(gcf, 'InvertHardCopy', 'off'); 
set(gcf,'Color',bgcolor);