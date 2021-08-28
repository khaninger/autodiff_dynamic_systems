%%% Data processing of datalogs from NJ60
% 5 - time
% 2 - looptime
%  3: 8 - x_actual
%  9:14 - x_target
% 15:20 - delta_pose
% 21:26 - q_actual
% 27:32 - c_force_c
% 33:38 - c_force_s
% 29 = Fz
% 
%%%% COLLISION DETECTION %%%%
% dat = load('datalogR3997.txt'); ind0 = 46575; % Flex jt, col det
% dat = load('datalogR7022.txt'); ind0 = 21593; %Yellow surf, col det
% dat = load('datalogR7450.txt'); ind0 = 28733; % Purp surf, col det
dat = load('datalogR9909.txt'); ind0 = 50169; % Yellow feet, col det
L = ind0-40:ind0+600;


f = dat(L,35);
x = 1000*sqrt(sum((dat(L, 3:5)-repmat(dat(ind0,3:5),length(L),1)).^2,2));
x(1:40) = -1*x(1:40);
% x0 = 1000*sqrt(sum((dat(L, 3:5)-repmat(dat(L(1),3:5),length(L),1)).^2,2));
v = conv(diff(x)*1000, 1/10*ones(10,1));
t = 1/1000*(0:(length(L)-1));
cont_det = find(f > 5.5);
figure(1)
plot([t(40), t(40)], [-10, 60],'k:', 'linewidth', 3)
hold on; grid on
plot([t(cont_det(1)), t(cont_det(1))], [-10, 60],'k:', 'linewidth', 3)
str = ['Contact detected in ',num2str(1/1000*(cont_det(1)-40)), ' sec'];
str2 = ['Final penetration ',num2str(x(end),'%.2f'), ' mm'];
plot(t,x,'k','linewidth',1.5)
plot(t,v(1:length(L)),'b','linewidth',1.5)
plot(t,f,'r','linewidth',1.5)
text(t(cont_det(1))+0.02, 45, str, 'Fontsize', 14, 'FontName', 'FixedWidth');
text(0.25, x(end)+4.5, str2, 'Fontsize', 12, 'FontName', 'FixedWidth');
axis tight
hold off

% legend('','','Position', 'Velocity', 'Force');
% ylabel('Force (N), Pos (mm), Vel (mm/sec)')
ylabel('')
xlabel('Time (sec)')
xlabel('')
hold off

%% Force control

% L = 1:60000;
% dat = load('datalogR3240.txt'); ind0 = 49934; % Flex jt
% dat = load('datalogR6951.txt'); ind0 = 16403; % Yellow surf
% dat = load('datalogR7382.txt'); ind0 = 4242; % Purp surf
dat = load('datalogR7945.txt'); ind0 = 50635; % Yellow feet

L = ind0-50:ind0+1500;
f = dat(L,35);
x = 1000*sqrt(sum((dat(L, 3:5)-repmat(dat(ind0,3:5),length(L),1)).^2,2));
x(1:50) = -1*x(1:50);
% x0 = 1000*sqrt(sum((dat(L, 3:5)-repmat(dat(L(1),3:5),length(L),1)).^2,2));
v = conv(diff(x)*1000, 1/10*ones(10,1));
t = 1/1000*(0:(length(L)-1));
figure(1)
hold on; grid on

plot(t,x,'k','linewidth',1.5)
plot(t,v(1:length(L)),'b','linewidth',1.5)
plot(t,f,'r','linewidth',1.5)
axis tight
ylim([-10, 55])
hold off

legend('Position', 'Velocity', 'Force');
% ylabel('Force (N), Pos (mm), Vel (mm/sec)')
ylabel('')
% xlabel('Time (sec)')
xlabel('')
hold off




%% Robot Time Response

L = 1:32000;
% dat = load('datalogR3240.txt'); ind0 = 49934; % Flex jt
% dat = load('datalogR6951.txt'); ind0 = 16403; % Yellow surf
dat = load('datalogR7382.txt');  % Purp surf
% dat = load('datalogR7945.txt'); ind0 = 50635; % Yellow feet

% L = ind0-50:ind0+1500;
% L = [900:8261, 44266:59000]; ind0 = 7519;
% L = 900:8261;, ind0 = 1;
% L = 44266:59000; ind0 = 1;
fd = dat(L,41);
f = dat(L,35);

x = dat(L, 3:5);
x = 1000*sqrt(sum((x-repmat(x(ind0, :),length(L),1)).^2,2));
xd = dat(L,9:11)+dat(L,15:17);
xd = 1000*sqrt(sum((xd-repmat(xd(ind0,:),length(L),1)).^2,2));

v = conv(diff(x)*1000, 1/10*ones(10,1));
t = 1/1000*(0:(length(L)-1));
figure(1)
plot(t,x,'k','linewidth',1.5)
hold on; grid on
plot(t,xd,'k:','linewidth',1.5)
% plot(t,v(1:length(L)),'b','linewidth',1.5)
plot(t,f,'r','linewidth',1.5)
plot(t,fd,'r:','linewidth',1.5)
axis tight
% ylim([-10, 55])
hold off

legend('Position', 'Velocity', 'Force');
% ylabel('Force (N), Pos (mm), Vel (mm/sec)')
ylabel('')
% xlabel('Time (sec)')
xlabel('')
hold off

%% Robot Freq Response

L = 1:60000;
% dat = load('datalogR3240.txt'); ind0 = 1; % Flex jt
% dat = load('datalogR6951.txt'); ind0 = 16403; % Yellow surf

dat = load('datalogR7382.txt'); ind0 = 1;  % Purp surf
% L = [44266:59000, 900:8261]; ind0 = 1;

% dat = load('datalogR7945.txt'); ind0 = 50635; % Yellow feet


fd = dat(L,41);
f = dat(L,35);
delta_f = (fd-f);
x = dat(L,3);
% x = (dat(L,3:5))+0.0*randn(length(L),3);
% x = (1000*sqrt(sum((x-repmat(x(ind0, :),length(L),1)).^2,2)));
xd = dat(L,9:11)+dat(L,15:17);
xd = 1000*sqrt(sum((xd-repmat(xd(ind0,:),length(L),1)).^2,2));
v = conv(diff(x)*1000, 1/10*ones(10,1));

% idd2 = iddata(x, xd, 1/1000);
figure(2)
fs = 1000;
NFFT = 2^nextpow2(length(L));
fr = fs/2*linspace(0,1,NFFT/2+1);
rng = 2:2500;
x2 = abs(fft(delta_f, NFFT));
y2 = abs(fft(x, NFFT));
x = x2(1:(NFFT/2+1));
x(2:end-1) = 2*x(2:end-1);
y = y2(1:(NFFT/2+1));
y(2:end-1) = 2*y(2:end-1);

adm = tf(1000,[12, 1000, 0]);
adm_fr = abs(squeeze(freqresp(adm, fr*2*pi)));
x = conv(x,lpf);
y = conv(y,lpf);

figure(2)
semilogx(fr(rng), 20*log10(y(rng))-20*log10(x(rng)),'k','linewidth',1.5)
hold on; grid on;
semilogx(fr(rng), 20*log10(adm_fr(rng)), 'k:', 'linewidth',1.5);
axis tight
title('|x|/|f-fd|')
xlabel('Frequency (Hz)')
ylabel('Magnitude (dB)')
axis tight

%% Test FFT

