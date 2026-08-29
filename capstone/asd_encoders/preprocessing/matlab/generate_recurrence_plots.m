% Generates Recurrence Plots (RPs) from the Move4AS EEG dataset using the
% CRP Toolbox (http://tocsy.pik-potsdam.de/crp.php), for use as CNN encoder
% inputs downstream.
%
% Each dataset/eeg+gait/<S|P><id>/<walk|dance>/eeg_<group><id><task><block>.mat
% file holds one variable `eegDataT`, size [n_eeg_ch+2, n_samples]:
%   rows 1:n_eeg_ch      - EEG channels
%   row  n_eeg_ch+1      - within-trial phase trigger (1-6)
%   row  n_eeg_ch+2      - movement-type trigger (1-3)
% (see dataset/eeg+gait/Preprocess_EEG_rawfiles.m, which reads the same two
% trailing rows; sample rate there is 250 Hz, confirmed here against the
% 0.5s beep marker durations.)
%
% For every trial file: the EEG rows are windowed (win_sec/overlap, matching
% configs/eeg_config.yaml), each window is z-scored and resampled to
% img_size points, and one recurrence plot is built per channel. All
% channels for a window are stacked into [n_eeg_ch x eff_size x eff_size]
% and saved as a .mat file, mirroring the dataset's <group><id>/<task> tree.

clear; clc;

%% ---- Configuration -------------------------------------------------
data_dir   = fullfile('..', '..', 'dataset', 'eeg+gait');
output_dir = fullfile('..', '..', 'data', 'recurrence_plots');

tasks    = {'walk', 'dance'};
n_eeg_ch = 15;      % EEG channels; trailing 2 rows of eegDataT are triggers
sfreq    = 250;     % Hz (Move4AS EEG sample rate)

win_sec  = 4.0;     % window length,   matches configs/eeg_config.yaml
overlap  = 0.5;     % window overlap,  matches configs/eeg_config.yaml
img_size = 128;     % each window is resampled to this many points before
                     % embedding, so RP side length is a config knob rather
                     % than however many raw samples the window happens to have

% Phase-space embedding for the recurrence plot
m   = 3;    % embedding dimension
tau = 1;    % time delay (samples, post-downsampling)

% A fixed distance threshold is not comparable across channels/subjects
% (amplitude and noise differ), so the threshold is instead picked per
% window/channel to hit this target recurrence rate - the standard
% "fixed recurrence rate" thresholding scheme.
target_rr = 0.10;

save_example_png = true;   % dump one example RP image per trial for a quick visual check

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% ---- Probe the CRP toolbox once, fall back to plain MATLAB if absent ----
use_toolbox = false;
if exist('crp', 'file') == 2
    try
        crp(randn(1, 20), randn(1, 20), m, tau, 0.5, 'euclidean', 'nonormalize', 'silent');
        use_toolbox = true;
    catch ME
        warning(['CRP toolbox found but the trial call to crp(...) failed (%s). ' ...
                 'Falling back to a plain-MATLAB recurrence plot; check the crp() ' ...
                 'call syntax for your installed toolbox version.'], ME.message);
    end
else
    warning(['CRP toolbox function "crp" not found on the MATLAB path. ' ...
             'Falling back to a plain-MATLAB recurrence plot (same m/tau/target_rr). ' ...
             'Install the toolbox (http://tocsy.pik-potsdam.de/crp.php) and addpath() ' ...
             'it to use the toolbox implementation.']);
end

%% ---- Discover trials -------------------------------------------------
subject_dirs = [dir(fullfile(data_dir, 'S*')); dir(fullfile(data_dir, 'P*'))];
subject_dirs = subject_dirs([subject_dirs.isdir]);

win_len   = round(win_sec * sfreq);
step      = max(1, round(win_len * (1 - overlap)));
eff_size  = img_size - (m - 1) * tau;   % RP side length after embedding

fprintf('Found %d subject folders under %s\n', numel(subject_dirs), data_dir);
fprintf('Window: %d samples @ %dHz, step %d, resampled to %d pts, RP %dx%d\n', ...
        win_len, sfreq, step, img_size, eff_size, eff_size);

total_windows = 0;
tic;

for si = 1:numel(subject_dirs)
    subj  = subject_dirs(si).name;         % e.g. 'S3' or 'P7'
    label = double(subj(1) == 'P');        % P = ASD group, S = control

    for ti = 1:numel(tasks)
        task = tasks{ti};
        files = dir(fullfile(data_dir, subj, task, 'eeg_*.mat'));

        for fi = 1:numel(files)
            file_path = fullfile(files(fi).folder, files(fi).name);
            [~, trial_name] = fileparts(files(fi).name);   % e.g. eeg_S3walk1

            S = load(file_path, 'eegDataT');
            eeg = S.eegDataT(1:n_eeg_ch, :);   % drop the 2 trigger rows

            n_samples = size(eeg, 2);
            n_windows = max(0, floor((n_samples - win_len) / step) + 1);
            if n_windows == 0
                fprintf('  skip %s: only %d samples (< %d needed)\n', ...
                        trial_name, n_samples, win_len);
                continue;
            end

            out_subdir = fullfile(output_dir, subj, task);
            if ~exist(out_subdir, 'dir')
                mkdir(out_subdir);
            end
            fprintf('%s/%s/%s: %d windows\n', subj, task, trial_name, n_windows);

            % One .mat per trial, all windows stacked, rather than one file
            % per window: a full run produces ~270 trial files instead of
            % ~65k window files, which matters a lot once this output has to
            % be zipped/synced through Google Drive for Colab training.
            RP_all = false(n_windows, n_eeg_ch, eff_size, eff_size);

            for wi = 1:n_windows
                start_idx = (wi - 1) * step + 1;
                w = eeg(:, start_idx:start_idx + win_len - 1);

                for ch = 1:n_eeg_ch
                    sig = w(ch, :);
                    sig = (sig - mean(sig)) / (std(sig) + 1e-8);
                    sig = interp1(1:win_len, sig, linspace(1, win_len, img_size), 'linear');

                    X = local_embed(sig, m, tau);
                    D = local_pairwise_dist(X);
                    off_diag = D(~eye(size(D)));
                    thresh = local_quantile(off_diag, target_rr);

                    rp_ch = [];
                    if use_toolbox
                        try
                            rp_ch = crp(sig, sig, m, tau, thresh, 'euclidean', 'nonormalize', 'silent');
                        catch
                            rp_ch = [];   % fall through to the plain-MATLAB path below
                        end
                    end
                    if isempty(rp_ch)
                        rp_ch = D <= thresh;
                    end

                    RP_all(wi, ch, :, :) = logical(rp_ch);
                end
            end

            out_name = sprintf('%s.mat', trial_name);
            out_path = fullfile(out_subdir, out_name);
            subject_id = subj; %#ok<NASGU>
            save(out_path, 'RP_all', 'subject_id', 'task', 'label', 'n_windows');
            % default (v7) format, not v7.3/HDF5: keeps files readable by
            % scipy.io.loadmat() directly on the Colab side, no h5py needed

            if save_example_png
                png_path = fullfile(out_subdir, sprintf('%s_ch01_example.png', trial_name));
                imwrite(squeeze(RP_all(1, 1, :, :)), png_path);
            end

            total_windows = total_windows + n_windows;
        end
    end
end

elapsed = toc;
fprintf('Done: %d windows written under %s in %.1f s.\n', total_windows, output_dir, elapsed);

%% ---- Local functions --------------------------------------------------
% pdist/squareform/quantile need the Statistics and Machine Learning
% Toolbox, which may not be installed alongside the CRP toolbox - so the
% embedding, distance matrix and thresholding below use only base MATLAB.

function X = local_embed(x, m, tau)
% Time-delay phase-space embedding of a 1D signal: X(i,:) = x(i : tau : i+(m-1)*tau).
    n = length(x) - (m - 1) * tau;
    X = zeros(n, m);
    for i = 1:m
        X(:, i) = x((1:n) + (i - 1) * tau);
    end
end

function D = local_pairwise_dist(X)
% Euclidean distance matrix between rows of X, via the Gram matrix
% (avoids the Statistics Toolbox's pdist/squareform).
    G = X * X';
    sq = diag(G);
    D2 = sq + sq' - 2 * G;
    D2(D2 < 0) = 0;   % guard against tiny negative values from float round-off
    D = sqrt(D2);
end

function q = local_quantile(v, p)
% Linear-interpolation quantile of vector v at probability p in [0,1]
% (avoids the Statistics Toolbox's quantile).
    v = sort(v(:));
    n = numel(v);
    idx = p * (n - 1) + 1;
    lo = floor(idx); hi = ceil(idx);
    if lo == hi
        q = v(lo);
    else
        q = v(lo) + (idx - lo) * (v(hi) - v(lo));
    end
end
