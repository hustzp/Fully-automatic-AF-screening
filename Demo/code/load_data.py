
import os
import numpy as np
import wfdb


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def load_data_afdb(data_path, file):
    ann = wfdb.rdann(data_path + '/' + file, 'atr')
    if os.path.exists(data_path + '/' + file + '.qrsc'):
        qrs = wfdb.rdann(data_path + '/' + file, 'qrsc')
    else:
        qrs = wfdb.rdann(data_path + '/' + file, 'qrs')

    fs = ann.fs

    sample = ann.sample
    aux = ann.aux_note

    r_peak = qrs.sample
    assert (np.sort(r_peak) == r_peak).all()

    label = np.zeros_like(r_peak)
    if len(list(set(aux))) >= 1:
        start = 0
        while start < len(aux):
            if aux[start] == '(AFIB':
                end = start + 1
                while end < len(aux) and aux[end] == '(AFIB':
                    end += 1
                if end < len(sample):
                    if (sample[end] - sample[start]) / fs < 30:
                        label[(r_peak >= sample[start]) & (r_peak < sample[end])] = 3
                    elif (sample[end] - sample[start]) / fs < 60:
                        label[(r_peak >= sample[start]) & (r_peak < sample[end])] = 2
                    else:
                        label[(r_peak >= sample[start]) & (r_peak < sample[end])] = 1
                else:
                    if (r_peak[-1]-sample[start]) / fs < 30:
                        label[r_peak >= sample[start]] = 3
                    elif (r_peak[-1]-sample[start]) / fs < 60:
                        label[r_peak >= sample[start]] = 2
                    else:
                        label[r_peak >= sample[start]] = 1
                start = end
            else:
                start += 1

    rr_seq = np.diff(r_peak) / fs
    label = np.asarray(label[:-1])
    assert rr_seq.shape[0] == label.shape[0]
    label[(rr_seq < 200 / 1000) | (rr_seq > 2000 / 1000)] = -1

    return rr_seq, label


def load_data_nsr(data_path, file):
    data = wfdb.rdrecord(data_path + '/' + file)
    ann = wfdb.rdann(data_path + '/' + file, 'atr')
    fs = data.fs
    r_peak = ann.sample
    label = np.zeros(shape=(len(r_peak),))

    rr_seq = np.diff(r_peak) / fs
    label = np.asarray(label[:-1])
    assert rr_seq.shape[0] == label.shape[0]
    label[(rr_seq < 200 / 1000) | (rr_seq > 2000 / 1000)] = -1

    return rr_seq, label


def load_data_nsrrri(data_path, file):
    data = wfdb.rdann(data_path + '/' + file, 'ecg')
    fs = data.fs
    r_peak = data.sample
    label = np.zeros(shape=(len(r_peak), ))

    rr_seq = np.diff(r_peak) / fs * 1000 / 1000
    label = np.asarray(label[:-1])
    assert rr_seq.shape[0] == label.shape[0]
    label[(rr_seq < 200 / 1000) | (rr_seq > 2000 / 1000)] = -1

    return rr_seq, label
