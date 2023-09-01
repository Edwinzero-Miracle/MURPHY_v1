import os
import numpy

data_path = ""
video_folder = 'Videos'
img_folder = 'Images'
raw_label_folder = 'RawLabels'
label_folder = 'Labels'
processed_folder = 'Processed'


def get_img_filename(subject, trial, frame):
    return subject + '_' + f'{trial:02d}' + '_' + str(frame) + '.jpg'


def get_video_filename(subject, trial, extension='mp4'):
    return subject + f'{trial:02d}' + extension


def get_raw_label_filename(subject, trial):
    return subject + f'{trial:02d}' + '.txt'


def get_label_filename(subject, trial):
    return subject + f'{trial:02d}' + '.txt'


def get_processed_pkl_filename(subject, trial, fps):
    return subject + f'{trial:02d}' + '_fps' + str(fps) + '.pkl'


def get_processed_json_filename(subject, trial, fps):
    return subject + f'{trial:02d}' + '_fps' + str(fps) + '.json'


def get_img_path():
    return os.path.join(data_path, img_folder)


def get_video_path():
    return os.path.join(data_path, video_folder)


def get_raw_label_path():
    return os.path.join(data_path, raw_label_folder)


def get_label_path():
    return os.path.join(data_path, label_folder)


def get_processed_path():
    return os.path.join(data_path, processed_folder)


def get_img(subject, trial, frame):
    return os.path.join(get_img_path(), get_img_filename(subject, trial, frame))


def get_video(subject, trial, extension='mp4'):
    return os.path.join(get_video_path(), get_video_filename(subject, trial, extension))


def get_raw_label_txt(subject, trial):
    return os.path.join(get_raw_label_path(), get_raw_label_filename(subject, trial))


def get_label_txt(subject, trial):
    return os.path.join(get_label_path(), get_label_filename(subject, trial))


def get_data_pkl(subject, trial, fps):
    return os.path.join(get_processed_path(), get_processed_pkl_filename(subject, trial, fps))


def get_data_json(subject, trial, fps):
    return os.path.join(get_processed_path(), get_processed_json_filename(subject, trial, fps))





