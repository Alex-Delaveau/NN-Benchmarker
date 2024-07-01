from tqdm import tqdm

def get_input_image_names(list_names, directory_name, if_train=True):
    list_img = []
    list_msk = []
    list_test_ids = []

    for filenames in tqdm(list_names['name'], miniters=1000):
        nred = 'red_' + filenames
        nblue = 'blue_' + filenames
        ngreen = 'green_' + filenames
        nnir = 'nir_' + filenames

        if if_train:
            dir_type_name = "train"
            fl_img = []
            nmask = 'gt_' + filenames
            fl_msk = f"{directory_name}/train_gt/{nmask}.TIF"
            list_msk.append(fl_msk)
        else:
            dir_type_name = "test"
            fl_img = []
            fl_id = f"{filenames}.TIF"
            list_test_ids.append(fl_id)

        fl_img_red = f"{directory_name}/{dir_type_name}_red/{nred}.TIF"
        fl_img_green = f"{directory_name}/{dir_type_name}_green/{ngreen}.TIF"
        fl_img_blue = f"{directory_name}/{dir_type_name}_blue/{nblue}.TIF"
        fl_img_nir = f"{directory_name}/{dir_type_name}_nir/{nnir}.TIF"
        fl_img.extend([fl_img_red, fl_img_green, fl_img_blue, fl_img_nir])

        list_img.append(fl_img)

    if if_train:
        return list_img, list_msk
    else:
        return list_img, list_test_ids