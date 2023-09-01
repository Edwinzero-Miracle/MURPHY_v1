
def get_training_mode_split(config_name: str, split_id: int):
    if config_name == 'mix':
        assert (0 < split_id < 9)
    else:
        raise NotImplemented("This is not a valid difficulty level name")

    split_dict = {
        'mix': {
            1: {'train': ['B00', 'B01', 'B02', 'B05', 'B09', 'B10', 'B15', 'C00', 'C06', 'C07', 'C08', 'C11', 'C12',
                          'B04', 'B06', 'B08', 'B12', 'B14', 'C01', 'C02', 'C03', 'C04',
                          'B03', 'B07', 'B11', 'B13', 'C05', 'C09', 'C10'],
                'test': ['A00', 'A01', 'A02', 'A03', 'A04']},
            2: {'train': ['B00', 'B01', 'B02', 'B05', 'B09', 'B10', 'B15', 'C00', 'C06', 'C07', 'C08', 'C11', 'C12',
                          'B04', 'B06', 'B08', 'B12', 'B14', 'C01', 'C02', 'C03', 'C04',
                          'B03', 'B07', 'B11', 'B13', 'C05', 'C09', 'C10'],
                'test': ['D01', 'D00', 'D02', 'D04', 'D05', 'D03', 'D06']},
            3: {'train': ['B00', 'B01', 'B02', 'B05', 'B09', 'B10', 'B15', 'C00', 'C06', 'C07', 'C08', 'C11', 'C12',
                          'B04', 'B06', 'B08', 'B12', 'B14', 'C01', 'C02', 'C03', 'C04',
                          'B03', 'B07', 'B11', 'B13', 'C05', 'C09', 'C10'],
                'test': ['E00', 'E04', 'E01', 'E02', 'E03']},
            5: {'train': ['LC01', 'LC02', 'LC03'],
                'test': ['LC00']},
            6: {'train': ['LC00', 'LC02', 'LC03'],
                'test': ['LC01']},
            7: {'train': ['LC00', 'LC01', 'LC03'],
                'test': ['LC02']},
            8: {'train': ['LC00', 'LC01', 'LC02'],
                'test': ['LC03']},
        }
    }
    return split_dict[config_name][split_id]
