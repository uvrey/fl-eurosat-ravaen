import dill
from pathlib import Path

from typing import List, Dict

from torch.utils.data import DataLoader, ConcatDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataset import Dataset
import json

from utils import load_obj
# from utils import filter_pathlist, deep_compare

###############


def filter_pathlist(path_list, expr):
    if expr == 'all':
        return path_list

    elif expr[:2] == 'I:':
        fl = eval(f'path_list[{expr[2:]}]')
        if type(fl) == str:
            return [fl]
        return fl

    elif expr[:2] == 'R:':
        regexp = re.compile(expr[2:])
        return [e for e in path_list if regexp.match(e.name)]

    elif isinstance(expr, list):
        not_included = set(path_list) - set(expr)
        assert not not_included, \
                f'Some experiences could not be found: {not_included}'

        return expr

    raise NotImplementedError


def deep_compare(sample_shape1, sample_shape2):
    assert type(sample_shape1) == type(sample_shape2), \
        f'{type(sample_shape1)} != {type(sample_shape2)}'
           

    if sample_shape1 is None or sample_shape2 is None:
        return None

    if isinstance(sample_shape1, Iterable):
        assert len(sample_shape1) == len(sample_shape2)
        return tuple([deep_compare(s1, s2) for s1, s2 in zip(sample_shape1, sample_shape2)])

    return sample_shape1 if sample_shape2 == sample_shape1 else None

################

class ParsedDataModule(LightningDataModule):
    """
    Main datamodule class. Given a configuration dict, it will create
    dynamically the datasets and dataloaders.

    The datamodule includes a caching mechanism to save in memory (pickle) an
    instance and load it in another script. This is helpful for its usage if
    its creation is a long process or to use it from jupyter notebooks.
    """

    def __init__(self, cfg: Dict):
        super().__init__()

        # Here so that I can cache it
        self.train_ds = self.create_dataset(self.root_folder, self.train,
                                            self.train.get('keep_separate',
                                                           False))

        self.val_ds = self.create_dataset(self.root_folder, self.valid,
                                          self.valid.get('keep_separate',
                                                         False))

        self.test_ds = self.create_dataset(self.root_folder, self.test,
                                           self.test.get('keep_separate',
                                                         False))

    def create_dataset(self, root_folder, dataset_cfg, keep_separate):
        ## INFO WE CHOOSE
        folder_filter = "I:0:-1"
        dataset_cls = "dataset.LocationDataset"
        normaliser_args = {
            "normaliser_specs": {
                "B1": {
                    "normaliser_cls": 'normalisers.ListNormaliser',
                    "normaliser_args": {
                        "normaliser_specs": [
                            {'normalisers.LogNormaliser': {}},
                            {
                                'normalisers.RescaleNormaliser': {
                                    "x0": 7.3,
                                    "x1": 7.6,
                                    "y0": -1,
                                    "y1": 1
                                }
                            }
                        ]
                    }
                },
                "B2": {
                    "normaliser_cls": 'normalisers.ListNormaliser',
                    "normaliser_args": {
                        "normaliser_specs": [
                            {'normalisers.LogNormaliser': {}},
                            {
                                'normalisers.RescaleNormaliser': {
                                    "x0": 6.9,
                                    "x1": 7.5,
                                    "y0": -1,
                                    "y1": 1
                                }
                            }
                        ]
                    }
                },
                "B3": {
                    "normaliser_cls": 'normalisers.ListNormaliser',
                    "normaliser_args": {
                        "normaliser_specs": [
                            {'normalisers.LogNormaliser': {}},
                            {
                                'normalisers.RescaleNormaliser': {
                                    "x0": 6.5,
                                    "x1": 7.4,
                                    "y0": -1,
                                    "y1": 1
                                }
                            }
                        ]
                    }
                },
                "B4": {
                    "normaliser_cls": 'normalisers.ListNormaliser',
                    "normaliser_args": {
                        "normaliser_specs": [
                            {'normalisers.LogNormaliser': {}},
                            {
                                'normalisers.RescaleNormaliser': {
                                    "x0": 6.2,
                                    "x1": 7.5,
                                    "y0": -1,
                                    "y1": 1
                                }
                            }
                        ]
                    }
                },
                "B5": {
                    "normaliser_cls": 'normalisers.ListNormaliser',
                    "normaliser_args": {
                        "normaliser_specs": [
                            {'src.data.normalisers.LogNormaliser': {}},
                            {
                                'src.data.normalisers.RescaleNormaliser': {
                                    "x0": 6.1,
                                    "x1": 7.5,
                                    "y0": -1,
                                    "y1": 1
                                }
                            }
                        ]
                    }
                },
                "B6": {
                    "normaliser_cls": 'normalisers.ListNormaliser',
                    "normaliser_args": {
                        "normaliser_specs": [
                            {'normalisers.LogNormaliser': {}},
                            {
                                'normalisers.RescaleNormaliser': {
                                    "x0": 6.5,
                                    "x1": 8,
                                    "y0": -1,
                                    "y1": 1
                                }
                            }
                        ]
                    }
                },
                "B7": {
                    "normaliser_cls": 'normalisers.ListNormaliser',
                    "normaliser_args": {
                        "normaliser_specs": [
                            {'normalisers.LogNormaliser': {}},
                            {
                                'normalisers.RescaleNormaliser': {
                                    "x0": 6.5,
                                    "x1": 8,
                                    "y0": -1,
                                    "y1": 1
                                }
                            }
                        ]
                    }
                },
                "B8": {
                    "normaliser_cls": 'normalisers.ListNormaliser',
                    "normaliser_args": {
                        "normaliser_specs": [
                            {'normalisers.LogNormaliser': {}},
                            {
                                'normalisers.RescaleNormaliser': {
                                    "x0": 6.5,
                                    "x1": 8,
                                    "y0": -1,
                                    "y1": 1
                                }
                            }
                        ]
                    }
                },
                "B8A": {
                    "normaliser_cls": 'normalisers.ListNormaliser',
                    "normaliser_args": {
                        "normaliser_specs": [
                            {'normalisers.LogNormaliser': {}},
                            {
                                'normalisers.RescaleNormaliser': {
                                    "x0": 6.5,
                                    "x1": 8,
                                    "y0": -1,
                                    "y1": 1
                                }
                            }
                        ]
                    }
                },
                "B9": {
                    "normaliser_cls": 'normalisers.ListNormaliser',
                    "normaliser_args": {
                        "normaliser_specs": [
                            {'normalisers.LogNormaliser': {}},
                            {
                                'normalisers.RescaleNormaliser': {
                                    "x0": 6,
                                    "x1": 7,
                                    "y0": -1,
                                    "y1": 1
                                }
                            }
                        ]
                    }
                },
                "B10": {
                    "normaliser_cls": 'normalisers.ListNormaliser',
                    "normaliser_args": {
                        "normaliser_specs": [
                            {'normalisers.LogNormaliser': {}},
                            {
                                'normalisers.RescaleNormaliser': {
                                    "x0": 2.5,
                                    "x1": 4.5,
                                    "y0": -1,
                                    "y1": 1
                                }
                            }
                        ]
                    }
                },
                "B11": {
                    "normaliser_cls": 'normalisers.ListNormaliser',
                    "normaliser_args": {
                        "normaliser_specs": [
                            {'normalisers.LogNormaliser': {}},
                            {
                                'normalisers.RescaleNormaliser': {
                                    "x0": 6,
                                    "x1": 8,
                                    "y0": -1,
                                    "y1": 1
                                }
                            }
                        ]
                    }
                },
                "B12": {
                    "normaliser_cls": 'normalisers.ListNormaliser',
                    "normaliser_args": {
                        "normaliser_specs": [
                            {'normalisers.LogNormaliser': {}},
                            {
                                'normalisers.RescaleNormaliser': {
                                    "x0": 6,
                                    "x1": 8,
                                    "y0": -1,
                                    "y1": 1
                                }
                            }
                        ]
                    }
                }
            }
        }

        dataset_cls_args = {
            "tiling_strategy_cls": 'tiling_strategy.TilingStrategyFullGrid',
            "tiling_strategy_args": {
                "window_size": "${dataset.window_size}",
                "overlap": [0, 0]
            },
            "dataset_specs": [
                {
                    "dataset.SingleFolderImageDataset": {
                        "folder_path": 'S2',
                        "filter_cls": 'filters.SliceFilter',
                        "filter_args": {
                            "f_slice": ':'
                        },
                        "normaliser_cls": "normalisers.ListNormaliser",
                        "normaliser_args": normaliser_args,
                        "channels": ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
                    }
                }
            ]
        }

        _folder_paths = Path(root_folder).glob('*')
        _folder_paths = [f for f in _folder_paths if f.is_dir()]
        _folder_paths.sort()

        folder_paths = []
        for filt in folder_filter:
            folder_paths += filter_pathlist(_folder_paths, filt)

        datasets = []
        for f in folder_paths:
            ds_cls = load_obj(dataset_cfg['dataset_cls'])
            ds = ds_cls(f, **dataset_cls_args)
            if len(ds) > 0:
                datasets += [ds]

        if keep_separate:
            return datasets
        return MyConcatDataset(datasets)

    @staticmethod
    def _len(ds):
        if isinstance(ds, list):
            #print("This dataset is a list")
            length = 0
            for d in ds:
                length += len(d)
        else:
            #print("This dataset is NOT a list. It's a sequence of LocationDataset, length of datasets[0] is:")
            #print(len(ds.datasets))
            length = len(ds)
        return length

    @property
    def len_train_ds(self):
        return self._len(self.train_ds)

    @property
    def len_val_ds(self):
        return self._len(self.val_ds)

    @property
    def len_test_ds(self):
        return self._len(self.test_ds)

    def _sample_shape(self, ds):
        if isinstance(ds, list):
            ds = self.train_ds[0]
        return ds.sample_shape

    @property
    def sample_shape_train_ds(self):
        return self._sample_shape(self.train_ds)

    @property
    def sample_shape_val_ds(self):
        return self._sample_shape(self.val_ds)

    @property
    def sample_shape_test_ds(self):
        return self._sample_shape(self.test_ds)

    def set_batch_sizes_and_n_workers(self,
                                      train_batch_size=None,
                                      train_num_workers=None,
                                      val_batch_size=None,
                                      val_num_workers=None,
                                      test_batch_size=None,
                                      test_num_workers=None):

        if train_batch_size is not None:
            self.train['batch_size'] = train_batch_size
        if train_num_workers is not None:
            self.train['num_workers'] = train_num_workers

        if val_batch_size is not None:
            self.valid['batch_size'] = val_batch_size
        if val_num_workers is not None:
            self.valid['num_workers'] = val_num_workers

        if test_batch_size is not None:
            self.test['batch_size'] = test_batch_size
        if test_num_workers is not None:
            self.test['num_workers'] = test_num_workers

    @staticmethod
    def _make_dataloader(dataset, config):
        if config.get('keep_separate', False): # used to be not if not config.get('keep_separate', False):
            return DataLoader(
                dataset,
                batch_size=config['batch_size'],
                num_workers=config['num_workers'],
                shuffle=config.get('shuffle', False))
        else:
            return [
                DataLoader(ds,
                           batch_size=config['batch_size'],
                           num_workers=config['num_workers'],
                           shuffle=config.get('shuffle', False))
                    for ds in dataset]

    def train_dataloader(self):
        return self._make_dataloader(self.train_ds, self.train)

    def val_dataloader(self):
        return self._make_dataloader(self.val_ds, self.valid)

    def test_dataloader(self):
        return self._make_dataloader(self.test_ds, self.test)

    @staticmethod
    def load(dataset_name, cache_dir):
        savepath = Path(cache_dir) / 'datamodules' / dataset_name

        print("SAVING TO: " + str(savepath))

        if savepath.exists():
            with open(savepath, 'r') as file:
                loaded_data = json.load(file)
                print(loaded_data)
                return loaded_data

        return None

    def save(self, cache_dir, overwrite=False):
        savepath = Path(cache_dir) / 'datamodules'
        savepath.mkdir(exist_ok=True, parents=True)

        savepath /= self.data_module_name
        if not savepath.exists() or (savepath.exists() and overwrite):
            with open(savepath, 'wb') as file:
                dill.dump(self, file)
        else:
            print(f'Datamodule {self.data_module_name} already in cache, not saving')

    @staticmethod
    def load_or_create(cfg, cache_dir, prompt_for_input=True):

        # print(f"Prior to creating, we have module name and cache dict {cache_dir}")
        datamodule = ParsedDataModule.load(cfg['data_module_name'], cache_dir)
        print("SUCCESS")
        
        # # NEVER ASK FOR PROMPT, it caused too many mixups already... always recreate, save for logs sake

        if datamodule is None:
            datamodule = ParsedDataModule(cfg)
            datamodule.save(cache_dir)
        else:
            datamodule = ParsedDataModule(cfg)
            datamodule.save(cache_dir, overwrite=True)

        # """
        # if datamodule is not None and not prompt_for_input:
        #     pass
        # elif datamodule is not None and prompt_for_input:
        #     ans = ''
        #     while not (ans.lower() == 'y' or ans.lower() == 'n'):
        #         ans = input(f'Datamodule {cfg["data_module_name"]} found in cache. Do you want to use it? [y/n] ')

        #     if ans.lower() == 'n':
        #         print("Creating and saving new datamodule")
        #         datamodule = ParsedDataModule(cfg)
        #         datamodule.save(cache_dir, overwrite=True)
        # else:
        #     datamodule = ParsedDataModule(cfg)
        #     datamodule.save(cache_dir)

        # """

        # datamodule.set_batch_sizes_and_n_workers(
        #     cfg['train']['batch_size'],
        #     cfg['train']['num_workers'],
        #     cfg['valid']['batch_size'],
        #     cfg['valid']['num_workers'],
        #     cfg['test']['batch_size'],
        #     cfg['test']['num_workers']
        # )

        return datamodule


class MyConcatDataset(ConcatDataset):
    """
    ConcatDataset that retains the shape of the inputs.
    """

    def __init__(self, datasets: List[Dataset]):
        super().__init__(datasets)

        self.sample_shape = self.get_sample_shape(datasets)

    @staticmethod
    def get_sample_shape(datasets):
        sample_shape = datasets[0].sample_shape
        for d in datasets:
            sample_shape.merge(d.sample_shape)
        return sample_shape
