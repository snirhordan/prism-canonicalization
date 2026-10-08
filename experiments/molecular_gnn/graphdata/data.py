"""Dataset loading, dispatched on the dataset name of the config."""

OGB_NAMES = ('OGBG-MOLTOX21', 'OGBG-MOLPCBA')


def LoadData(DATASET_NAME, data_root='data'):
    if DATASET_NAME == 'ZINC':
        from graphdata.molecules import MoleculeDataset
        return MoleculeDataset(DATASET_NAME, data_root)
    if DATASET_NAME in OGB_NAMES:
        from graphdata.ogb_mol import OGBMOLDataset
        return OGBMOLDataset(DATASET_NAME, data_root)
    raise ValueError("unknown dataset %r" % DATASET_NAME)
