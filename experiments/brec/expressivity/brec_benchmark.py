"""BREC expressivity benchmark with canonicalized Laplacian positional encodings.

For every BREC pair a freshly initialised model (GINE backbone with a DeepSets
eigenvector embedding, or the graph transformer) is trained with a cosine
embedding loss to separate the two graphs of the pair under 32 random
relabellings; the pair counts as distinguished when the Hotelling T^2 statistic
of the embedding differences passes the BREC reliability test.  The evaluation
protocol follows the BREC benchmark (https://github.com/GraphPKU/BREC, MIT
License, Copyright (c) 2023 Yanbo Wang) and the example of "Aligning
Transformers with Weisfeiler-Leman" (https://github.com/luis-mueller/wl-transformers,
MIT License, Copyright (c) 2024 Luis Mueller).

The Laplacian eigenvectors are canonicalized with the ``prism`` package (PRiSM,
MAP, OAP); ``pe_type`` selects the method, see ``wl_transformers.pe_types``.
"""

import time

import hydra
import torch
import torch_geometric
import torch_geometric.loader
from brec.dataset import BRECDataset
from loguru import logger
from torch.nn import CosineEmbeddingLoss
from tqdm import tqdm

from wl_transformers import (
    RunLogger,
    Transform,
    count_parameters,
    create_gnn,
    create_transformer,
    ensure_data_dir,
    log_config,
    make_canonicalizer,
)

NUM_RELABEL = 32
EPSILON_CMP = 1e-6
SAMPLE_NUM = 400
MARGIN = 0.0
LEARNING_RATE = 1e-4
THRESHOLD = 72.34
WEIGHT_DECAY = 1e-4
LOSS_THRESHOLD = 0.2

PART_DICT = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
}


def get_model(cfg, device):
    atom_encoder = torch.nn.Linear(1, cfg.embed_dim)
    bond_encoder = torch.nn.Embedding(2, cfg.embed_dim)

    if cfg.backbone == "transformer":
        model = create_transformer(
            cfg, atom_encoder, bond_encoder, output_dim=cfg.output_dim, device=device
        )
    elif cfg.backbone == "gnn":
        model = create_gnn(
            cfg, atom_encoder, bond_encoder, output_dim=cfg.output_dim, device=device
        )
    else:
        raise ValueError(f"Backbone {cfg.backbone} is not supported")
    return model


def evaluate(dataset, cfg, device):
    batch_size = cfg.batch_size

    def T2_calculation(dataset, model, log_flag=False):
        with torch.no_grad():
            loader = torch_geometric.loader.DataLoader(dataset, batch_size=batch_size)
            pred_0_list = []
            pred_1_list = []
            for data in loader:
                pred = model(data.to(device)).detach()
                pred_0_list.extend(pred[0::2])
                pred_1_list.extend(pred[1::2])
            X = torch.cat([x.reshape(1, -1) for x in pred_0_list], dim=0).T
            Y = torch.cat([x.reshape(1, -1) for x in pred_1_list], dim=0).T
            if log_flag:
                logger.info(f"X_mean = {torch.mean(X, dim=1)}")
                logger.info(f"Y_mean = {torch.mean(Y, dim=1)}")
            D = X - Y
            D_mean = torch.mean(D, dim=1).reshape(-1, 1)
            S = torch.cov(D)
            inv_S = torch.linalg.pinv(S)
            return torch.mm(torch.mm(D_mean.T, inv_S), D_mean)

    run_logger = RunLogger(cfg, config_dict=dict(cfg))
    parts = list(cfg.parts)
    for part in parts:
        if part not in PART_DICT:
            raise ValueError(f"unknown part {part!r}; available: {list(PART_DICT)}")

    time_start = time.process_time()

    cnt = 0
    num_pairs = 0
    correct_list = []
    fail_in_reliability = 0
    loss_func = CosineEmbeddingLoss(margin=MARGIN)
    per_part = {}
    logged_parameters = False

    for part_name in parts:
        part_range = PART_DICT[part_name]
        ids = range(part_range[0], part_range[1])
        if cfg.max_pairs_per_part is not None:
            ids = ids[: cfg.max_pairs_per_part]
        logger.info(f"{part_name} part starting ---")

        cnt_part = 0
        fail_in_reliability_part = 0
        start = time.process_time()

        for id in tqdm(ids):
            logger.info(f"ID: {id}")
            model = get_model(cfg, device)
            if not logged_parameters:
                logger.info(f"Model parameters: {count_parameters(model)}")
                logged_parameters = True
            optimizer = torch.optim.Adam(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            dataset_traintest = dataset[
                id * NUM_RELABEL * 2 : (id + 1) * NUM_RELABEL * 2
            ]
            dataset_reliability = dataset[
                (id + SAMPLE_NUM) * NUM_RELABEL * 2 : (id + SAMPLE_NUM + 1) * NUM_RELABEL * 2
            ]
            model.train()
            for _ in range(cfg.epochs):
                traintest_loader = torch_geometric.loader.DataLoader(
                    dataset_traintest, batch_size=batch_size
                )
                loss_all = 0
                for data in traintest_loader:
                    optimizer.zero_grad()
                    pred = model(data.to(device))
                    loss = loss_func(
                        pred[0::2],
                        pred[1::2],
                        torch.tensor([-1] * (len(pred) // 2)).to(device),
                    )
                    loss.backward()
                    optimizer.step()
                    loss_all += len(pred) / 2 * loss.item()
                loss_all /= NUM_RELABEL
                if loss_all < LOSS_THRESHOLD:
                    break
                scheduler.step(loss_all)

            model.eval()
            T_square_traintest = T2_calculation(dataset_traintest, model, True)
            T_square_reliability = T2_calculation(dataset_reliability, model, True)

            isomorphic_flag = False
            reliability_flag = False
            if T_square_traintest > THRESHOLD and not torch.isclose(
                T_square_traintest, T_square_reliability, atol=EPSILON_CMP
            ):
                isomorphic_flag = True
            if T_square_reliability < THRESHOLD:
                reliability_flag = True

            if isomorphic_flag:
                cnt += 1
                cnt_part += 1
                correct_list.append(id)
                logger.info(f"Correct num in current part: {cnt_part}")
            if not reliability_flag:
                fail_in_reliability += 1
                fail_in_reliability_part += 1
            logger.info(f"isomorphic: {isomorphic_flag} {T_square_traintest}")
            logger.info(f"reliability: {reliability_flag} {T_square_reliability}")

            run_logger.log({f"{part_name}/correct": cnt_part, "total_correct": cnt})

        num_pairs += len(ids)
        per_part[part_name] = cnt_part
        time_cost_part = round(time.process_time() - start, 2)
        logger.info(
            f"{part_name} part costs time {time_cost_part}; Correct in {cnt_part} / {len(ids)}"
        )
        logger.info(f"Fail in reliability: {fail_in_reliability_part} / {len(ids)}")

    time_cost = round(time.process_time() - time_start, 2)
    logger.info(f"evaluation time cost: {time_cost}")
    logger.info(f"Per part: {per_part}")
    logger.info(f"Correct in {cnt} / {num_pairs}, Acc = {round(cnt / max(num_pairs, 1), 2)}")
    logger.info(f"Fail in reliability: {fail_in_reliability} / {num_pairs}")
    logger.info(correct_list)
    run_logger.finish()


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    data_dir = ensure_data_dir(cfg.root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    if cfg.smoke:
        cfg.parts = ["Basic"]
        cfg.max_pairs_per_part = 2
        cfg.epochs = 2
    if cfg.seed is not None:
        torch_geometric.seed_everything(cfg.seed)

    refine = bool(cfg.refine_ordering)
    canonicalizer = make_canonicalizer(cfg.pe_type, refine, cfg.prism_preset)
    log_config(cfg, pe_type=cfg.pe_type, canonicalizer=canonicalizer, refine_ordering=refine)
    transform = Transform(
        cfg.num_vecs, cfg.order, cfg.normalized_laplacian, cfg.normalize_eigenvecs,
        canonicalizer=canonicalizer,
    )

    def transform_func(data):
        data.edge_attr = torch.ones(data.edge_index.size(1), dtype=torch.long)
        return transform(data)

    dataset = BRECDataset(root=data_dir, transform=transform_func)
    evaluate(dataset, cfg, device)


if __name__ == "__main__":
    main()
