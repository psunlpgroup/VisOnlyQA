import random
from pathlib import Path
import pickle as pkl
import signal
import hashlib

import numpy as np
from tap import Tap

# from ../alphageometry
import problem as pr
from utils.loading_utils import load_definitions_and_rules

from src.path import alphageometry_intermediate_dir
import src.training_dataset.alphageometry.src.graph as gh
from src.training_dataset.alphageometry.src.clause_generation import CompoundClauseGen


alphageometry_intermediate_image_dir = alphageometry_intermediate_dir / "images"
alphageometry_questions_dir = alphageometry_intermediate_dir / "questions"
alphageometry_graph_dir = alphageometry_intermediate_dir / "graphs"


class TimeoutException(Exception):
    """Custom exception to indicate a timeout."""
    pass


def generate_and_save_geometric_shape(seed: int, definitions):
    hashed_seed = int(hashlib.sha256(str(seed).encode("utf-8")).hexdigest(), 16)
    
    cc_gen = CompoundClauseGen(definitions, 2, 3, 2, seed=hashed_seed)
    txt = cc_gen.generate_clauses()
    p = pr.Problem.from_txt(txt)
    
    g, _ = gh.Graph.build_problem(p, definitions)
    
    # Additionaly draw this generated problem
    gh.nm.draw(
        g.type2nodes[gh.Point],
        g.type2nodes[gh.Line],
        g.type2nodes[gh.Circle],
        g.type2nodes[gh.Segment],
        output_figure_path=alphageometry_intermediate_image_dir / f'alphageometry_{seed:05d}.png'
    )
    
    with open(alphageometry_questions_dir / f'alphageometry_{seed:05d}.txt', 'w') as f:
        f.write(txt)
    
    with open(alphageometry_graph_dir / f'alphageometry_{seed:05d}.pkl', 'wb') as f:
        pkl.dump(g, f)


class AlphaGeometryTap(Tap):
    seed: int
    timeout: int = 3


if __name__ == "__main__":
    args = AlphaGeometryTap().parse_args()
    
    alphageometry_intermediate_dir.mkdir(parents=True, exist_ok=True)
    alphageometry_intermediate_image_dir.mkdir(parents=True, exist_ok=True)
    alphageometry_questions_dir.mkdir(parents=True, exist_ok=True)
    alphageometry_graph_dir.mkdir(parents=True, exist_ok=True)

    alphageometry_src_dir = Path("src/training_dataset/alphageometry")
    defs_path = alphageometry_src_dir / 'defs.txt'
    rules_path = alphageometry_src_dir / 'rules.txt'
    
    # Load definitions and rules
    definitions, rules = load_definitions_and_rules(defs_path, rules_path)
    
    print(f'Problem created, Building graph ...')
    try:
        # Set an alarm for 10 seconds
        signal.alarm(3)

        random.seed(args.seed)
        np.random.seed(args.seed)

        # Code block to execute with timeout
        generate_and_save_geometric_shape(args.seed, definitions)

        # Disable the alarm
        signal.alarm(0)
    except TimeoutException as e:
        print("Graph couldn't bre create in reasonable time. Perhaps problem with the premises. Exiting ...")
        raise e
