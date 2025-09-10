import argparse
import json
import sys
from datetime import datetime
from argparse import Namespace
import networkx as nx
from pybbn.graph.dag import Bbn
from pybbn.generator.bbngenerator import (
    convert_for_exact_inference,
    generate_multi_bbn,
    generate_singly_bbn,
)

def parse_args(args: list[str]) -> Namespace:
    """
    Parse command line arguments.
    
    Args:
        args: List of command line arguments (typically sys.argv[1:])
        
    Returns:
        Namespace: Parsed arguments object
    """
    parser = argparse.ArgumentParser(description="BBN Generator")
    
    parser.add_argument(
        "--type", "-t",
        choices=["m", "s"],
        default="s",
        help="Type parameter: 'm' or 's'"
    )

    parser.add_argument(
        "--nodes", "-n",
        type=int,
        default=2,
        help="Number of nodes (default: 2)"
    )
    
    parser.add_argument(
        "--iters", "-i",
        type=int,
        default=10,
        help="Number of iterations (default: 10)"
    )
    
    parser.add_argument(
        "--values", "-v",
        type=int,
        default=2,
        help="Values parameter (default: 2)"
    )
    
    parser.add_argument(
        "--alpha", "-a",
        type=int,
        default=10,
        help="Alpha parameter (default: 10)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="",
        help="Output JSON path"
    )
    
    return parser.parse_args(args)


def validate(G: nx.DiGraph):
    n = G.number_of_nodes()
    e = G.number_of_edges()

    if n == 0:
        return False, {"reason": "empty graph", "nodes": n, "edges": e}

    # Connected if we ignore direction
    is_one_component = nx.is_weakly_connected(G)

    # Weak components (ignoring direction)
    comps = list(nx.weakly_connected_components(G))
    num_components = len(comps)

    # “Orphan” nodes = components of size 1 when there are other nodes present
    orphan_nodes = set()
    if n > 1:
        orphan_nodes = {next(iter(c)) for c in comps if len(c) == 1}

    ok = is_one_component and len(orphan_nodes) == 0

    return {
        "ok": ok,
        "nodes": n,
        "edges": e,
        "is_weakly_connected": is_one_component,
        "num_components": num_components,
        "orphan_nodes": len(orphan_nodes),
    }

def generate_bbn_json(args: Namespace) -> str:
    params = {
        "n": args.nodes,
        "max_iter": args.iters,
        "max_values": args.values,
        "max_alpha": args.alpha
    }
    
    if args.type == "s":
        g, p = generate_singly_bbn(**params)
    else:
        g, p = generate_multi_bbn(**params)

    validation = validate(g)
    print(json.dumps(validation, indent=2))

    bbn = convert_for_exact_inference(g, p)
    s = json.dumps(Bbn.to_dict(bbn), indent=2)
    return s

def get_output_path(args: Namespace) -> str:
    _type = "multi" if args.type == "m" else "singly"
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = f"{_type}_{_ts}.json"
    output_path = args.output if args.output and len(args.output) > 0 else output_path
    return output_path


def main():
    """Main application entry point."""
    args = parse_args(sys.argv[1:])
    json_str = generate_bbn_json(args)
    output_path = get_output_path(args)

    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write(json_str)
    print(f"BBN JSON written to {output_path}")


if __name__ == "__main__":
    main()