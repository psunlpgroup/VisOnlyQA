import pickle as pkl
import random
from typing import TypedDict, Union, Literal
import json
from pathlib import Path
import hashlib
import itertools
import shutil
import copy

from tap import Tap
from tqdm import tqdm
import numpy as np
import PIL.Image

from src.typing import TRAIN_VAL_TEST, VisonlyQA_Instance
from src.path import (
    train_dataset_dir, train_intermediate_data_dir, val_dataset_dir, val_intermediate_data_dir, test_dataset_dir, test_intermediate_data_dir,
    dataset_stats_dir,
)
from src.prompts import shuffle_options
from src.training_dataset.dataset_creation.dataset_creation_utils import get_data_instance
from src.training_dataset.dataset_creation.create_geometry3k import get_relative_lengths_of_all_lines
from src.training_dataset.dataset_creation.dataset_creation_configs import (
    triangle_prompt_template, quadrilateral_prompt_template, diameter_prompt_template, length_prompt_template, angle_prompt_template, area_prompt_template
)

# alphageometry
import geometry as gm
import src.training_dataset.alphageometry.src.graph as gh
from src.training_dataset.alphageometry.generate_random_geometric_shapes import alphageometry_graph_dir, alphageometry_intermediate_dir
from src.training_dataset.alphageometry.src.utils import save_alphageometry_figure


alphageometry_info_dir = alphageometry_intermediate_dir / "info"


class GeoInfo(TypedDict):
    point_positions: dict[str, tuple[float, float]]
    line_instances: list[list[str]]
    circle_instances: dict[str, tuple[tuple[float, float], float]]


def convert_graph_to_information(graph: gh.Graph, id: str) -> GeoInfo:
    # points
    point_positions = {}
    for point in graph.type2nodes[gh.Point]:
        point_positions[point.name.upper()] = (point.num.x, point.num.y)
        
        # print("point", point.name.upper())
        # print(point.__repr__())
    
    # lines
    line_instances = []
    for line in graph.type2nodes[gh.Line]:
        line_instances.append([p.name.upper() for p in line.neighbors(gm.Point)])
    
        # print("line", line.name.upper)
        # print(line.__repr__())

    # circles
    circle_instances = {}
    for circle in graph.type2nodes[gh.Circle]:
        # circle.num.neighbor
        
        circle_name = circle.name.upper()[0]
        
        circle_instances[circle_name] = (
            (circle.num.center.x, circle.num.center.y),
            circle.num.radius,
        )
        
        # print("circle", circle_name)
        # print(circle.__repr__())
    
    return {
        "id": id,
        "point_positions": point_positions,
        "line_instances": line_instances,
        "circle_instances": circle_instances,
    }


def convert_visual_information_to_text(information: GeoInfo) -> str:
    output: str = "The following text provides information about geometric shapes in the provided figure.\n\n"
    
    output += "Points:\n"
    for point_name, xy in information["point_positions"].items():
        x, y = xy
        output += f"  Point {point_name}: Position(x={x:.2f}, y={y:.2f})\n"
    
    output += "\nLine: "
    for line in information["line_instances"]:
        line_str = "".join(line)
        output += line_str + ", "
    output = output[:-2] + "\n"

    output += "\nCircles:\n"
    if len(information["circle_instances"]) == 0:
        output += "  No circle\n"
    else:
        for circle_name, xy_r in information["circle_instances"].items():
            xy, r = xy_r
            x, y = xy
            output += f"  Circle {circle_name}: CenterPosition(x={x:.2f}, y={y:.2f}), Radius={r:.2f}"
    
    output += "\nUsing the figure and above information, answer the following questions."
    
    return output


def add_text_visual_information_to_instance(original_instance: VisonlyQA_Instance, visual_information_text: str) -> VisonlyQA_Instance:
    updated_instance = copy.deepcopy(original_instance)
    
    for key in ["question", "prompt_reasoning", "prompt_no_reasoning"]:
        if original_instance[key] is not None:
            updated_instance[key] = visual_information_text + "\n\n" + original_instance[key]
    
    updated_instance["id"] = "text_" + original_instance["id"]
    
    return updated_instance


def are_points_on_the_same_line(points_list: list[str], line_instances: list[list[str]]) -> bool:
    for line in line_instances:
        on_line = [point in line for point in points_list]
        if all(on_line):
            return True
    
    return False


def is_valid_polygon(points_list: list[tuple[float, float]]) -> bool:
    """ Check whether there is no self-intersection in the polygon. Input is a list of (x, y) tuples of the vertices. """
    
    def ccw(A, B, C):
        """Check if three points are listed in counterclockwise order."""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


    def do_intersect(p1, p2, p3, p4):
        """Check if line segment p1p2 intersects with line segment p3p4."""
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    n = len(points_list)
    if n < 4:  # A polygon with less than 3 vertices can't intersect with itself
        return True

    # Check for intersection between every pair of edges that are not consecutive
    edges = [(points_list[i], points_list[(i + 1) % n]) for i in range(n)]
    for (p1, p2), (p3, p4) in itertools.combinations(edges, 2):
        # Avoid checking adjacent edges or the same edge (since they share a point)
        if p1 == p3 or p1 == p4 or p2 == p3 or p2 == p4:
            continue
        if do_intersect(p1, p2, p3, p4):
            return False  # Found a self-intersection

    return True  # No intersections found


def get_all_triangles_or_quadrilaterals_alpha_geometry(info: GeoInfo, shape_name: Literal["triangle", "quadrilateral"]) -> tuple[list[Union[tuple[str, str, str], tuple[str, str, str, str]]], list[Union[tuple[str, str, str], tuple[str, str, str, str]]]]:
    point_positions = sorted(list(info["point_positions"].items()))
    line_instances = info["line_instances"]
    
    target_points_num = 3 if shape_name == "triangle" else 4
    
    if len(point_positions) < target_points_num:  # not enough points
        return [], []
    
    valid_shapes: list[tuple[str, str, str]] = []
    invalid_shapes: list[tuple[str, str, str]] = []  # negative examples
    
    # permutation of target_points_num points
    for points in itertools.combinations(point_positions, target_points_num):
        points = sorted([point[0] for point in points])
        
        # if three points are on the same line, it is not a triangle or quadrilateral
        three_points_on_line = False
        for line in line_instances:
            on_line = [point in line for point in points]
            if sum(on_line) >= 3:
                invalid_shapes.append(points)
                three_points_on_line = True
                break
        if three_points_on_line:
            continue
        
        # check whether the points form a triangle or quadrilateral
        invalid_shape_added = False  # add only one invalid shape
        # we check all permutations of the points
        # this is redundant for triangles but necessary for quadrilaterals
        # it also shuffles the order of the points
        points_permutations = random.Random("".join(points)).sample(list(itertools.permutations(points)), len(points))
        for point_fixed_order in points_permutations:
            line_exists = [False for _ in range(target_points_num)]
            for idx in range(target_points_num):
                if are_points_on_the_same_line([point_fixed_order[idx], point_fixed_order[(idx + 1) % target_points_num]], line_instances):
                    line_exists[idx] = True
            
            if all(line_exists) and is_valid_polygon([info["point_positions"][p] for p in point_fixed_order]):  # triangle or quadrilateral
                valid_shapes.append(point_fixed_order)
                break
            elif sum(line_exists) >= target_points_num - 1:  # not triangle or quadrilateral but close
                if not invalid_shape_added:  # add only once
                    invalid_shapes.append(point_fixed_order)
                    invalid_shape_added = True
    
    return valid_shapes, invalid_shapes


def get_diameters_alpha_geometry(info: GeoInfo) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    line_instances = info["line_instances"]
    circle_instances = sorted(list(info["circle_instances"].keys()))
    
    valid_diameters: list[tuple[str, str]] = []
    invalid_diameters: list[tuple[str, str]] = []
    
    for circle in circle_instances:
        center, radius = info["circle_instances"][circle]
        
        # find diameters
        for line in line_instances:
            # line (list[str]) includes all points on the same line and can includes more than two points
            # we need to check all combinations of two points
            for point1, point2 in itertools.combinations(line, 2):
                # check if the two points are on the same circle
                points_on_circle: list[bool] = []
                for p in [point1, point2]:
                    if np.isclose(
                            (info["point_positions"][p][0] - center[0])**2 + (info["point_positions"][p][1] - center[1])**2,
                            radius**2
                        ):
                        points_on_circle.append(True)
                    else:
                        points_on_circle.append(False)
                
                if not all(points_on_circle):  # not on the same circle
                    continue
                
                # if the two points are on the circle, check if the line is a diameter
                points_distance_square = (info["point_positions"][point1][0] - info["point_positions"][point2][0])**2 + (info["point_positions"][point1][1] - info["point_positions"][point2][1])**2
                if np.isclose(points_distance_square, (2 * radius)**2):
                    valid_diameters.append((point1, point2))
                elif np.sqrt(points_distance_square) > (2 * radius) * 0.75:  # not a diameter but long enough
                    invalid_diameters.append((point1, point2))
    
    return valid_diameters, invalid_diameters


def calculate_angles_from_three_points(point1: tuple[float, float], point2: tuple[float, float], point3: tuple[float, float]) -> float:
    """ calculate angle between line point1-point2 and point2-point3. Output is a float value from 0 to 180. """
    
    # make sure that there is no numerical issues
    
    # calculate the angle
    vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
    vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
    
    cos_theta = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    
    if cos_theta > 0.999:
        angle = 0
    elif cos_theta < -0.999:
        angle = 180
    else:
        angle = np.arccos(cos_theta) * 180 / np.pi
    
    return angle


def convert_angle_to_answer(angle: float) -> Union[str, None]:
    if 5 < angle and angle < 15:
        return "a"
    elif 40 < angle and angle < 50:
        return "b"
    elif 85 < angle and angle < 95:
        return "c"
    elif 130 < angle and angle < 140:
        return "d"
    elif 175 < angle:
        return "e"
    else:
        return None


def get_angle_answer(point1: tuple[float, float], point2: tuple[float, float], point3: tuple[float, float]) -> Union[str, None]:
    angle = calculate_angles_from_three_points(
        point1, point2, point3
    )
    
    return convert_angle_to_answer(angle)


def get_angles_alpha_geometry(info: GeoInfo) -> dict[str, list[str]]:
    points_list = sorted(list(info["point_positions"].keys()))
    output_dict: dict[str, list[str]] = {}  # key: answer, value: list of angle strings (e.g., "ABC")
    
    for point_idx, three_points in enumerate(itertools.combinations(points_list, 3)):  # select three points
        permultation_seed = point_idx + len(points_list)
        permutations = random.Random(permultation_seed).sample(list(itertools.permutations(three_points)), len(three_points))
        for points_ordered in permutations:  # iterate over all permutations of the three points
            line_exists: list[bool] = []
            for idx in range(2):  # if point1-point2 and point2-point3 are on the same line, we can calculate the angle
                line_exists.append(are_points_on_the_same_line([points_ordered[idx], points_ordered[idx + 1]], info["line_instances"]))
            if all(line_exists):
                # answer is a, b, c, d, e or None
                answer = get_angle_answer(
                    info["point_positions"][points_ordered[0]],
                    info["point_positions"][points_ordered[1]],
                    info["point_positions"][points_ordered[2]]
                )
                
                if answer is None:
                    continue
                
                angle_str = "".join(points_ordered)  # e.g., "ABC"
                output_dict.setdefault(answer, []).append(angle_str)
    
    return output_dict


def get_relative_lengths_of_all_lines_alpha_geometry(info: GeoInfo) -> dict[str, list[tuple[str, str]]]:
    # convert lines
    converted_lines: list[str] = []
    for line_list in info["line_instances"]:
        line_list = sorted(line_list)
        for i in range(len(line_list) - 1):
            converted_lines.append(f"{line_list[i]}{line_list[i+1]}")
    
    pseudo_logic_form = {
        "point_positions": info["point_positions"],
        "line_instances": converted_lines,
    }
    
    return get_relative_lengths_of_all_lines(pseudo_logic_form)


def calculate_area_of_polygon(points_list: list[tuple[float, float]]) -> float:
    area = 0.0
    for i in range(len(points_list)):
        x1, y1 = points_list[i]
        x2, y2 = points_list[(i + 1) % len(points_list)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


def get_relative_areas_alpha_geometry(info: GeoInfo) -> dict[str, list[tuple[str, str]]]:
    triangles = get_all_triangles_or_quadrilaterals_alpha_geometry(info, shape_name="triangle")[0]
    quadrilaterals = get_all_triangles_or_quadrilaterals_alpha_geometry(info, shape_name="quadrilateral")[0]
    
    # calculate areas
    shapes_list = triangles + quadrilaterals
    areas_list = []
    for shape in shapes_list:
        points = [info["point_positions"][p] for p in shape]
        area = calculate_area_of_polygon(points)
        areas_list.append(area)
    
    # get relative areas
    output_dict: dict[str, list[str]] = {}
    for shape1, shape2 in itertools.combinations(shapes_list, 2):
        area1 = areas_list[shapes_list.index(shape1)]
        area2 = areas_list[shapes_list.index(shape2)]
        
        rate = area1 / area2
        if 0.8 < rate and rate < 1.2:
            output_dict.setdefault("c", []).append((shape1, shape2))
        elif 1.8 < rate and rate < 2.2:
            output_dict.setdefault("d", []).append((shape1, shape2))
            output_dict.setdefault("b", []).append((shape2, shape1))
        elif 3.5 < rate and rate < 4.5:
            output_dict.setdefault("e", []).append((shape1, shape2))
            output_dict.setdefault("a", []).append((shape2, shape1))
    
    return output_dict


class CreateAlphageometryTap(Tap):
    split: TRAIN_VAL_TEST


if __name__ == "__main__":
    args = CreateAlphageometryTap().parse_args()
    
    target_dataset_size = {
        "train": 10000,
        "val": 100,
        "test": 100,
    }
    
    alphageometry_info_dir.mkdir(parents=True, exist_ok=True)
    
    intermediate_dir_dict = {
        "train": train_intermediate_data_dir,
        "val": val_intermediate_data_dir,
        "test": test_intermediate_data_dir,
    }
    
    dataset_dir_dict = {
        "train": train_dataset_dir,
        "val": val_dataset_dir,
        "test": test_dataset_dir,
    }
    for dir in list(dataset_dir_dict.values()) + list(intermediate_dir_dict.values()):
        image_dir = dir / "images" / "SyntheticGeometry"
        image_dir.mkdir(parents=True, exist_ok=True)
    
    train_val_test = args.split
    
    information_list: list[dict] = []
    annotations_dict: dict[str, dict[str, list[dict]]] = {}
    text_version_annotations_dict: dict[str, dict[str, list[dict]]] = {}
    stats_dict: dict[str, dict[str, int]] = {}

    seed_list = {
        "val": list(range(1000)),
        "test": list(range(1000, 2000)),
        "train": list(range(2000, 50000)),
    }[train_val_test]
    for seed in tqdm(seed_list):  # TODO: change the number
        graph_path = alphageometry_graph_dir / f'alphageometry_{seed:05d}.pkl'
        if not graph_path.exists():
            continue
        
        with open(graph_path, 'rb') as f:
            graph: gh.Graph = pkl.load(f)
        
        points_list = graph.type2nodes[gh.Point]
        lines_list = graph.type2nodes[gh.Line]
        circles_list = graph.type2nodes[gh.Circle]
        
        # filter out too simple/complex objects
        if len(points_list) <= 2:  # too simple
            continue
        if len(points_list) > 20:  # too complex
            continue
        
        if len(lines_list) <= 2:  # too simple
            continue
        
        # save stats
        id_str = f"syntheticgeometry_{seed:05d}"
        stats_dict[id_str] = {
            "points": len(points_list), "lines": len(lines_list), "circles": len(circles_list)
        }
        
        ###
        # save figure
        image_paths_list: list[Path] = []
        
        # standard version
        image_output_path =  f"images/SyntheticGeometry/{id_str}.png"
        
        standard_imsize = 512 / 100
        height = standard_imsize * random.Random(seed).uniform(1, 2) * (1.5 if len(points_list) > 10 else 1)
        width = standard_imsize * random.Random(seed).uniform(1, 3) * (1.5 if len(points_list) > 10 else 1)
        try:
            save_alphageometry_figure(graph, intermediate_dir_dict[train_val_test] / image_output_path, height=height, width=width)
        except:
            continue
        image_paths_list.append(str(image_output_path))
        
        # noisy version
        image = PIL.Image.open(intermediate_dir_dict[train_val_test] / image_output_path).convert("RGB")
        
        # add noise to black and white image
        for idx, quality in enumerate([5, 2]):
            image_paths_list.append(image_output_path.replace(".png", f"noise_{idx}.jpeg"))
            image.save(intermediate_dir_dict[train_val_test] / image_paths_list[-1], format="JPEG", quality=quality)
        
        ###
        # convert graph to dict
        information = convert_graph_to_information(graph=graph, id=id_str)
        with open(alphageometry_info_dir / f'{id_str}.json', 'w') as f:
            json.dump(information, f, indent=4)
        
        ###
        # text for visual information (only used for text_ version)
        text_visual_information = convert_visual_information_to_text(information)
        
        ###
        # make annotations
        
        # triangle and quadrilateral annotation
        for split_name in ["triangle", "quadrilateral"]:
            valid_shapes, invalid_shapes = get_all_triangles_or_quadrilaterals_alpha_geometry(information, shape_name=split_name)
            for a_or_no in ["a", "no"]:  # we include negation questions
                for valid_invalid, candidate_list in [["valid", valid_shapes], ["invalid", invalid_shapes]]:
                    if len(candidate_list) == 0:
                        continue
                    
                    seed_str = f"{split_name}_{valid_invalid}_{seed}"
                    selected_shape = random.Random(seed_str).choice(candidate_list)  # randomly select one shape
                    shape_str = "".join(selected_shape)
                    
                    if split_name == "triangle":
                        question = triangle_prompt_template.format(a_or_no=a_or_no, triangle=shape_str)
                    else:
                        assert split_name == "quadrilateral"
                        question = quadrilateral_prompt_template.format(a_or_no=a_or_no, quadrilateral=shape_str)
                    
                    # negation problems
                    if valid_invalid == "valid":
                        label = "True" if a_or_no == "a" else "False"
                    else:
                        label = "False" if a_or_no == "a" else "True"
                    
                    selected_image_path = random.Random(seed_str).choice(image_paths_list)
                    q_hash = hashlib.sha256(question.encode('utf-8')).hexdigest()
                    
                    instance = get_data_instance(
                        split=train_val_test,
                        image_path=selected_image_path,
                        answer=label,
                        question=question,
                        options=["True", "False"],
                        image_category="geometry",
                        task_category=split_name,
                        dataset_name="SyntheticGeometry",
                        id_str=f"{id_str}_{split_name}_{q_hash}",
                    )
                    
                    annotations_dict.setdefault(split_name, {}).setdefault(f"{valid_invalid}-{label}", []).append(instance)
                    
                    text_version_annotations_dict.setdefault(split_name, {}).setdefault(f"{valid_invalid}-{label}", []).append(
                        add_text_visual_information_to_instance(instance, text_visual_information)
                    )
        
        # length annotation
        relative_length_dict = get_relative_lengths_of_all_lines_alpha_geometry(information)
        for original_answer, length in sorted(list(relative_length_dict.items())):
            seed_str = f"length_{original_answer}_{seed}"
            
            relative_length_lines = random.Random(seed_str).choice(length)
            
            options = ["a", "b", "c", "d", "e"]
            prompt, answer = shuffle_options(
                question=length_prompt_template.format(line1=relative_length_lines[0], line2=relative_length_lines[1]),
                answer=original_answer, options=options, question_type="single_answer"
            )
            
            selected_image_path = random.Random(seed_str).choice(image_paths_list)
            q_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()

            instance = get_data_instance(
                split=train_val_test,
                image_path=selected_image_path,
                answer=answer,
                question=prompt,
                options=options,
                image_category="geometry",
                task_category="length",
                dataset_name="SyntheticGeometry",
                id_str=f"{id_str}_length_{q_hash}",
            )

            # make a balanced dataset later
            annotations_dict.setdefault("length", {}).setdefault(original_answer, []).append(instance)
            
            text_version_annotations_dict.setdefault("length", {}).setdefault(original_answer, []).append(
                add_text_visual_information_to_instance(instance, text_visual_information)
            )
        
        # angle annotation
        angle_dict = get_angles_alpha_geometry(information)
        for original_answer, angles in sorted(list(angle_dict.items())):
            seed_str = f"angle_{original_answer}_{seed}"
            
            angle = random.Random(seed_str).choice(angles)  # randomly select one angle
            
            options = ["a", "b", "c", "d", "e"]
            prompt, answer = shuffle_options(
                question=angle_prompt_template.format(angle=angle),
                answer=original_answer, options=options, question_type="single_answer"
            )
            
            selected_image_path = random.Random(seed_str).choice(image_paths_list)
            q_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
            
            instance = get_data_instance(
                split=train_val_test,
                image_path=selected_image_path,
                answer=answer,
                question=prompt,
                options=options,
                image_category="geometry",
                task_category="angle",
                dataset_name="SyntheticGeometry",
                id_str=f"{id_str}_angle_{q_hash}",
            )
            
            # make a balanced dataset later
            annotations_dict.setdefault("angle", {}).setdefault(original_answer, []).append(instance)
            
            text_version_annotations_dict.setdefault("angle", {}).setdefault(original_answer, []).append(
                add_text_visual_information_to_instance(instance, text_visual_information)
            )
        
        # area annotation
        area_dict = get_relative_areas_alpha_geometry(information)
        for original_answer, areas in sorted(list(area_dict.items())):
            seed_str = f"area_{original_answer}_{seed}"
            
            area = random.Random(seed_str).choice(areas)  # randomly select one area
            
            options = ["a", "b", "c", "d", "e"]
            prompt, answer = shuffle_options(
                question=area_prompt_template.format(shape1="".join(area[0]), shape2="".join(area[1])),
                answer=original_answer, options=options, question_type="single_answer"
            )
            
            selected_image_path = random.Random(seed_str).choice(image_paths_list)
            q_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
            
            instance = get_data_instance(
                split=train_val_test,
                image_path=selected_image_path,
                answer=answer,
                question=prompt,
                options=options,
                image_category="geometry",
                task_category="area",
                dataset_name="SyntheticGeometry",
                id_str=f"{id_str}_area_{q_hash}",
            )
            
            # make a balanced dataset later
            annotations_dict.setdefault("area", {}).setdefault(original_answer, []).append(instance)
            
            text_version_annotations_dict.setdefault("area", {}).setdefault(original_answer, []).append(
                add_text_visual_information_to_instance(instance, text_visual_information)
            )
    
    ###
    # make balanced dataset
    data_dict_list: list[tuple[str, dict[str, dict]]] = [["", annotations_dict], ["text_", text_version_annotations_dict]]
    for data_name, data_dict in data_dict_list:
        balanced_annotations_dict: dict[str, list[dict]] = {}
        for split, split_annotations in data_dict.items():
            options_num = len(split_annotations.keys())
            target_num = target_dataset_size[train_val_test] // options_num
            
            min_num = min([len(annotations) for annotations in split_annotations.values()])
            if min_num <= target_num:
                print(f"Warning: {split} has less than {target_num} annotations (min={min_num}) for each option ({min_num * options_num} / {target_num * options_num} in total)")
                target_num = min_num
            
            for answer, answer_annotations in sorted(list(split_annotations.items())):
                balanced_annotations_dict.setdefault(split, []).extend(random.Random(f"{split}_{answer}").sample(answer_annotations, target_num))
        
        ###
        # save annotations
        save_dir = {"train": train_dataset_dir, "val": val_dataset_dir, "test": test_dataset_dir}[train_val_test]
        if data_name == "":
            save_dir = save_dir / "synthetic"
        else:
            save_dir = save_dir / "synthetic_with_text"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for split, annotations in balanced_annotations_dict.items():
            # copy images
            for annotation in annotations:
                image_path = annotation["image_path"]
                shutil.copy(intermediate_dir_dict[train_val_test] / image_path, dataset_dir_dict[train_val_test] / image_path)
            
            full_split_name = data_name + f"syntheticgeometry__{split}"

            # shuffle
            annotations = random.Random(full_split_name).sample(annotations, len(annotations))
            
            # save
            with open(save_dir / f"{full_split_name}.jsonl", "w") as f:
                for annotation in annotations:
                    f.write(json.dumps(annotation) + "\n")
