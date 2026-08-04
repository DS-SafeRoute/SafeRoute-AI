"""
Safe Route 파이프라인 테스트 코드 (v2)

python3 test_pipeline.py

실행하면 pipeline_result.png 파일 생성
(1) 원본 마스크 (2) 방/복도 라벨링 (3) 스켈레톤 (4) 최종 그래프+대피경로

"""

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# 세그멘테이션 클래스 정의
WALL = 0
FLOOR = 1  # 방 내부 + 복도
DOOR = 2


# ------------------------------------------------------------
# 0. 테스트용 평면도 마스크 생성
# ------------------------------------------------------------
def create_demo_floorplan_mask(size=120):
    mask = np.full((size, size), WALL, dtype=np.uint8)

    mask[10:45, 10:45] = FLOOR          # 방 1
    mask[65:110, 10:45] = FLOOR         # 방 2
    mask[10:118, 55:65] = FLOOR         # 복도
    mask[25:30, 45:55] = DOOR           # 방1 <-> 복도
    mask[85:90, 45:55] = DOOR           # 방2 <-> 복도
    mask[118:size, 58:62] = DOOR        # 복도 <-> 출구 (도면 하단 경계에 닿음)

    # 일부러 벽에 작은 노이즈(끊김)를 넣어서 Morphology 필요성을 보여줌
    mask[57, 57:60] = FLOOR  # 복도 옆 벽에 실수로 뚫린 구멍처럼 시뮬레이션

    return mask


# ------------------------------------------------------------
# 1. 방/복도 영역 라벨링 (8-neighbor connectivity 적용)
# ------------------------------------------------------------
def label_rooms(mask):
    walkable_only = (mask == FLOOR)
    structure = np.ones((3, 3), dtype=int)  # 8-neighbor
    room_labels, num_rooms = ndimage.label(walkable_only, structure=structure)
    return room_labels, num_rooms


# ------------------------------------------------------------
# 2. 문 인스턴스별 연결 관계 계산
# ------------------------------------------------------------
def compute_door_connections(mask, room_labels):
    door_component_labels, num_doors = ndimage.label(mask == DOOR)

    door_info = {}
    for door_id in range(1, num_doors + 1):
        door_pixels = (door_component_labels == door_id)
        dilated = ndimage.binary_dilation(door_pixels, iterations=2)
        touching_room_ids = set(room_labels[dilated & (room_labels > 0)].tolist())
        centroid = ndimage.center_of_mass(door_pixels)

        rows, cols = np.where(door_pixels)
        touches_boundary = (
            rows.min() <= 1 or cols.min() <= 1
            or rows.max() >= mask.shape[0] - 2
            or cols.max() >= mask.shape[1] - 2
        )

        door_info[door_id] = {
            "centroid": centroid,
            "connects_rooms": touching_room_ids,
            "is_exit_candidate": touches_boundary,
        }

    return door_info


# ------------------------------------------------------------
# 3. Morphology 후처리 + 스켈레톤화
# ------------------------------------------------------------
def clean_and_skeletonize(mask):
    walkable = (mask == FLOOR) | (mask == DOOR)

    # closing: 작은 끊김/구멍을 메움 (벽 노이즈로 스켈레톤이 갈라지는 문제 방지)
    cleaned = ndimage.binary_closing(walkable, structure=np.ones((3, 3)))
    # opening: 튀어나온 작은 잡티 제거
    cleaned = ndimage.binary_opening(cleaned, structure=np.ones((3, 3)))
    # hole filling: 내부에 갇힌 구멍 메움
    cleaned = ndimage.binary_fill_holes(cleaned)

    skeleton = skeletonize(cleaned)
    return skeleton


# ------------------------------------------------------------
# 4. 스켈레톤 -> 그래프 변환
# ------------------------------------------------------------
def skeleton_to_graph(skeleton):
    G = nx.Graph()
    rows, cols = np.where(skeleton)
    coords = set(zip(rows.tolist(), cols.tolist()))

    for (r, c) in coords:
        G.add_node((r, c), row=float(r), col=float(c))

    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                        (0, 1), (1, -1), (1, 0), (1, 1)]
    for (r, c) in coords:
        for dr, dc in neighbor_offsets:
            neighbor = (r + dr, c + dc)
            if neighbor in coords and not G.has_edge((r, c), neighbor):
                dist = (dr ** 2 + dc ** 2) ** 0.5
                G.add_edge((r, c), neighbor, weight=dist)

    return G


def simplify_graph(G):
    """degree==2인 노드(직선 경로 중간점)를 병합해서 그래프를 단순화."""
    G = G.copy()
    changed = True
    while changed:
        changed = False
        for node in list(G.nodes):
            if node not in G:
                continue
            if G.degree(node) == 2:
                neighbors = list(G.neighbors(node))
                if len(neighbors) != 2 or neighbors[0] == neighbors[1]:
                    continue
                n1, n2 = neighbors
                merged_weight = G[node][n1]["weight"] + G[node][n2]["weight"]

                if G.has_edge(n1, n2):
                    if G[n1][n2]["weight"] > merged_weight:
                        G[n1][n2]["weight"] = merged_weight
                else:
                    G.add_edge(n1, n2, weight=merged_weight)

                G.remove_node(node)
                changed = True

    # 노드 키를 (row, col) 튜플 -> 순번 정수로 재매핑
    mapping = {node: idx for idx, node in enumerate(G.nodes)}
    G = nx.relabel_nodes(G, mapping)
    return G


# ------------------------------------------------------------
# 5. 그래프 노드에 room_id / is_door / is_exit 부여
# ------------------------------------------------------------
def annotate_graph(G, room_labels, door_info, door_match_radius=10):
    door_list = [
        (door_id, info["centroid"], info["connects_rooms"], info["is_exit_candidate"])
        for door_id, info in door_info.items()
    ]

    for node_id, data in G.nodes(data=True):
        row, col = data["row"], data["col"]
        r = int(np.clip(round(row), 0, room_labels.shape[0] - 1))
        c = int(np.clip(round(col), 0, room_labels.shape[1] - 1))
        G.nodes[node_id]["room_id"] = int(room_labels[r, c])

        nearest_door_id, nearest_dist = None, float("inf")
        for door_id, centroid, connects, is_exit in door_list:
            dist = ((row - centroid[0]) ** 2 + (col - centroid[1]) ** 2) ** 0.5
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_door_id = door_id

        if nearest_door_id is not None and nearest_dist <= door_match_radius:
            info = door_info[nearest_door_id]
            G.nodes[node_id]["is_door"] = True
            G.nodes[node_id]["is_exit"] = info["is_exit_candidate"]
        else:
            G.nodes[node_id]["is_door"] = False
            G.nodes[node_id]["is_exit"] = False

    return G


# ------------------------------------------------------------
# 6. Dijkstra로 대피 경로 계산
# ------------------------------------------------------------
def compute_evacuation_paths(G):
    exit_nodes = [n for n, d in G.nodes(data=True) if d.get("is_exit")]
    if not exit_nodes:
        raise ValueError("출구로 지정된 노드가 없습니다.")

    results = {}
    for node in G.nodes:
        best_path, best_dist = None, float("inf")
        for exit_node in exit_nodes:
            try:
                dist, path = nx.single_source_dijkstra(G, node, target=exit_node, weight="weight")
            except nx.NetworkXNoPath:
                continue
            if dist < best_dist:
                best_dist, best_path = dist, path
        results[node] = {"distance": best_dist, "path": best_path}

    return results, exit_nodes


# ------------------------------------------------------------
# 7. 결과 시각화
# ------------------------------------------------------------
def visualize(mask, room_labels, skeleton, G, results, exit_nodes, save_path="pipeline_result.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # (1) 원본 세그멘테이션 마스크
    axes[0, 0].imshow(mask, cmap="tab10", vmin=0, vmax=2)
    axes[0, 0].set_title("1. Segmentation Mask (wall / floor / door)")
    axes[0, 0].axis("off")

    # (2) 방/복도 라벨링 결과
    axes[0, 1].imshow(room_labels, cmap="tab20")
    axes[0, 1].set_title("2. Room Labeling (color = room_id)")
    axes[0, 1].axis("off")

    # (3) 스켈레톤
    axes[1, 0].imshow(mask, cmap="gray", alpha=0.3)
    ys, xs = np.where(skeleton)
    axes[1, 0].scatter(xs, ys, s=1, c="red")
    axes[1, 0].set_title("3. Skeleton (after morphology cleanup)")
    axes[1, 0].axis("off")

    # (4) 최종 그래프 + 대피 경로
    axes[1, 1].imshow(mask, cmap="gray", alpha=0.3)
    for u, v in G.edges:
        r1, c1 = G.nodes[u]["row"], G.nodes[u]["col"]
        r2, c2 = G.nodes[v]["row"], G.nodes[v]["col"]
        axes[1, 1].plot([c1, c2], [r1, r2], c="lightgray", linewidth=0.8, zorder=1)

    for node, data in G.nodes(data=True):
        color = "gray"
        size = 15
        if data.get("is_exit"):
            color, size = "red", 100
        elif data.get("is_door"):
            color, size = "orange", 60
        axes[1, 1].scatter(data["col"], data["row"], c=color, s=size, zorder=2)

    # 방1에 속한 임의의 노드 하나에서 출구까지의 경로를 하이라이트
    sample_node = next(
        (n for n, d in G.nodes(data=True) if d.get("room_id") not in (0, None)), None
    )
    if sample_node is not None:
        path = results[sample_node]["path"]
        if path:
            path_rows = [G.nodes[n]["row"] for n in path]
            path_cols = [G.nodes[n]["col"] for n in path]
            axes[1, 1].plot(path_cols, path_rows, c="blue", linewidth=2.5, zorder=3)

    legend_handles = [
        mpatches.Patch(color="red", label="Exit node"),
        mpatches.Patch(color="orange", label="Door node"),
        mpatches.Patch(color="gray", label="Regular node"),
        mpatches.Patch(color="blue", label="Computed evacuation path"),
    ]
    axes[1, 1].legend(handles=legend_handles, loc="upper right", fontsize=8)
    axes[1, 1].set_title("4. Final Graph + Evacuation Path (simplified)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"결과 이미지 저장 완료: {save_path}")


# ------------------------------------------------------------
# 실행
# ------------------------------------------------------------
def main():
    print("1. 데모 평면도 마스크 생성...")
    mask = create_demo_floorplan_mask()

    print("2. 방/복도 영역 라벨링 (8-neighbor)...")
    room_labels, num_rooms = label_rooms(mask)
    print(f"   -> {num_rooms}개의 방/복도 영역 발견")

    print("3. 문의 연결 관계 계산...")
    door_info = compute_door_connections(mask, room_labels)
    for door_id, info in door_info.items():
        print(f"   문 {door_id}: 연결된 공간 {info['connects_rooms']}, "
              f"출구 후보: {info['is_exit_candidate']}")

    print("4. Morphology 후처리 + 스켈레톤화...")
    skeleton = clean_and_skeletonize(mask)
    print(f"   -> 스켈레톤 픽셀 수: {skeleton.sum()}")

    print("5. 스켈레톤 -> 그래프 변환...")
    raw_graph = skeleton_to_graph(skeleton)
    print(f"   -> 병합 전 노드 수: {raw_graph.number_of_nodes()}")

    print("6. Graph Simplification...")
    G = simplify_graph(raw_graph)
    print(f"   -> 병합 후 노드 수: {G.number_of_nodes()} (많이 줄어들수록 정상)")

    print("7. 그래프에 room_id / is_door / is_exit 부여...")
    G = annotate_graph(G, room_labels, door_info)

    print("8. Dijkstra로 대피 경로 계산...")
    results, exit_nodes = compute_evacuation_paths(G)
    print(f"   -> 출구로 지정된 노드 수: {len(exit_nodes)}")

    print("9. 결과 시각화...")
    visualize(mask, room_labels, skeleton, G, results, exit_nodes)


if __name__ == "__main__":
    main()