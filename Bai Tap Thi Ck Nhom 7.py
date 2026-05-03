import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import heapq
import copy
from collections import deque

class GraphLogic:
    def __init__(self):
        self.nodes = set()
        self.edges = []  
        self.is_directed = False

    def reset(self):
        self.nodes.clear()
        self.edges.clear()

    def add_edge(self, u, v, w):
        self.nodes.update([u, v])
        self.edges.append((u, v, w))

    def get_adjacency_list(self):
        adj = {node: [] for node in self.nodes}
        for u, v, w in self.edges:
            adj[u].append((v, w))
            if not self.is_directed:
                adj[v].append((u, w))
        return adj

    def get_adjacency_matrix(self):
        nodes_list = sorted(list(self.nodes))
        idx = {node: i for i, node in enumerate(nodes_list)}
        n = len(nodes_list)
        matrix = np.zeros((n, n))
        for u, v, w in self.edges:
            matrix[idx[u]][idx[v]] = w
            if not self.is_directed:
                matrix[idx[v]][idx[u]] = w
        return nodes_list, matrix

    def run_bfs(self, start_node):
        adj = self.get_adjacency_list()
        if start_node not in adj: return []
        visited, queue, order = {start_node}, deque([start_node]), []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v, _ in adj[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        return order

    def run_dfs(self, start_node):
        adj = self.get_adjacency_list()
        if start_node not in adj: return []
        visited, order = set(), []
        def _dfs(u):
            visited.add(u)
            order.append(u)
            for v, _ in adj[u]:
                if v not in visited: _dfs(v)
        _dfs(start_node)
        return order

    def run_dijkstra(self, start_node):
        adj = self.get_adjacency_list()
        if start_node not in adj: return {}
        distances = {node: float('inf') for node in self.nodes}
        distances[start_node] = 0
        pq = [(0, start_node)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > distances[u]: continue
            for v, w in adj[u]:
                if distances[u] + w < distances[v]:
                    distances[v] = distances[u] + w
                    heapq.heappush(pq, (distances[v], v))
        return distances

    def check_bipartite(self):
        adj = self.get_adjacency_list()
        colors = {}
        for node in self.nodes:
            if node not in colors:
                queue = deque([node])
                colors[node] = 0
                while queue:
                    u = queue.popleft()
                    for v, _ in adj[u]:
                        if v not in colors:
                            colors[v] = 1 - colors[u]
                            queue.append(v)
                        elif colors[v] == colors[u]: return False
        return True

    def run_kruskal(self):
        if self.is_directed: return [], 0
        sorted_edges = sorted(self.edges, key=lambda x: x[2])
        parent = {node: node for node in self.nodes}
        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]
        mst, total = [], 0
        for u, v, w in sorted_edges:
            root_u, root_v = find(u), find(v)
            if root_u != root_v:
                mst.append((u, v, w))
                parent[root_u] = root_v
                total += w
        return mst, total

    def run_prim(self):
        if self.is_directed or not self.nodes: return [], 0
        adj = self.get_adjacency_list()
        start_node = list(self.nodes)[0]
        mst, total, visited = [], 0, {start_node}
        edges = [(w, start_node, v) for v, w in adj[start_node]]
        heapq.heapify(edges)
        while edges and len(visited) < len(self.nodes):
            w, u, v = heapq.heappop(edges)
            if v not in visited:
                visited.add(v)
                mst.append((u, v, w))
                total += w
                for nv, nw in adj[v]:
                    if nv not in visited: heapq.heappush(edges, (nw, v, nv))
        return mst, total

    def run_ford_fulkerson(self, s, t):
        nodes_list = sorted(list(self.nodes))
        if s not in self.nodes or t not in self.nodes: return 0
        idx = {node: i for i, node in enumerate(nodes_list)}
        n = len(nodes_list)
        cap = np.zeros((n, n))
        for u, v, w in self.edges: cap[idx[u]][idx[v]] += w
        
        parent = [-1] * n
        def bfs(s_idx, t_idx):
            vis = [False] * n
            q = deque([s_idx])
            vis[s_idx] = True
            while q:
                u = q.popleft()
                for v in range(n):
                    if not vis[v] and cap[u][v] > 0:
                        q.append(v)
                        vis[v] = True
                        parent[v] = u
                        if v == t_idx: return True
            return False

        max_f, si, ti = 0, idx[s], idx[t]
        while bfs(si, ti):
            path_f = float("inf")
            curr = ti
            while curr != si:
                path_f = min(path_f, cap[parent[curr]][curr])
                curr = parent[curr]
            max_f += path_f
            v = ti
            while v != si:
                u = parent[v]
                cap[u][v] -= path_f
                cap[v][u] += path_f
                v = parent[v]
        return max_f

    def run_hierholzer(self):
        adj = {u: [v for v, w in nbs] for u, nbs in self.get_adjacency_list().items()}
        if not adj: return []
        start = list(adj.keys())[0]
        stack, path = [start], []
        while stack:
            u = stack[-1]
            if adj[u]:
                v = adj[u].pop()
                if not self.is_directed and u in adj[v]: adj[v].remove(u)
                stack.append(v)
            else: path.append(stack.pop())
        return path[::-1]

    def run_fleury(self):
        if self.is_directed: return "Fleury chỉ demo cho vô hướng"
        adj = {u: [v for v, w in nbs] for u, nbs in self.get_adjacency_list().items()}
        start = list(adj.keys())[0]
        path = []
        
        def is_bridge(u, v, local_adj):
            if len(local_adj[u]) == 1: return False
            count1 = self._dfs_count(u, local_adj)
            local_adj[u].remove(v)
            local_adj[v].remove(u)
            count2 = self._dfs_count(u, local_adj)
            local_adj[u].append(v)
            local_adj[v].append(u)
            return count1 > count2

        def solve(u, local_adj):
            for v in local_adj[u]:
                if not is_bridge(u, v, local_adj):
                    path.append((u, v))
                    local_adj[u].remove(v)
                    local_adj[v].remove(u)
                    solve(v, local_adj)
                    break
            else: 
                if local_adj[u]:
                    v = local_adj[u][0]
                    path.append((u, v))
                    local_adj[u].remove(v)
                    local_adj[v].remove(u)
                    solve(v, local_adj)
        
        solve(start, adj)
        return path

    def _dfs_count(self, u, adj):
        vis = {u}
        q = [u]
        while q:
            curr = q.pop()
            for v in adj[curr]:
                if v not in vis:
                    vis.add(v)
                    q.append(v)
        return len(vis)

class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HỆ THỐNG PHÂN TÍCH ĐỒ THỊ - NHÓM 7")
        self.root.geometry("1200x850")
        self.root.configure(bg="#f1f5f9")
        self.logic = GraphLogic()
        self._setup_ui()

    def _setup_ui(self):
        self.sidebar = tk.Frame(self.root, width=350, bg="#ffffff", padx=20, pady=20, highlightthickness=1)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)
        self.canvas_area = tk.Frame(self.root, bg="#f1f5f9")
        self.canvas_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(self.sidebar, text="NHÓM 7 - UTH", font=("Segoe UI", 18, "bold"), bg="white", fg="#2563eb").pack()
        
        input_frame = tk.LabelFrame(self.sidebar, text=" Nhập cạnh (u, v, w) ", bg="white", padx=5, pady=5)
        input_frame.pack(fill=tk.X, pady=10)
        self.is_directed_var = tk.BooleanVar(value=False)
        ttk.Radiobutton(input_frame, text="Vô hướng", variable=self.is_directed_var, value=False).pack(side=tk.LEFT)
        ttk.Radiobutton(input_frame, text="Có hướng", variable=self.is_directed_var, value=True).pack(side=tk.LEFT)
        
        self.txt_input = tk.Text(self.sidebar, height=8, font=("Consolas", 10))
        self.txt_input.pack(fill=tk.X)
        self.txt_input.insert("1.0", "A,B,2\nB,C,3\nC,D,1\nD,A,5\nA,C,4")

        ttk.Button(self.sidebar, text="VẼ ĐỒ THỊ TRỰC QUAN", command=self.on_build).pack(fill=tk.X, pady=5)

        tk.Label(self.sidebar, text="CHỌN THUẬT TOÁN:", bg="white", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        self.algo_selector = ttk.Combobox(self.sidebar, state="readonly", values=[
            "1. Chuyển đổi biểu diễn (Matrix/List)",
            "2. Duyệt BFS", "3. Duyệt DFS", "4. Dijkstra (Ngắn nhất)", 
            "5. Kiểm tra đồ thị 2 phía", "6. MST Kruskal", "7. MST Prim", 
            "8. Ford-Fulkerson (Max Flow)", "9. Fleury (Euler Path)", "10. Hierholzer (Euler Circuit)"
        ])
        self.algo_selector.pack(fill=tk.X, pady=5)
        self.algo_selector.current(1)

        tk.Label(self.sidebar, text="Đỉnh khởi đầu:", bg="white").pack(anchor=tk.W)
        self.start_node_entry = ttk.Entry(self.sidebar)
        self.start_node_entry.insert(0, "A")
        self.start_node_entry.pack(fill=tk.X, pady=5)

        ttk.Button(self.sidebar, text="CHẠY & TRỰC QUAN HÓA", command=self.on_run).pack(fill=tk.X, pady=5)

        self.txt_output = tk.Text(self.sidebar, height=12, bg="#f8fafc", state="disabled", font=("Courier New", 9))
        self.txt_output.pack(fill=tk.X, pady=10)
        ttk.Button(self.sidebar, text="LƯU DỮ LIỆU / XUẤT BÁO CÁO", command=self.on_export).pack(fill=tk.X)

    def write_log(self, text):
        self.txt_output.config(state="normal")
        self.txt_output.delete("1.0", tk.END)
        self.txt_output.insert(tk.END, f">>> KẾT QUẢ:\n{text}")
        self.txt_output.config(state="disabled")

    def on_build(self):
        self.logic.reset()
        self.logic.is_directed = self.is_directed_var.get()
        try:
            data = self.txt_input.get("1.0", tk.END).strip()
            for line in data.split('\n'):
                p = [x.strip() for x in line.split(',')]
                if len(p) == 3: self.logic.add_edge(p[0], p[1], float(p[2]))
            self.draw()
        except: messagebox.showerror("Lỗi", "Định dạng: u, v, w")

    def draw(self, highlight_edges=None):
        for w in self.canvas_area.winfo_children(): w.destroy()
        G = nx.DiGraph() if self.logic.is_directed else nx.Graph()
        for u, v, w in self.logic.edges: G.add_edge(u, v, weight=w)
        if not G.nodes: return
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='#f1f5f9')
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='#2563eb', edge_color='#94a3b8', 
                node_size=1000, font_weight='bold', font_color='white', ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), ax=ax)
        if highlight_edges:
            nx.draw_networkx_edges(G, pos, edgelist=[(u,v) for u,v,w in highlight_edges], edge_color='red', width=3, ax=ax)
        canvas = FigureCanvasTkAgg(fig, self.canvas_area)
        canvas.draw()
        NavigationToolbar2Tk(canvas, self.canvas_area).update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_run(self):
        algo = self.algo_selector.get()
        start = self.start_node_entry.get().strip()
        if not self.logic.nodes: return
        
        if "Chuyển đổi" in algo:
            nodes, matrix = self.logic.get_adjacency_matrix()
            res = f"DANH SÁCH CẠNH:\n{self.logic.edges}\n\nMA TRẬN KỀ ({', '.join(nodes)}):\n{matrix}"
            self.write_log(res)
        elif "BFS" in algo: self.write_log(f"BFS: {' -> '.join(self.logic.run_bfs(start))}")
        elif "DFS" in algo: self.write_log(f"DFS: {' -> '.join(self.logic.run_dfs(start))}")
        elif "Dijkstra" in algo:
            dists = self.logic.run_dijkstra(start)
            self.write_log("\n".join([f"Đến {k}: {v}" for k, v in dists.items()]))
        elif "phân đôi" in algo: self.write_log(f"Đồ thị 2 phía: {self.logic.check_bipartite()}")
        elif "Kruskal" in algo:
            mst, t = self.logic.run_kruskal()
            self.draw(mst); self.write_log(f"Kruskal MST - Tổng: {t}")
        elif "Prim" in algo:
            mst, t = self.logic.run_prim()
            self.draw(mst); self.write_log(f"Prim MST - Tổng: {t}")
        elif "Ford-Fulkerson" in algo:
            sink = simpledialog.askstring("Input", "Nhập đỉnh ĐÍCH (sink):")
            if sink: self.write_log(f"Max Flow ({start}->{sink}): {self.logic.run_ford_fulkerson(start, sink)}")
        elif "Fleury" in algo:
            path = self.logic.run_fleury()
            self.draw([(u, v, 0) for u, v in path]); self.write_log(f"Fleury Path: {path}")
        elif "Hierholzer" in algo:
            p = self.logic.run_hierholzer()
            self.write_log(f"Euler Circuit: {' -> '.join(p)}")

    def on_export(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"BÁO CÁO: NHÓM 7\n{'-'*20}\n{self.txt_output.get('1.0', tk.END)}")
            messagebox.showinfo("OK", "Đã lưu báo cáo!")

if __name__ == "__main__":
    root = tk.Tk(); app = GraphApp(root); root.mainloop()