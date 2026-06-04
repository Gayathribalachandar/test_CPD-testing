import pandas as pd
import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
from app_config import get_workspace_dir

class CpdEngine:
    def __init__(self, project_dir, input_dir=None, input_paths=None):
        self.project_dir = project_dir
        self.input_dir = input_dir or project_dir
        self.input_paths = dict(input_paths or {})
        self.workspace_dir = str(get_workspace_dir())
        os.makedirs(self.workspace_dir, exist_ok=True)
        self.output_dir = os.path.join(self.workspace_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.results_dir = os.path.join(self.output_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.nodes = None
        self.elements_df = None
        self.materials = {}
        self.fixed_dofs = {}  # {particle_id: [0 for x, 1 for y]}
        self.external_forces = None
        self.positions = None
        self.velocities = None
        self.initial_positions = None # NEW: To remember starting shape

    def _resolve_input_file(self, filename, explicit_path=None, input_dir=None):
        if explicit_path:
            explicit_path = os.fspath(explicit_path)
            if os.path.exists(explicit_path):
                return explicit_path

        candidate_dirs = []
        for root in (input_dir, self.input_dir, self.project_dir, self.workspace_dir):
            if not root:
                continue
            root = os.fspath(root)
            if root not in candidate_dirs:
                candidate_dirs.append(root)

        for root in candidate_dirs:
            candidate = os.path.join(root, filename)
            if os.path.exists(candidate):
                return candidate

        if explicit_path:
            return explicit_path
        if candidate_dirs:
            return os.path.join(candidate_dirs[0], filename)
        return filename

    def _setup_visualization(self):
        """Sets up the matplotlib visualization."""
        plt.ion() 
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_aspect('equal')
        self.ax.set_title("CPD Simulation")
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.grid(True)

        # Set initial plot limits
        if self.nodes is not None and len(self.nodes) > 0:
            min_x, max_x = self.nodes[:, 0].min(), self.nodes[:, 0].max()
            min_y, max_y = self.nodes[:, 1].min(), self.nodes[:, 1].max()
            pad_x = (max_x - min_x) * 0.2 if (max_x - min_x) != 0 else 1.0
            pad_y = (max_y - min_y) * 0.2 if (max_y - min_y) != 0 else 1.0
            self.ax.set_xlim(min_x - pad_x, max_x + pad_x)
            self.ax.set_ylim(min_y - pad_y, max_y + pad_y)
        else:
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)

        self.node_plot, = self.ax.plot([], [], 'o', color='blue', markersize=4, alpha=0.6)
        self.element_lines = [] 
        # Note: We won't draw every single element line every frame to save speed,
        # instead we just draw particles and fixed points.
        
        self.fixed_dof_plot, = self.ax.plot([], [], 's', color='red', markersize=6)
        self.force_quiver = self.ax.quiver([], [], [], [], color='green', scale=1.0)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _update_visualization(self):
        """Updates the plot."""
        # Update particles
        self.node_plot.set_data(self.positions[:, 0], self.positions[:, 1])

        # Update Fixed Points (Red Squares)
        fixed_coords = []
        for node_id in self.fixed_dofs.keys():
            fixed_coords.append(self.positions[node_id])
        
        if fixed_coords:
            fixed_arr = np.array(fixed_coords)
            self.fixed_dof_plot.set_data(fixed_arr[:, 0], fixed_arr[:, 1])

        # Update Plot
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def load_inputs(self, input_dir=None, input_paths=None):
        """Load CSVs and fix column name issues."""
        print("Engine: Loading inputs...")
        resolved_paths = dict(self.input_paths)
        if input_paths:
            resolved_paths.update(input_paths)
        
        # 1. LOAD PARTICLES
        nodal_path = self._resolve_input_file(
            "solver_particles.csv",
            explicit_path=resolved_paths.get("solver_particles.csv"),
            input_dir=input_dir,
        )
        if not os.path.exists(nodal_path):
            legacy = self._resolve_input_file(
                "solver_nodal.csv",
                explicit_path=resolved_paths.get("solver_nodal.csv"),
                input_dir=input_dir,
            )
            if os.path.exists(legacy):
                nodal_path = legacy
            else:
                raise FileNotFoundError("solver_particles.csv not found.")
        
        # strip() fixes the 'KeyError: n1' caused by spaces in CSV headers
        nodal_df = pd.read_csv(nodal_path, comment='#', skipinitialspace=True)
        nodal_df.columns = nodal_df.columns.str.strip() 

        self.nodes = nodal_df[['x', 'y']].to_numpy()
        self.positions = self.nodes.copy()
        self.initial_positions = self.nodes.copy() # Store resting shape
        self.velocities = np.zeros_like(self.positions)
        self.external_forces = nodal_df[['fx', 'fy']].fillna(0).to_numpy()

        # Parse BCs
        id_col = "particle_id" if "particle_id" in nodal_df.columns else "node_id"
        for index, row in nodal_df.iterrows():
            node_id = int(row[id_col])
            if pd.notna(row['ux']): self.fixed_dofs.setdefault(node_id, []).append(0)
            if pd.notna(row['uy']): self.fixed_dofs.setdefault(node_id, []).append(1)
        
        print(f"Engine: Loaded {len(self.nodes)} particles.")

        # 2. LOAD ELEMENTS (CONNECTIONS)
        elements_path = self._resolve_input_file(
            "connections.csv",
            explicit_path=resolved_paths.get("connections.csv"),
            input_dir=input_dir,
        )
        if not os.path.exists(elements_path):
            legacy = self._resolve_input_file(
                "elements.csv",
                explicit_path=resolved_paths.get("elements.csv"),
                input_dir=input_dir,
            )
            if os.path.exists(legacy):
                elements_path = legacy
        if os.path.exists(elements_path):
            self.elements_df = pd.read_csv(elements_path, comment='#', skipinitialspace=True)
            self.elements_df.columns = self.elements_df.columns.str.strip() # Fix spaces here too
            print(f"Engine: Loaded {len(self.elements_df)} connections.")
        else:
            self.elements_df = pd.DataFrame()

    def run_simulation(self, num_steps=300, dt=0.002, visualize=False):
        print("Engine: Starting simulation...")
        if visualize:
            self._setup_visualization()

        # --- PHYSICS PARAMETERS ---
        stiffness = 20.0  # Lower stiffness = less explosion risk
        damping = 2.5       # Higher damping = stable, jelly-like motion
        results_dir = self.results_dir
        os.makedirs(results_dir, exist_ok=True)

        for step in range(num_steps):
            internal_forces = np.zeros_like(self.positions)
            
            # 1. CALCULATE FORCES (The "Rubber Band" Logic)
            # We look at every connection defined in connections.csv
            if not self.elements_df.empty:
                # Get all n1, n2, n3 indices as numpy arrays for speed
                if {"p1", "p2", "p3"}.issubset(self.elements_df.columns):
                    n1 = self.elements_df['p1'].values.astype(int)
                    n2 = self.elements_df['p2'].values.astype(int)
                    n3 = self.elements_df['p3'].values.astype(int)
                else:
                    n1 = self.elements_df['n1'].values.astype(int)
                    n2 = self.elements_df['n2'].values.astype(int)
                    n3 = self.elements_df['n3'].values.astype(int)

                # Define the pairs of connected particles: (n1-n2), (n2-n3), (n3-n1)
                pairs = [(n1, n2), (n2, n3), (n3, n1)]

                for (idx_a, idx_b) in pairs:
                    # Current positions of connected pairs
                    pos_a = self.positions[idx_a]
                    pos_b = self.positions[idx_b]
                    
                    # Initial positions (Resting shape)
                    init_a = self.initial_positions[idx_a]
                    init_b = self.initial_positions[idx_b]

                    # Vector & Distance NOW
                    vec = pos_b - pos_a
                    dist = np.linalg.norm(vec, axis=1) # Length now

                    # Vector & Distance ORIGINAL
                    rest_vec = init_b - init_a
                    rest_dist = np.linalg.norm(rest_vec, axis=1) # Length originally

                    # Avoid division by zero
                    with np.errstate(divide='ignore', invalid='ignore'):
                        # Direction unit vector
                        direction = vec / dist[:, None] 
                        direction[np.isnan(direction)] = 0

                    # Hooke's Law: Force = k * (Current_Length - Original_Length)
                    extension = dist - rest_dist
                    force_mag = stiffness * extension
                    
                    # Force vector
                    force = direction * force_mag[:, None]

                    # Add forces to particles (Newton's 3rd Law: Equal & Opposite)
                    # We use np.add.at because indices can repeat
                    np.add.at(internal_forces, idx_a, force)
                    np.add.at(internal_forces, idx_b, -force)

            # 2. ADD DAMPING & EXTERNAL FORCES
            # Damping resists motion (like moving through water)
            damping_force = -damping * self.velocities
            total_forces = internal_forces + self.external_forces + damping_force
            
            # 3. APPLY ANCHORS (Fixed Particles)
            for node_id, dofs in self.fixed_dofs.items():
                total_forces[node_id, dofs] = 0
                self.velocities[node_id, dofs] = 0

            # 4. MOVE PARTICLES (Integration)
            # a = F / m (mass = 1)
            acceleration = total_forces 
            self.velocities += acceleration * dt
            self.positions += self.velocities * dt
            
            # 5. VISUALIZE
            if visualize and step % 5 == 0:
                self._update_visualization()
                print(f"Step {step}/{num_steps} complete.")
                sys.stdout.flush()
            
            # 6. WRITE FRAME FOR GUI VISUALIZATION
            frame_df = pd.DataFrame(
                {
                    "particle_id": np.arange(len(self.positions)),
                    "x": self.positions[:, 0],
                    "y": self.positions[:, 1],
                }
            )

            frame_df.to_csv(
                os.path.join(results_dir, f"step_{step:04d}.csv"),
                index=False
            )

        print("Engine: Simulation finished.")

        if visualize:
            plt.ioff()
            plt.show()        

        initial_df = pd.DataFrame(
            {
                "particle_id": np.arange(len(self.initial_positions)),
                "x": self.initial_positions[:, 0],
                "y": self.initial_positions[:, 1],
            }
        )
        final_df = pd.DataFrame(
            {
                "particle_id": np.arange(len(self.positions)),
                "x": self.positions[:, 0],
                "y": self.positions[:, 1],
            }
        )
        initial_df.to_csv(os.path.join(self.output_dir, "initial_pos.csv"), index=False)
        final_df.to_csv(os.path.join(self.output_dir, "final_pos.csv"), index=False)

def main():
    if len(sys.argv) < 2:
        print("Usage: python cpd_engine.py <project_directory_path>")
        sys.exit(1)
    
    project_dir = sys.argv[1]
    try:
        engine = CpdEngine(project_dir)
        engine.load_inputs()
        visualize = "--visualize" in sys.argv
        engine.run_simulation(visualize=visualize)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
