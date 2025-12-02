# Partner-Aware Hierarchical Skill Discovery for Robust Human-AI Collaboration

## Results

Below are visualizations showing **partner-conditioned skill evolution** during evaluation in two layouts: **Cramped Room** and **Coordination Ring**. For each layout, we present side-by-side GIFs comparing **HiPT** and **PASD**, highlighting differences in skill selection and adaptive behaviors.

---

### Cramped Room Evaluation

- **Skill Evolution Plot**  
![Cramped Room Skills](plots/cramped_room_skill_evolution.png)

- **Gameplay Comparison**  
<div style="display:flex; gap:20px;">
  <div>
    <p><strong>HiPT</strong></p>
    ![HiPT Cramped Room](Results/frames_output/hipt/cramped_room/hipt_cramped_room.gif)
  </div>
  <div>
    <p><strong>PASD</strong></p>
    ![PASD Cramped Room](Results/frames_output/pasd/cramped_room/pasd_cramped_room.gif)
  </div>
</div>

**Observations:**  
- PASD dynamically selects skills aligned with partner behavior, e.g., coordinating plate or soup collection efficiently.  
- HiPT shows **skill collapse**, switching skills less frequently and failing to adapt to partner strategies.  
- These visualizations highlight how PASD disentangles partner-adaptive behaviors, maintaining **coherent sequences of low-level actions**.

---

### Coordination Ring Evaluation

- **Skill Evolution Plot**  
![Coordination Ring Skills](plots/coordination_ring_skill_evolution.png)

- **Gameplay Comparison**  
<div style="display:flex; gap:20px;">
  <div>
    <p><strong>HiPT</strong></p>
    ![HiPT Coordination Ring](Results/frames_output/hipt/coordination_ring/hipt_coordination_ring.gif)
  </div>
  <div>
    <p><strong>PASD</strong></p>
    ![PASD Coordination Ring](Results/frames_output/pasd/coordination_ring/pasd_coordination_ring.gif)
  </div>
</div>

**Observations:**  
- PASD captures partner-preferred coordination patterns, e.g., clockwise or counter-clockwise movement, ensuring smooth collaboration.  
- HiPT struggles to differentiate partner strategies, resulting in delayed adaptation and occasional coordination failures.  
- Positive pairs of sub-trajectories under PASD encourage **consistent skill execution** across similar partner behaviors.

---

### Installation

* Create a new environment:

```bash
conda create -n pasd python=3.10
conda activate pasd
