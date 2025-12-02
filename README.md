# Partner-Aware Hierarchical Skill Discovery for Robust Human-AI Collaboration

## Results

Below are visualizations showing **partner-conditioned skill evolution** during evaluation in two layouts: **Cramped Room** and **Coordination Ring**. For each layout, we present **side-by-side comparisons** of HiPT and PASD to illustrate differences in skill selection and adaptive behaviors.

---

### Cramped Room Evaluation

The following GIFs illustrate how a **HiPT agent (left) and a PASD agent (right)** adapt their skills when partnering with the same self-play agent in the Cramped Room layout.

| HiPT | PASD |
|------|------|
| ![HiPT Cramped Room](Results/frames_output/hipt/cramped_room/hipt_cramped_room.gif) | ![PASD Cramped Room](Results/frames_output/pasd/cramped_room/pasd_cramped_room.gif) |

**Skill evolution strips:**  

| HiPT | PASD |
|------|------|
| ![HiPT Skill Strip](Results/frames_output/hipt/cramped_room/skill_strip_hipt_cramped_room.png) | ![PASD Skill Strip](Results/frames_output/pasd/cramped_room/skill_strip_pasd_cramped_room.png) |

**Observations:**  
- PASD adapts dynamically to partner behaviors, showing diverse skill sequences.  
- HiPT exhibits fewer skill switches and less partner-aligned adaptation.  
- Skill strips highlight that PASD maintains **distinct, consistent skills** over time, while HiPT shows partial skill collapse.

---

### Coordination Ring Evaluation

Side-by-side comparison of **HiPT and PASD** in the Coordination Ring layout.

| HiPT | PASD |
|------|------|
| ![HiPT Coordination Ring](Results/frames_output/hipt/coordination_ring/hipt_coordination_ring.gif) | ![PASD Coordination Ring](Results/frames_output/pasd/coordination_ring/pasd_coordination_ring.gif) |

**Skill evolution strips:**  

| HiPT | PASD |
|------|------|
| ![HiPT Skill Strip](Results/frames_output/hipt/coordination_ring/skill_strip_hipt_coordination_ring.png) | ![PASD Skill Strip](Results/frames_output/pasd/coordination_ring/skill_strip_pasd_coordination_ring.png) |

**Observations:**  
- PASD captures partner-preferred coordination patterns and executes smooth, consistent skill sequences.  
- HiPT struggles to differentiate partner strategies, leading to delayed adaptation.  
- Skill strips demonstrate **clear partner-aware skill separation** in PASD versus partial skill overlap in HiPT.
